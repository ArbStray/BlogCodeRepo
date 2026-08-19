#include <vector>
#include <atomic>
#include <new>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <chrono>
#include <numeric>
#include <cassert>
#include <iomanip>

// 为了跨平台支持自旋暂停指令，引入相应的微指令头文件
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h> // 提供 _mm_pause()
#endif
#endif

// 鲁棒的跨平台 Fallback，万一某些编译器未定义干扰大小
#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#else
constexpr size_t hardware_destructive_interference_size = 64;
#endif

// 极轻量的自旋退避类
class SpinBackoff {
private:
    static constexpr int MAX_PASSES = 10;
    int passes_ = 1;

public:
    void pause() {
        if (passes_ <= MAX_PASSES) {
            for (int i = 0; i < passes_; ++i) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
                _mm_pause();
#else
                __builtin_ia32_pause();
#endif
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
                asm volatile("yield" ::: "memory");
#else
                std::this_thread::yield();
#endif
            }
            passes_ <<= 1;
        }
        else {
            std::this_thread::yield();
        }
    }
};

template <typename T>
class LockFreeMPMCQueue {
private:
    struct Cell {
        std::atomic<size_t> sequence;
        T data;

        Cell() = default;

        // 由于 std::atomic 的拷贝和移动构造函数都是被禁用的（deleted）
        // 含有 std::atomic 成员的 Cell 结构体会隐式禁用移动和拷贝。
        // 这会导致 std::vector::resize() 在进行模板实例化和实例化扩容保障时报错。
        // 解决方案：为 Cell 手动实现轻量级移动构造和移动赋值运算符
        Cell(Cell&& other) noexcept
            : sequence(other.sequence.load(std::memory_order_relaxed))
            , data(std::move(other.data)) {
        }

        Cell& operator=(Cell&& other) noexcept {
            if (this != &other) {
                sequence.store(other.sequence.load(std::memory_order_relaxed), std::memory_order_relaxed);
                data = std::move(other.data);
            }
            return *this;
        }

        Cell(const Cell&) = delete;
        Cell& operator=(const Cell&) = delete;
    };

    using CellBuffer = std::vector<Cell>;

    const size_t buffer_mask_;
    CellBuffer buffer_;

    // 避抗伪共享：让生产者专用的头部指针和消费者专用的尾部指针，各居一条 Cache Line，互不干扰
    alignas(hardware_destructive_interference_size) std::atomic<size_t> enqueue_pos_;
    alignas(hardware_destructive_interference_size) std::atomic<size_t> dequeue_pos_;

public:
    explicit LockFreeMPMCQueue(size_t capacity)
        : buffer_mask_(capacity - 1)
        , enqueue_pos_(0)
        , dequeue_pos_(0)
    {
        // 修正原版逻辑缺陷：容量必须大于 0，且必须是 2 的幂。
        // （原版当 capacity = 0 时，0 & (0 - 1) == 0，会绕过校验引发严重 Bug）
        if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
            throw std::invalid_argument("Capacity must be a power of 2 and greater than 0");
        }

        buffer_.resize(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    ~LockFreeMPMCQueue() = default;

    LockFreeMPMCQueue(const LockFreeMPMCQueue&) = delete;
    LockFreeMPMCQueue& operator=(const LockFreeMPMCQueue&) = delete;

    /**
     * @brief 尝试向队列压入数据
     * @return true 压入成功；false 队列已满
     */
    bool enqueue(T&& val) {
        Cell* cell;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        SpinBackoff backoff;

        for (;;) {
            cell = &buffer_[pos & buffer_mask_];
            // Acquire 语义保证我们能看到该 Cell 的 sequence 以及对应的 data 最新写入
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (dif == 0) {
                // 如果 dif == 0，说明这个坑位是空的。抢票机制开启：
                // 如果 CAS 失败，内存中最真实的最新写指针会自动写回到 pos 变量中，硬件级纠错！
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
                backoff.pause();
            }
            else if (dif < 0) {
                // 写指针绕了一圈赶上了读指针还没被取走，队列已满
                return false;
            }
            else {
                // 有人动作快把当格写完了且我们没有执过 CAS （通过上面的 if 分支 A）
                // 手动刷新本地对于写位置的猜测
                backoff.pause();
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }

        // 写入数据（由于步骤一已经拿到了独占的所有权，这里写入是线程安全的）
        cell->data = std::move(val);

        // 极重要的发布屏障：用 Release 动作更新 sequence。
        // 这行执行完，读取 seq 的消费者核心由于 acquire 同步线，必能立即看到上面的 data 覆写
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief 尝试从队列拉取数据
     * @return std::optional<T> 包含数据则成功，若为空则说明队列已空
     */
    std::optional<T> dequeue() {
        Cell* cell;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        SpinBackoff backoff;

        for (;;) {
            cell = &buffer_[pos & buffer_mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (dif == 0) {
                // 槽位的 sequence == pos + 1 说明此格已被生产者安全写毕发布。开始抢占：
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
                backoff.pause();
            }
            else if (dif < 0) {
                // 数据未就绪，队列已空
                return std::nullopt;
            }
            else {
                backoff.pause();
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }

        // 提取数据
        T ret = std::move(cell->data);

        // 将当前格子的 sequence 逻辑进度，更新为下一次这个格子可以被生产者写入的时代号 (pos + mask + 1)
        cell->sequence.store(pos + buffer_mask_ + 1, std::memory_order_release);
        return ret;
    }
};
