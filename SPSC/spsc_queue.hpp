// spsc_queue.hpp
#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace lf {

#ifdef __cpp_lib_hardware_interference_size
    constexpr std::size_t kCacheLine = std::hardware_destructive_interference_size;
#else
    constexpr std::size_t kCacheLine = 64;
#endif

    // 单生产者单消费者无锁队列。Capacity 必须是 2 的幂。
    // 约束：只允许 1 个线程调用 push/emplace，1 个线程调用 pop/consume。
    template <typename T, std::size_t Capacity>
    class SpscQueue {
        static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
            "Capacity must be a power of two and >= 2");
        static constexpr std::size_t kMask = Capacity - 1;

    public:
        using value_type = T;

        SpscQueue() = default;
        SpscQueue(const SpscQueue&) = delete;
        SpscQueue& operator=(const SpscQueue&) = delete;

        ~SpscQueue() {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                // 析构残留元素（此时假定无并发访问）
                std::size_t r = read_.load(std::memory_order_relaxed);
                std::size_t w = write_.load(std::memory_order_relaxed);
                for (; r != w; ++r) slot(r)->~T();
            }
        }

        // ---------------- 生产者侧 ----------------

        template <typename... Args>
        bool emplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
            const std::size_t w = write_.load(std::memory_order_relaxed);
            if (w - read_cache_ == Capacity) {                  // 可能满，刷新缓存再判
                read_cache_ = read_.load(std::memory_order_acquire);
                if (w - read_cache_ == Capacity) return false;   // 真的满了
            }
            ::new (static_cast<void*>(storage_ + (w & kMask) * sizeof(T)))
                T(std::forward<Args>(args)...);
            write_.store(w + 1, std::memory_order_release);      // 发布数据
            return true;
        }

        bool push(const T& v) { return emplace(v); }
        bool push(T&& v) { return emplace(std::move(v)); }

        // ---------------- 消费者侧 ----------------

        bool pop(T& out) {
            const std::size_t r = read_.load(std::memory_order_relaxed);
            if (r == write_cache_) {                             // 可能空
                write_cache_ = write_.load(std::memory_order_acquire);
                if (r == write_cache_) return false;             // 真的空
            }
            T* p = slot(r);
            out = std::move(*p);
            p->~T();
            read_.store(r + 1, std::memory_order_release);       // 释放槽位
            return true;
        }

        // 零拷贝消费：f(T&) 处理完后自动析构并推进读指针
        template <typename F>
        bool consume(F&& f) {
            const std::size_t r = read_.load(std::memory_order_relaxed);
            if (r == write_cache_) {
                write_cache_ = write_.load(std::memory_order_acquire);
                if (r == write_cache_) return false;
            }
            T* p = slot(r);
            std::forward<F>(f)(*p);
            p->~T();
            read_.store(r + 1, std::memory_order_release);
            return true;
        }

        // 批量消费，返回处理条数（减少原子操作次数，吞吐更高）
        template <typename F>
        std::size_t consume_all(F&& f) {
            std::size_t r = read_.load(std::memory_order_relaxed);
            write_cache_ = write_.load(std::memory_order_acquire);
            const std::size_t n = write_cache_ - r;
            for (std::size_t i = 0; i < n; ++i, ++r) {
                T* p = slot(r);
                f(*p);
                p->~T();
            }
            if (n) read_.store(r, std::memory_order_release);
            return n;
        }

        // ---------------- 观测（近似值） ----------------

        std::size_t size_approx() const noexcept {
            const std::size_t w = write_.load(std::memory_order_acquire);
            const std::size_t r = read_.load(std::memory_order_acquire);
            return w - r;
        }
        bool empty() const noexcept { return size_approx() == 0; }
        static constexpr std::size_t capacity() noexcept { return Capacity; }

    private:
        T* slot(std::size_t i) noexcept {
            return std::launder(reinterpret_cast<T*>(storage_ + (i & kMask) * sizeof(T)));
        }

        alignas(kCacheLine) alignas(T) unsigned char storage_[Capacity * sizeof(T)];

        // 生产者私有行：write_ 由生产者写、消费者读；read_cache_ 仅生产者用
        alignas(kCacheLine) std::atomic<std::size_t> write_{ 0 };
        std::size_t read_cache_{ 0 };

        // 消费者私有行
        alignas(kCacheLine) std::atomic<std::size_t> read_{ 0 };
        std::size_t write_cache_{ 0 };

        char pad_[kCacheLine]{};  // 防止与相邻对象共享 cache line
    };

}  // namespace lf