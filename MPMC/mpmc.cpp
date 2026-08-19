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
#include "spsc_queue.hpp"
struct TrackedElement {
    int value = 0;
    static std::atomic<int> active_instances;
    // 默认构造（队列扩容/分配时，Cell 数组会被提前默认初始化，所有槽位都会有移动前的占位符）
    TrackedElement() : value(-999) {
        ++active_instances;
    }
    // 显式值构造
    explicit TrackedElement(int val) : value(val) {
        ++active_instances;
    }
    // 严禁拷贝（确保无锁流程的高效并杜绝不必要的内存分配）
    TrackedElement(const TrackedElement&) = delete;
    TrackedElement& operator=(const TrackedElement&) = delete;
    // 移动构造
    TrackedElement(TrackedElement&& other) noexcept : value(other.value) {
        other.value = -1; // 标记墓碑值
        ++active_instances;
    }
    // 移动赋值
    TrackedElement& operator=(TrackedElement&& other) noexcept {
        if (this != &other) {
            value = other.value;
            other.value = -1; // 标记墓碑值
        }
        return *this;
    }
    ~TrackedElement() {
        --active_instances;
    }
};
std::atomic<int> TrackedElement::active_instances{ 0 };
// 单元测试函数
void run_unit_tests() {
    std::cout << "[单元测试] 开始基本功能检验..." << std::endl;
    // 1.1 2的幂级校验
    try {
        LockFreeMPMCQueue<int> invalid_q(15);
        std::cerr << "!!! 错误: 允许了非 2 幂大小的队列初始化！" << std::endl;
        assert(false);
    }
    catch (const std::invalid_argument& e) {
        std::cout << " -> 非2幂检验捕获异常成功: " << e.what() << std::endl;
    }
    try {
        LockFreeMPMCQueue<int> zero_q(0);
        std::cerr << "!!! 错误: 允许了大小为 0 的队列初始化！" << std::endl;
        assert(false);
    }
    catch (const std::invalid_argument& e) {
        std::cout << " -> 0尺寸校验捕获异常成功: " << e.what() << std::endl;
    }
    // 1.2 单线程压入与弹出（基本 FIFO 验证）
    {
        constexpr size_t CAP = 8;
        LockFreeMPMCQueue<int> queue(CAP);
        // 压入
        for (int i = 1; i <= static_cast<int>(CAP); ++i) {
            bool ok = queue.enqueue(std::move(i));
            assert(ok);
        }
        // 此时已满
        bool full_enqueue = queue.enqueue(9);
        assert(!full_enqueue);
        // 弹出
        for (int i = 1; i <= static_cast<int>(CAP); ++i) {
            auto val = queue.dequeue();
            assert(val.has_value());
            assert(*val == i);
        }
        // 此时已空
        auto empty_dequeue = queue.dequeue();
        assert(!empty_dequeue.has_value());
    }
    std::cout << " -> 基本 FIFO 与边界溢出处理通过。" << std::endl;
    // 1.3 复杂的生命周期追踪与安全析构验证
    {
        constexpr size_t CAP = 4;
        // 这一步会为4个 Cell 分配内存并调用 TrackedElement 的默认构造。
        // 此时，active_instances 应该等于 4
        assert(TrackedElement::active_instances == 0);
        {
            LockFreeMPMCQueue<TrackedElement> q(CAP);
            assert(TrackedElement::active_instances == static_cast<int>(CAP));
            // Enqueue 会移动覆盖默认构造完的占位单元
            q.enqueue(TrackedElement(10));
            q.enqueue(TrackedElement(20));
            // 局部临时对象会被销毁，此时 active_instances 依然等于静态分配下的 CAP = 4 个
            assert(TrackedElement::active_instances == static_cast<int>(CAP));
            auto val1 = q.dequeue();
            assert(val1.has_value() && val1->value == 10);

            // 当前出队的 val1 存在于当前作用域，所以堆上+栈上活跃实例数为：CAP + 1 = 5
            assert(TrackedElement::active_instances == static_cast<int>(CAP) + 1);
        }
        // 当超出作用域后，q 和 val1 均销毁，活跃实例数恢复为 0
        assert(TrackedElement::active_instances == 0);
    }
    std::cout << " -> 对象生命周期与移动流转验证通过。" << std::endl;
    std::cout << "[单元测试] 全部基本测试均获得通过！\n" << std::endl;
}
// 2. 多线程高并发压力测试兼吞吐量基准测试
void run_stress_and_benchmark(int num_producers, int num_consumers, size_t items_per_producer, size_t queue_capacity) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "[压力测试与基准性能检验]" << std::endl;
    std::cout << " - 队列容量: " << queue_capacity << std::endl;
    std::cout << " - 生产者线程数: " << num_producers << " | 每个线程压入数: " << items_per_producer << std::endl;
    std::cout << " - 消费者线程数: " << num_consumers << std::endl;
    std::cout << "==========================================================" << std::endl;
    LockFreeMPMCQueue<uint64_t> queue(queue_capacity);
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    std::atomic<int> active_producers{ num_producers };

    // 统计量
    std::vector<uint64_t> expected_sums(num_producers, 0);
    std::vector<uint64_t> actual_sums(num_consumers, 0);
    std::vector<size_t> actual_counts(num_consumers, 0);
    auto start_time = std::chrono::high_resolution_clock::now();
    // 启动生产者线程
    for (int t = 0; t < num_producers; ++t) {
        producers.emplace_back([&queue, t, items_per_producer, &expected_sums, &active_producers]() {
            // 通过偏移机制生成全局唯一的、不重合的数据序列
            uint64_t base = static_cast<uint64_t>(t) * items_per_producer * 1000;
            uint64_t sum = 0;
            for (size_t i = 0; i < items_per_producer; ++i) {
                uint64_t val = base + i;
                SpinBackoff backoff;
                // 非阻塞队列下的自旋：如果队列已满则启动轻量级退避自旋直到压入成功
                while (!queue.enqueue(std::move(val))) {
                    backoff.pause();
                }
                sum += val;
            }
            expected_sums[t] = sum;
            --active_producers;
            });
    }
    // 启动消费者线程
    for (int t = 0; t < num_consumers; ++t) {
        consumers.emplace_back([&queue, t, &actual_sums, &actual_counts, &active_producers]() {
            uint64_t sum = 0;
            size_t count = 0;
            SpinBackoff backoff;
            for (;;) {
                auto val = queue.dequeue();
                if (val.has_value()) {
                    sum += *val;
                    ++count;
                }
                else {
                    // 若弹空，首先检测是否所有生产者都已停止生成
                    if (active_producers.load(std::memory_order_relaxed) == 0) {
                        // 最后一轮双重空槽确认：再次尝试拉取，防止发生时序上的交错遗漏
                        auto final_val = queue.dequeue();
                        if (final_val.has_value()) {
                            sum += *final_val;
                            ++count;
                            continue;
                        }
                        break; // 确实没有遗留数据了，消费者退出
                    }
                    backoff.pause(); // 队列暂时空，启动指数自旋退避
                }
            }
            actual_sums[t] = sum;
            actual_counts[t] = count;
            });
    }
    // 汇合线程
    for (auto& thread : producers) {
        if (thread.joinable()) thread.join();
    }
    for (auto& thread : consumers) {
        if (thread.joinable()) thread.join();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    // 3. 数据完整性无损审计
    uint64_t total_expected_sum = std::accumulate(expected_sums.begin(), expected_sums.end(), 0ULL);
    uint64_t total_actual_sum = std::accumulate(actual_sums.begin(), actual_sums.end(), 0ULL);
    size_t total_actual_count = std::accumulate(actual_counts.begin(), actual_counts.end(), 0ULL);
    size_t total_expected_count = static_cast<size_t>(num_producers) * items_per_producer;
    std::cout << "[审计结果]" << std::endl;
    std::cout << " - 理论预期推送总量: " << total_expected_count << " 行记录" << std::endl;
    std::cout << " - 实测消费接收总量: " << total_actual_count << " 行记录" << std::endl;
    std::cout << " - 理论发送值校验和: " << total_expected_sum << std::endl;
    std::cout << " - 实测接收值校验和: " << total_actual_sum << std::endl;
    assert(total_expected_count == total_actual_count);
    assert(total_expected_sum == total_actual_sum);
    std::cout << " >>> \033[32m【正确性验证成功】\033[0m: 数据无任何丢失、重复或写乱！" << std::endl;
    // 吞吐指标输出
    double mops = static_cast<double>(total_actual_count) / elapsed.count() / 1000000.0;
    std::cout << "[性能表现]" << std::endl;
    std::cout << " - 并发吞吐耗时: " << std::fixed << std::setprecision(4) << elapsed.count() << " 秒" << std::endl;
    std::cout << " - 每秒无锁动作: " << std::fixed << std::setprecision(2) << mops << " 百万次操作 (M_ops)" << std::endl;
    std::cout << std::endl;
}
int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "     Lock-Free MPMC Queue (Vyukov's) Test & Benchmark" << std::endl;
    std::cout << "==========================================================" << std::endl;
    try {
        // 先进行单线程逻辑与生命周期校验
        run_unit_tests();
        // 接下来启动不同核数比配与高竞争模型下的基准压测
        // 场景 A：对称轻量级竞争
        run_stress_and_benchmark(2, 2, 2000000, 1024);
        // 场景 B：对称高烈度竞争（常见 4 生产者对 4 消费者）
        run_stress_and_benchmark(4, 4, 2000000, 4096);
        // 场景 C：非平衡竞争：多生产者单消费者（如高频并发日志流合并汇总入盘）
        run_stress_and_benchmark(6, 1, 1000000, 2048);
        // 场景 D：非平衡竞争：单生产者多消费者（如分布式单发大任务的多工人抢道拼手速执行）
        run_stress_and_benchmark(1, 6, 6000000, 2048);
    }
    catch (const std::exception& e) {
        std::cerr << "运行期间发生异常: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "\n>>> 所有测试均获得通过，Vyukov MPMC 无锁队列极度安全稳健！" << std::endl;
    return 0;
}