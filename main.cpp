// main.cpp
#include "spsc_queue.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

// 队列满/空时的等待策略（低延迟场景可换成 _mm_pause()）
inline void spin_hint() { std::this_thread::yield(); }

// ============================================================
// 1. 基础功能：满/空判断、FIFO 顺序、环绕
// ============================================================
static void test_basic() {
    lf::SpscQueue<int, 4> q;
    assert(q.empty());
    assert(q.capacity() == 4);

    // 能存满 4 个（不牺牲槽位）
    for (int i = 0; i < 4; ++i) assert(q.push(i));
    assert(!q.push(99));                 // 满了
    assert(q.size_approx() == 4);

    int v = -1;
    for (int i = 0; i < 4; ++i) {
        assert(q.pop(v));
        assert(v == i);                  // FIFO 顺序
    }
    assert(!q.pop(v));                   // 空了
    assert(q.empty());

    // 反复环绕 1000 次，验证索引回卷正确
    for (int r = 0; r < 1000; ++r) {
        assert(q.push(r));
        assert(q.pop(v) && v == r);
    }
    puts("[ok] test_basic");
}

// ============================================================
// 2. 对象生命周期：构造/析构次数必须配平
// ============================================================
struct Tracked {
    static std::atomic<int> alive;
    int v;

    explicit Tracked(int x = 0) : v(x) { ++alive; }
    Tracked(const Tracked& o) : v(o.v) { ++alive; }
    Tracked(Tracked&& o) noexcept : v(o.v) { ++alive; }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&) noexcept = default;
    ~Tracked() { --alive; }
};
std::atomic<int> Tracked::alive{ 0 };

static void test_lifetime() {
    assert(Tracked::alive == 0);
    {
        lf::SpscQueue<Tracked, 8> q;
        // 构造队列本身不该产生任何 T 对象（裸存储，不预构造）
        assert(Tracked::alive == 0);

        for (int i = 0; i < 8; ++i) assert(q.emplace(i));   // 原地构造
        assert(Tracked::alive == 8);

        Tracked out{ -1 };                                    // +1
        for (int i = 0; i < 5; ++i) {
            assert(q.pop(out));
            assert(out.v == i);
        }
        assert(Tracked::alive == 3 + 1);                    // 队列剩 3 + out

        // 队列析构时，残留的 3 个元素要被正确析构
    }
    assert(Tracked::alive == 0);
    puts("[ok] test_lifetime (析构配平)");
}

// ============================================================
// 3. 只可移动类型（unique_ptr）——极简版做不到这一点
// ============================================================
static void test_move_only() {
    lf::SpscQueue<std::unique_ptr<int>, 8> q;

    for (int i = 0; i < 8; ++i)
        assert(q.push(std::make_unique<int>(i * 10)));

    std::unique_ptr<int> p;
    for (int i = 0; i < 8; ++i) {
        assert(q.pop(p));
        assert(p && *p == i * 10);
    }
    puts("[ok] test_move_only (unique_ptr)");
}

// ============================================================
// 4. 双线程 + 非平凡类型（std::string），校验内容与顺序
// ============================================================
static void test_concurrent_string() {
    constexpr int N = 200'000;
    auto q = std::make_unique<lf::SpscQueue<std::string, 1024>>();

    std::thread producer([&] {
        for (int i = 0; i < N; ++i) {
            std::string s = "msg-" + std::to_string(i);
            while (!q->push(std::move(s))) spin_hint();
        }
        });

    std::thread consumer([&] {
        std::string s;
        for (int i = 0; i < N; ++i) {
            while (!q->pop(s)) spin_hint();
            assert(s == "msg-" + std::to_string(i));   // 顺序 + 内容都对
        }
        });

    producer.join();
    consumer.join();
    assert(q->empty());
    printf("[ok] test_concurrent_string (%d 条)\n", N);
}

// ============================================================
// 5. consume / consume_all 零拷贝与批量消费
// ============================================================
static void test_consume_api() {
    constexpr std::uint64_t N = 1'000'000;
    auto q = std::make_unique<lf::SpscQueue<std::uint64_t, 4096>>();

    std::atomic<bool> done{ false };
    std::uint64_t sum = 0, count = 0, max_batch = 0;

    std::thread producer([&] {
        for (std::uint64_t i = 1; i <= N; ++i)
            while (!q->emplace(i)) spin_hint();
        done.store(true, std::memory_order_release);
        });

    while (count < N) {
        std::size_t n = q->consume_all([&](std::uint64_t& x) { sum += x; });
        if (n == 0) { spin_hint(); continue; }
        count += n;
        max_batch = std::max<std::uint64_t>(max_batch, n);
    }
    producer.join();
    (void)done;

    const std::uint64_t expect = N * (N + 1) / 2;
    assert(sum == expect);
    printf("[ok] test_consume_api (sum=%llu, 最大批量=%llu)\n",
        (unsigned long long)sum, (unsigned long long)max_batch);
}

// ============================================================
// 6. 性能基准：pop 逐条 vs consume_all 批量
// ============================================================
template <bool UseBatch>
static void bench(const char* label, std::uint64_t N) {
    auto q = std::make_unique<lf::SpscQueue<std::uint64_t, 8192>>();
    std::uint64_t sink = 0;

    const auto t0 = Clock::now();

    std::thread producer([&] {
        for (std::uint64_t i = 0; i < N; ++i)
            while (!q->emplace(i)) spin_hint();
        });

    std::uint64_t got = 0;
    if constexpr (UseBatch) {
        while (got < N)
            got += q->consume_all([&](std::uint64_t& x) { sink += x; });
    }
    else {
        std::uint64_t v;
        while (got < N) {
            if (q->pop(v)) { sink += v; ++got; }
        }
    }
    producer.join();

    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    printf("  %-14s %.3f s | %7.1f M ops/s | %6.1f ns/op  (checksum=%llu)\n",
        label, sec, N / sec / 1e6, sec / N * 1e9, (unsigned long long)sink);
}

// ============================================================
int main() {
    puts("=== 正确性测试 ===");
    test_basic();
    test_lifetime();
    test_move_only();
    test_concurrent_string();
    test_consume_api();

    puts("\n=== 性能基准（uint64_t, 容量 8192, 1000 万条） ===");
    constexpr std::uint64_t N = 10'000'000;
    bench<false>("pop() 逐条", N);
    bench<true>("consume_all()", N);

    puts("\n全部通过 ✅");
    return 0;
}