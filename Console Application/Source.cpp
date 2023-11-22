#include <cmath>
#include <vector>
#include <iostream>
#include <string>
#include <functional>
#include <algorithm>
#include <climits>
#include <cstring>
#include <bitset>
#include <cmath>
#include <stack>
#include <queue>
#include <cstring>
#include <set>
#include <array>
#include <cassert>
#include <map>
#include <iterator>
#include <iomanip>
#include <complex>
#include <unordered_set>
#include <unordered_map>
#include <cfloat>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <typeinfo>
#include <coroutine>
#include <thread>
#include <mutex>
#include <format>
#include <condition_variable>

using namespace std::complex_literals;

using namespace  std;

#define for_(var,elem,max) for (int var = elem; var < max; ++var)
#define mp(x, y) make_pair(x, y)
#define FAST_IO ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL)
#define REDIRECT_IO freopen("input.txt", "r", stdin); freopen("output.txt", "w", stdout)
#define REDIRECT_INPUT freopen("input.txt", "r", stdin)

using cd = complex<double>;
using ll = long long;
using ull = unsigned long long;
using pii = pair<int, int>;
using pll = pair<long long, long long>;
using vi = vector<int>;
using vb = vector<bool>;
using vll = vector<long long>;
using vd = vector<double>;
using vvi = vector<vector<int>>;
using vvb = vector<vector<bool>>;
using vvll = vector<vector<long long>>;
using vdi = vector<deque<int>>;
using vdll = vector<deque<long long>>;
using vpii = vector<pair<int, int>>;
using vpib = vector<pair<ll, bool>>;
using vvpii = vector<vpii>;
using vpll = vector<pair<long long, long long>>;
using vvpll = vector<vector<pair<long long, long long>>>;
using di = deque<int>;
using sti = stack<int>;

inline std::ostream& fendl(std::ostream& os)
{
    os.put(os.widen('\n'));
    return os;
}

ll binpow(ll a, ll b, ll m)
{
    ll val = a;

    ll res(1);
    while (b > 0)
    {
        if (b & 1)
            res = (res * val) % m;
        val = (val * val) % m;
        b >>= 1;
    }
    return res;
}

template <typename T>
T binpow(T a, T b)
{
    T val = a;

    T res(1);
    while (b > 0)
    {
        if (b & 1)
            res = (res * val);
        val = (val * val);
        b >>= 1;
    }
    return res;
}

template <const long long MOD>
class ModuloWrapper
{
private:
    long long int number;

    static inline auto Modulo(long long n)
    {
        return ((n % MOD) + MOD) % MOD;
    }

public:
    ModuloWrapper(long long int num = 0) : number(Modulo(num))
    {

    }

    [[nodiscard]] const ll& get() const
    {
        return number;
    }

    [[nodiscard]] ModuloWrapper inverse() const
    {
        return binpow(number, MOD - 2, MOD);
    }

    ModuloWrapper operator+(const ModuloWrapper& rhs) const
    {
        return this->number + rhs.number;
    }

    ModuloWrapper operator-(const ModuloWrapper& rhs) const
    {
        return this->number - rhs.number;
    }

    ModuloWrapper operator*(const ModuloWrapper& rhs) const
    {
        return this->number * rhs.number;
    }

    ModuloWrapper operator/(const ModuloWrapper& rhs) const
    {
        return this->number * rhs.inverse().get();
    }

    ModuloWrapper operator- () const
    {
        return ModuloWrapper(-this->number);
    }

    ModuloWrapper& operator +=(const ModuloWrapper& rhs)
    {
        this->number = Modulo(this->number + rhs.number);
        return *this;
    }

    ModuloWrapper& operator -=(const ModuloWrapper& rhs)
    {
        this->number = Modulo(this->number - rhs.number);
        return *this;
    }

    ModuloWrapper& operator *=(const ModuloWrapper& rhs)
    {
        this->number = Modulo(this->number * rhs.number);
        return *this;
    }

    ModuloWrapper& operator /=(const ModuloWrapper& rhs)
    {
        this->number = ModuloWrapper(*this / rhs).number;
        return *this;
    }

    bool operator==(const ModuloWrapper& rhs) const
    {
        return this->number == rhs.number;
    }

    bool operator!=(const ModuloWrapper& rhs) const
    {
        return !(*this == rhs);
    }

    static ModuloWrapper factorial(long long n)
    {
        ModuloWrapper ans = 1;
        while (n > 1)
        {
            ans *= n;
            n--;
        }
        return ans;
    }

    friend std::ostream& operator<< (std::ostream& out, const ModuloWrapper& num)
    {
        out << num.number;
        return out;
    }

    friend std::istream& operator>> (std::istream& in, ModuloWrapper& mw)
    {
        in >> mw.number;

        mw.number = Modulo(mw.number);

        return in;
    }
};

namespace FFT
{
    const double PI = acos(-1);
    const int threshold = 200;

    long long ntt_primitive_root(int p)
    {
        int MOD = p;
        int r = 2, pw = 0, phi = --p;
        while (binpow(r, p >> 1, MOD) == 1) ++r;
        assert(binpow(r, p, MOD) == 1);         // a ^ phi(p) == 1, will be always true
        while (!(p & 1)) p >>= 1, pw++;                 // extracts the odd part of prime - 1
        return binpow(r, phi >> pw, MOD);
    }

    template <const int MOD, class T>
    void FFT(vector<T>& a, bool invert = false)
    {
        using MW = ModuloWrapper<MOD>;
        static_assert((is_same<MW, T>::value && MOD > 0) || (is_same<cd, T>::value && MOD == 0),
            "MOD must be 0 (for complex double) or must match for modulo wrapper");

        // always work with even powers of 2
        assert((a.size() & (a.size() - 1)) == 0);
        T root, root_1;
        ll root_pw = 0;

        if constexpr (MOD > 0)
        {
            root = MW(ntt_primitive_root(MOD));
            root_1 = MW(root).inverse();
            const auto temp = MOD - 1;
            root_pw = temp & ~(temp & (temp - 1));
        }

        int n = a.size();

        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;

            if (i < j)
                swap(a[i], a[j]);
        }

        for (int len = 2; len <= n; len <<= 1)
        {
            T wlen;
            if constexpr (MOD > 0)
            {
                wlen = invert ? root_1 : root;
                for (int i = len; i < root_pw; i <<= 1)
                    wlen *= wlen;
            }
            else
            {
                double ang = 2 * PI / len * (invert ? -1 : 1);
                wlen = cd(cos(ang), sin(ang));
            }


            for (int i = 0; i < n; i += len) {
                T w = 1;

                for (int j = 0; j < len / 2; j++) {
                    T u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        if (!invert)
            return;

        T n_1;
        if constexpr (MOD > 0)
            n_1 = MW(n).inverse();
        else
            n_1 = 1.0 / (double)n;

        for (auto& x : a)
            x *= n_1;
    }

    template<class T>
    void multiply_slow(vector<T>& a, const vector<T>& b)
    {
        if (a.empty() || b.empty())
        {
            a.clear();
            return;
        }

        int n = a.size();
        int m = b.size();
        a.resize(n + m - 1);
        for (int k = n + m - 2; k >= 0; k--)
        {
            a[k] *= b[0];
            for (int j = max(k - n + 1, 1); j < min(k + 1, m); j++)
                a[k] += a[k - j] * b[j];
        }
    }

    // if T = U (under is_same trait), then a and b will become empty.
    // else, a and b will remain as it is as the copy of contents is made.
    // Never assume a and b to be same just to be safe
    template <const int MOD, class T, class U>
    vector<U> multiply(vector<T>& a, vector<T>& b)
    {
        vector<U> fa, fb;

        if constexpr (is_same<T, U>::value)
            fa = std::move(a), fb = std::move(b);
        else
            fa = vector<U>(a.begin(), a.end()), fb = vector<U>(b.begin(), b.end());

        int n = 1;
        while (n < fa.size() + fb.size())
            n <<= 1;
        fa.resize(n);
        fb.resize(n);

        if (n <= threshold)
        {
            multiply_slow(fa, fb);
            return fa;
        }

        FFT<MOD>(fa, false);
        FFT<MOD>(fb, false);
        for (int i = 0; i < n; i++)
            fa[i] *= fb[i];
        FFT<MOD>(fa, true);

        return fa;
    }

    template <const int MOD, class T>
    vector<T> multiply(vector<T>& a, vector<T>& b)
    {
        using MW = ModuloWrapper<MOD>;
        static_assert(
            (MOD > 0 && is_same<MW, T>::value) ||
            (MOD == 0 && is_same<cd, T>::value) ||
            is_integral<T>::value,
            "(MOD must be 0 (for complex double) or must match for modulo wrapper) or an integer");

        if constexpr (MOD > 0)
        {
            if constexpr (is_same<MW, T>::value)
                return multiply<MOD, T, MW>(a, b);
            else
            {
                auto res = multiply<MOD, T, MW>(a, b);
                vector<T> ans(res.size());
                for (int i = 0; i < res.size(); ++i) ans[i] = res[i].get();
                return ans;
            }
        }
        else
        {
            auto res = multiply<MOD, T, cd>(a, b);
            if constexpr (is_same<cd, T>::value)
                return res;
            else
            {
                vector<T> result(res.size());
                for (int i = 0; i < res.size(); i++)
                    result[i] = round(res[i].real());
                return result;
            }
        }
    }

    // 0.48s on cses for n=10^5. 1840ms for same size on codeforces
    // a[0] != 0
    template <const int MOD, class T>
    void inverse_recursive(vector<T>& a, const int k)
    {
        using MW = ModuloWrapper<MOD>;

        // Step 0: Compute mod x^k
        a.resize(k);

        if (k == 1)
        {
            if constexpr (MOD > 0)
                a[0] = MW(a[0]).inverse();
            else
                a[0] = 1.0 / a[0];
            return;
        }

        // Step 1: Find A(-x)
        auto a_minus = a;
        for (int i = 1; i < k; i += 2)
            a_minus[i] *= -1;

        // Step 2: Compute A(x) * A(-x)
        auto a_minus_copy = a_minus;
        auto b = multiply<MOD>(a, a_minus_copy);

        // Step 3: Reduce b
        for (size_t i = 0; 2 * i < b.size(); ++i)
            b[i] = b[2 * i];

        // Step 4: Compute B^-1(x) mod x^(floor(k/2))
        inverse_recursive<MOD>(b, (k + 1) >> 1);

        // Step 5: Expand B^-1(x)
        b.resize(k);
        for (int i = ((k + 1) >> 1) - 1; i >= 0; --i)
            b[2 * i] = b[i];

        for (int i = 1; i < k; i += 2)
            b[i] = 0;

        // Step 6: Multiply A(-x) with B^-1(x)
        a = multiply<MOD>(a_minus, b);
        a.resize(k);
    }

    // 0.42s on cses for n=10^5. 1700 ms for same size on codeforces
    // a[0] != 0
    template <const int MOD, class T>
    void inverse_iterative(vector<T>& a, int k)
    {
        using MW = ModuloWrapper<MOD>;

        // Step 0: B0 = a0^-1
        vector<T> b(1);

        if constexpr (MOD > 0)
            b[0] = MW(a[0]).inverse();
        else
            b[0] = 1.0 / a[0];

        int n = 1;
        while (n < k)
            n <<= 1;

        k = 1;
        while (k < n)
        {
            // 1. Compute C := A * B mod x^2k
            // 2. Compute B := B * (2 - C) mod x^2k
            k <<= 1;
            auto a_copy = a;
            auto b_copy = b;
            auto c = multiply<MOD>(a_copy, b_copy);
            c.resize(k);

            for (int i = 0; i < k; ++i)
                c[i] = -c[i];
            c[0] += 2;

            b = multiply<MOD>(b, c);
            b.resize(k);
        }

        a = std::move(b);
    }

    // Final size of vector = initial size - 1. Final size will always be >= 1.
    template <class T>
    void derivative(vector<T>& a, int k)
    {
        a.resize(k + 1, T(0));

        for (int i = 0; i < k; ++i)
            a[i] = T(i + 1) * a[i + 1];

        a.pop_back();
    }

    // Final size of vector = initial size + 1. constant term will always be zero
    template <class T>
    void integral(vector<T>& a, int k)
    {
        a.resize(k + 1, T(0));
        for (int i = a.size() - 1; i > 0; --i)
            a[i] = a[i - 1] / T(i);
        a[0] = 0;
    }

    // a[0] != 0
    template <const int MOD, class T>
    void logrithm(vector<T>& a, int k)
    {
        // d/dx(lnP) = P'/P => lnP = integral(P'/P)
        auto copy = a;
        derivative(copy, k);
        inverse_recursive<MOD, T>(a, k);
        a.resize(k);
        a = multiply<MOD, T>(copy, a);
        integral(a, k);
        a.resize(min(a.size(), (ull)k));
    }

    // a[0] should be 0, as it makes no sense for MOD to have equivalent e
    template <const int MOD, class T>
    void exponent(vector<T>& a, int k)
    {
        assert(a[0] == T(0));
        vector<T> q(1, T(1));

        int n = k;
        while ((n & (n - 1)) != 0)
            n &= (n - 1);
        n <<= 1;

        if ((k & (k - 1)) == 0)
            n >>= 1;

        int _k = 1;
        a.resize(n, T(0));
        a[0] = T(1);

        while (_k < n)
        {
            // 1. Compute C := (1 + P) - lnQ mod x^2k
            // 2. Compute Q := Q * C mod x^2k
            _k <<= 1;
            auto q_copy = q;
            logrithm<MOD, T>(q_copy, _k);

            auto c = a;
            for (int i = 0; i < q_copy.size(); ++i)
                c[i] -= q_copy[i];

            q = multiply<MOD, T>(q, c);
            q.resize(_k);
        }

        a = std::move(q);
        a.resize(k);
    }
}

namespace DS
{
    class DSU
    {
        vector<int> parent;
        vector<int> size;
        vector<int> start;
        vector<int> end;
        const int n;

        void make_set(int v) {
            parent[v] = v;
            size[v] = 1;
            start[v] = v;
            end[v] = v;
        }

    public:
        int n_comps;
        explicit DSU(int n) : n{ n }, parent(n), size(n), start(n), end(n)
        {
            for (int i = 0; i < n; ++i)
                make_set(i);
            n_comps = n;
        }

        int find_set(int v) {
            if (v == parent[v])
                return v;
            return parent[v] = find_set(parent[v]);
        }

        void union_sets(int a, int b)
        {
            a = find_set(a);
            b = find_set(b);

            if (a == b)
                return;

            if (size[a] < size[b])
                swap(a, b);
            parent[b] = a;
            size[a] += size[b];
            end[a] = end[b];
            n_comps--;
        }

        int get_size(int a)
        {
            return size[find_set(a)];
        }

        int get_start(int a)
        {
            return start[find_set(a)];
        }

        int get_end(int a)
        {
            return end[find_set(a)];
        }
    };
}

const ll mod = 1'000'000'000 + 7;
using MW = ModuloWrapper<mod>;
using vMW = vector<MW>;

template <typename T>
class generator
{
public:
    struct promise_type
    {
        T value{};
        generator get_return_object() { return coroutine_handle<promise_type>::from_promise(*this); }

        std::suspend_always yield_value(T t) { value = t; return {}; }

        void return_void() { }
        void unhandled_exception() { }

        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
    };

    struct sentinel {};

    struct iterator
    {
        coroutine_handle<promise_type> handle;
        bool operator!=(sentinel) const { return !handle.done(); }
        iterator& operator++() { handle.resume(); return *this; }
        const T& operator*() const { return handle.promise().value; }
    };

    iterator begin() { handle.resume(); return iterator{ handle }; }
    sentinel end() { return {}; }

private:
    using handle_type = coroutine_handle<promise_type>;

    handle_type handle;

    generator(handle_type h) : handle(std::move(h)) { }
    auto& promise() { return handle.promise(); }
};

template <typename T>
class threadsafe_queue
{
    std::queue<T> queue;
    std::mutex mutex_;
    std::condition_variable cv;

public:
    void push(T val)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        queue.push(std::move(val));
        cv.notify_one();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue.empty())
            cv.wait(lock);

        auto res = std::move(queue.front());
        queue.pop();
        return std::move(res);
    }

    bool is_available()
    {
        return !queue.empty();
    }
};

template <typename T, bool is_io_task = false>
class task
{
    using value_type = std::conditional_t<std::is_void_v<T>, void*, T>;
    class final_awaiter
    {
    public:
        inline bool await_ready() noexcept
        {
            return false;
        }

        template <typename PROMISE>
        coroutine_handle<> await_suspend(coroutine_handle<PROMISE> h) noexcept
        {
            auto recursive_info = h.promise().recursive_info;
            assert(recursive_info->back().first == h.address());

            // Top is what we are returning from
            if (recursive_info->size() == 1 || recursive_info->back().second)
            {
                recursive_info->pop_back();
                return std::noop_coroutine();
            }

            recursive_info->pop_back();
            return coroutine_handle<>::from_address(recursive_info->back().first);
        }

        void await_resume() noexcept
        {

        }
    };

    friend class EventLoop;

public:
    class promise_type
    {
        template <typename U, bool>
        friend class task;
        value_type value{};
        bool is_latest_value = false;
        shared_ptr<vector<pair<void*, bool>>> recursive_info;
    public:
        task get_return_object() noexcept { return task{ coroutine_handle<promise_type>::from_promise(*this) }; }

        suspend_always initial_suspend() noexcept
        {
            // recursive_info will be overwritten if I am not the first in chain of calls
            // Useful only if I am the first call in chain and eventloop calls me, as no one else can start me after suspend
            recursive_info = make_shared<vector<pair<void*, bool>>>();
            recursive_info->push_back({ coroutine_handle<promise_type>::from_promise(*this).address(), is_io_task });
            return {};
        }
        
        final_awaiter final_suspend() noexcept { return {}; }

        final_awaiter yield_value(value_type&& t)
        {
            static_assert(!std::is_void_v<T>, "Cannot yield a value for a void task");
            value = std::move(t);
            is_latest_value = true;
            return {};
        }

        final_awaiter yield_value(value_type& t)
        {
            static_assert(!std::is_void_v<T>, "Cannot yield a value for a void task");
            value = t;
            is_latest_value = true;
            return {};
        }

        void return_void() noexcept { }

        void unhandled_exception() noexcept {
            try
            {
                std::rethrow_exception(std::current_exception());
            }
            catch (const std::exception& e)
            {
                std::cerr << "Caught exception: '" << e.what() << "'\n";
            }
            std::terminate();
        }
    };

    task(task&& t) noexcept : coro_(std::exchange(t.coro_, {}))
    {

    }

    ~task()
    {
        if (coro_)
            coro_.destroy();
    }

    bool await_ready() noexcept
    {
        return false;
    }

    template <typename U>
    coroutine_handle<> await_suspend(coroutine_handle<U> previous) noexcept
    {
        auto& previous_promise = previous.promise();
        auto& cur_promise = coro_.promise();

        void* prev_addr = previous.address();
        void* cur_addr = coro_.address();
        cur_promise.recursive_info = previous_promise.recursive_info;
        cur_promise.recursive_info->push_back({ cur_addr, is_io_task });

        if constexpr (is_io_task)
            return std::noop_coroutine();
        return coro_;
    }

    T await_resume()
    {
        if constexpr (is_void_v<T>)
            return;
        else
        {
            if (!coro_.promise().is_latest_value)
                throw std::runtime_error(std::format(
                    "Callee returned without yielding anything. Last yielded value was {}.", coro_.promise().value));

            coro_.promise().is_latest_value = false;
            return coro_.promise().value;
        }
    }

private:
    pair<coroutine_handle<>, bool> get_handle_to_resume()
    {
        auto& info = coro_.promise().recursive_info;
        if (info->empty())
            return { coro_, is_io_task };

        return { coroutine_handle<>::from_address(info->back().first), info->back().second };
    }

    coroutine_handle<promise_type> coro_;
    explicit task(coroutine_handle<promise_type> h) noexcept
        : coro_(h)
    {}
};

class EventLoop
{
    threadsafe_queue<task<void>> queue_for_loop;
    threadsafe_queue<task<void>> queue_for_io;

    EventLoop() {}
    EventLoop(const EventLoop&) = delete;
    EventLoop& operator=(const EventLoop&) = delete;
    EventLoop(EventLoop&&) = delete;
    EventLoop& operator=(EventLoop&&) = delete;

public:
    static EventLoop& get_instance()
    {
        static EventLoop instance;
        return instance;
    }

    void run()
    {
        while (true)
        {
            task<void> _task = std::move(queue_for_loop.pop());  // blocks if no more task is available
            auto [target_coroutine, is_io] = _task.get_handle_to_resume();

            if (target_coroutine.done())
                continue;

            target_coroutine.resume();

            queue_for_loop.push(std::move(_task));

            continue;
            if (_task.get_handle_to_resume().second)
                queue_for_io.push(std::move(_task));    // perform the task in thread pool
            else
                queue_for_loop.push(std::move(_task));
        }
    }

    void schedule_loop_task(task<void>&& task)
    {
        queue_for_loop.push(std::move(task));
    }
};

task<int, true> hoo(int x, bool third)
{
	co_yield x + 2;
}

task<int> goo(int x, bool third)
{
    co_yield x + 1;
    co_yield co_await hoo(x, third);
    co_yield x + 3;
}

task<void> foo(int x, bool third = false)
{
    auto res = goo(x, third);
    cout << (co_await res) << endl;
    cout << (co_await res) << endl;
    cout << (co_await res) << endl;
    co_return;
}

int main()
{
    auto& el = EventLoop::get_instance();
    el.schedule_loop_task(foo(0));
    el.schedule_loop_task(foo(10));
    el.run();
}