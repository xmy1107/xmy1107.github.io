---
title: min25筛
date: 2024-12-19 13:33:54
mathjax: true
tags:
    - acm
    - 数学
---
前面部分引用自[OI wiki](https://oi-wiki.org/math/number-theory/min-25/)

# 定义

从此种筛法的思想方法来说，其又被称为「Extended Eratosthenes Sieve」。

由于其由 [Min\_25](http://min-25.hatenablog.com/) 发明并最早开始使用，故称「Min\_25 筛」。

# 性质

其可以在 $O\left(\frac{n^{\frac{3}{4}}}{\log{n}}\right)$ 或 $\Theta\left(n^{1 - \epsilon}\right)$ 的时间复杂度下解决一类 **积性函数** 的前缀和问题。

要求：$f(p)$ 是一个关于 $p$ 可以快速求值的完全积性函数之和（例如多项式）；$f(p^{c})$ 可以快速求值。

# 记号

-   **如无特别说明，本节中所有记为 $p$ 的变量的取值集合均为全体质数。**
-   $x / y := \left\lfloor\frac{x}{y}\right\rfloor$
-   $\operatorname{isprime}(n) := [ |\{d : d \mid n\}| = 2 ]$，即 $n$ 为质数时其值为 $1$，否则为 $0$。
-   $p_{k}$：全体质数中第 $k$ 小的质数（如：$p_{1} = 2, p_{2} = 3$）。特别地，令 $p_{0} = 1$。
-   $\operatorname{lpf}(n) := [1 < n] \min\{p : p \mid n\} + [1 = n]$，即 $n$ 的最小质因数。特别地，$n=1$ 时，其值为 $1$。
-   $F_{\mathrm{prime}}(n) := \sum_{2 \le p \le n} f(p)$
-   $F_{k}(n) := \sum_{i = 2}^{n} [p_{k} \le \operatorname{lpf}(i)] f(i)$

# 解释

观察 $F_{k}(n)$ 的定义，可以发现答案即为 $F_{1}(n) + f(1) = F_{1}(n) + 1$。

***

考虑如何求出 $F_{k}(n)$。通过枚举每个 $i$ 的最小质因子及其次数可以得到递推式：

$$
\begin{aligned}
	F_{k}(n)
	&= \sum_{i = 2}^{n} [p_{k} \le \operatorname{lpf}(i)] f(i) \\\\
	&= \sum_{\substack{k \le i \\ p_{i}^{2} \le n}} \sum_{\substack{c \ge 1 \\ p_{i}^{c} \le n}} f\left(p_{i}^{c}\right) ([c > 1] + F_{i + 1}\left(n / p_{i}^{c}\right)) + \sum_{\substack{k \le i \\ p_{i} \le n}} f(p_{i}) \\\\
	&= \sum_{\substack{k \le i \\ p_{i}^{2} \le n}} \sum_{\substack{c \ge 1 \\ p_{i}^{c} \le n}} f\left(p_{i}^{c}\right) ([c > 1] + F_{i + 1}\left(n / p_{i}^{c}\right)) + F_{\mathrm{prime}}(n) - F_{\mathrm{prime}}(p_{k - 1}) \\\\
	&= \sum_{\substack{k \le i \\ p_{i}^{2} \le n}} \sum_{\substack{c \ge 1 \\ p_{i}^{c + 1} \le n}} \left(f\left(p_{i}^{c}\right) F_{i + 1}\left(n / p_{i}^{c}\right) + f\left(p_{i}^{c + 1}\right)\right) + F_{\mathrm{prime}}(n) - F_{\mathrm{prime}}(p_{k - 1})
\end{aligned}
$$

最后一步推导基于这样一个事实：对于满足 $p_{i}^{c} \le n < p_{i}^{c + 1}$ 的 $c$，有 $p_{i}^{c + 1} > n \iff n / p_{i}^{c} < p_{i} < p_{i + 1}$，故 $F_{i + 1}\left(n / p_{i}^{c}\right) = 0$。  
其边界值即为 $F_{k}(n) = 0 (p_{k} > n)$。

假设现在已经求出了所有的 $F_{\mathrm{prime}}(n)$，那么有两种方式可以求出所有的 $F_{k}(n)$：

1.  直接按照递推式计算。
2.  从大到小枚举 $p$ 转移，仅当 $p^{2} < n$ 时转移增加值不为零，故按照递推式后缀和优化即可。

***

现在考虑如何计算 $F_{\mathrm{prime}}{(n)}$。  
观察求 $F_{k}(n)$ 的过程，容易发现 $F_{\mathrm{prime}}$ 有且仅有 $1, 2, \dots, \left\lfloor\sqrt{n}\right\rfloor, n / \sqrt{n}, \dots, n / 2, n$ 这 $O(\sqrt{n})$ 处的点值是有用的。  
一般情况下，$f(p)$ 是一个关于 $p$ 的低次多项式，可以表示为 $f(p) = \sum a_{i} p^{c_{i}}$。  
那么对于每个 $p^{c_{i}}$，其对 $F_{\mathrm{prime}}(n)$ 的贡献即为 $a_{i} \sum_{2 \le p \le n} p^{c_{i}}$。  
分开考虑每个 $p^{c_{i}}$ 的贡献，问题就转变为了：给定 $n, s, g(p) = p^{s}$，对所有的 $m = n / i$，求 $\sum_{p \le m} g(p)$。

> Notice：$g(p) = p^{s}$ 是完全积性函数！

于是设 $G_{k}(n) := \sum_{i = 2}^{n} \left[p_{k} < \operatorname{lpf}(i) \lor \operatorname{isprime}(i)\right] g(i)$，即埃筛第 $k$ 轮筛完后剩下的数的 $g$ 值之和。  
对于一个合数 $x \le n$，必定有 $\operatorname{lpf}(x) \le \sqrt{x} \le \sqrt{n}$。设 $p_{\ell(n)}$ 为不大于 $\sqrt{n}$ 的最大质数，则 $F_{\mathrm{prime}}(n) = G_{\ell(n)}(n)$，即在埃筛进行 $\ell$ 轮之后剩下的均为质数。
考虑 $G$ 的边界值，显然为 $G_{0}(n) = \sum_{i = 2}^{n} g(i)$。（还记得吗？特别约定了 $p_{0} = 1$）  
对于转移，考虑埃筛的过程，分开讨论每部分的贡献，有：

1.  对于 $n < p_{k}^{2}$ 的部分，$G$ 值不变，即 $G_{k}(n) = G_{k - 1}(n)$。
2.  对于 $p_{k}^{2} \le n$ 的部分，被筛掉的数必有质因子 $p_{k}$，即 $-g(p_{k}) G_{k - 1}(n / p_{k})$。
3.  对于第二部分，由于 $p_{k}^{2} \le n \iff p_{k} \le n / p_{k}$，满足 $\operatorname{lpf}(i) < p_{k}$ 的 $i$ 会被额外减去。这部分应当加回来，即 $g(p_{k}) G_{k - 1}(p_{k - 1})$。

则有：

$$
G_{k}(n) = G_{k - 1}(n) - \left[p_{k}^{2} \le n\right] g(p_{k}) (G_{k - 1}(n / p_{k}) - G_{k - 1}(p_{k - 1}))
$$

***

# 复杂度分析

对于 $F_{k}(n)$ 的计算，其第一种方法的时间复杂度被证明为 $O\left(n^{1 - \epsilon}\right)$（见 zzt 集训队论文 2.3）；  
对于第二种方法，其本质即为洲阁筛的第二部分，在洲阁论文中也有提及（6.5.4），其时间复杂度被证明为 $O\left(\frac{n^{\frac{3}{4}}}{\log{n}}\right)$。

对于 $F_{\mathrm{prime}}(n)$ 的计算，事实上，其实现与洲阁筛第一部分是相同的。  
考虑对于每个 $m = n / i$，只有在枚举满足 $p_{k}^{2} \le m$ 的 $p_{k}$ 转移时会对时间复杂度产生贡献，则时间复杂度可估计为：

$$
\begin{aligned}
	T(n)
	&= \sum_{i^{2} \le n} O\left(\pi\left(\sqrt{i}\right)\right) + \sum_{i^{2} \le n} O\left(\pi\left(\sqrt{\frac{n}{i}}\right)\right) \\\\
	&= \sum_{i^{2} \le n} O\left(\frac{\sqrt{i}}{\ln{\sqrt{i}}}\right) + \sum_{i^{2} \le n} O\left(\frac{\sqrt{\frac{n}{i}}}{\ln{\sqrt{\frac{n}{i}}}}\right) \\\\
	&= O\left(\int_{1}^{\sqrt{n}} \frac{\sqrt{\frac{n}{x}}}{\log{\sqrt{\frac{n}{x}}}} \mathrm{d} x\right) \\\\
	&= O\left(\frac{n^{\frac{3}{4}}}{\log{n}}\right)
\end{aligned}
$$

对于空间复杂度，可以发现不论是 $F_{k}$ 还是 $F_{\mathrm{prime}}$，其均只在 $n / i$ 处取有效点值，共 $O(\sqrt{n})$ 个，仅记录有效值即可将空间复杂度优化至 $O(\sqrt{n})$。

首先，通过一次数论分块可以得到所有的有效值，用一个大小为 $O(\sqrt{n})$ 的数组 $\text{lis}$ 记录。对于有效值 $v$，记 $\text{id}(v)$ 为 $v$ 在 $\text{lis}$ 中的下标，易得：对于所有有效值 $v$，$\text{id}(v) \le \sqrt{n}$。

然后分开考虑小于等于 $\sqrt{n}$ 的有效值和大于 $\sqrt{n}$ 的有效值：对于小于等于 $\sqrt{n}$ 的有效值 $v$，用一个数组 $le$ 记录其 $id(v)$，即 $le_v = \text{id}(v)$；对于大于 $\sqrt{n}$ 的有效值 $v$，用一个数组 $\text{ge}$ 记录 $\text{id}(v)$，由于 $v$ 过大所以借助 $v' = n / v < \sqrt{n}$ 记录 $\text{id}(v)$，即 $\text{ge}_{v'} = \text{id}(v)$。

这样，就可以使用两个大小为 $O(\sqrt{n})$ 的数组记录所有有效值的 $\text{id}$ 并 $O(1)$ 查询。在计算 $F_{k}$ 或 $F_{\mathrm{prime}}$ 时，使用有效值的 $\text{id}$ 代替有效值作为下标，即可将空间复杂度优化至 $O(\sqrt{n})$。

# 过程

对于 $F_{k}(n)$ 的计算，我们实现时一般选择实现难度较低的第一种方法，其在数据规模较小时往往比第二种方法的表现要好；

对于 $F_{\mathrm{prime}}(n)$ 的计算，直接按递推式实现即可。

对于 $p_{k}^{2} \le n$，可以用线性筛预处理出 $s_{k} := F_{\mathrm{prime}}(p_{k})$ 来替代 $F_{k}$ 递推式中的 $F_{\mathrm{prime}}(p_{k - 1})$。  
相应地，$G$ 递推式中的 $G_{k - 1}(p_{k - 1}) = \sum_{i = 1}^{k - 1} g(p_{i})$ 也可以用此方法预处理。

用 Extended Eratosthenes Sieve 求 **积性函数**  $f$ 的前缀和时，应当明确以下几点：

-   如何快速（一般是线性时间复杂度）筛出前 $\sqrt{n}$ 个 $f$ 值；
-   $f(p)$ 的多项式表示；
-   如何快速求出 $f(p^{c})$。

明确上述几点之后按顺序实现以下几部分即可：

1.  筛出 $[1, \sqrt{n}]$ 内的质数与前 $\sqrt{n}$ 个 $f$ 值；
2.  对 $f(p)$ 多项式表示中的每一项筛出对应的 $G$，合并得到 $F_{\mathrm{prime}}$ 的所有 $O(\sqrt{n})$ 个有用点值；
3.  按照 $F_{k}$ 的递推式实现递归，求出 $F_{1}(n)$。

# 例题

## 求莫比乌斯函数的前缀和

求 $\displaystyle \sum_{i = 1}^{n} \mu(i)$。

易知 $f(p) = -1$。则 $g(p) = -1, G_{0}(n) = \sum_{i = 2}^{n} g(i) = -n + 1$。  
直接筛即可得到 $F_{\mathrm{prime}}$ 的所有 $O(\sqrt{n})$ 个所需点值。

## 求欧拉函数的前缀和

求 $\displaystyle \sum_{i = 1}^{n} \varphi(i)$。

首先易知 $f(p) = p - 1$。  
对于 $f(p)$ 的一次项 $(p)$，有 $g(p) = p, G_{0}(n) = \sum_{i = 2}^{n} g(i) = \frac{(n + 2) (n - 1)}{2}$；  
对于 $f(p)$ 的常数项 $(-1)$，有 $g(p) = -1, G_{0}(n) = \sum_{i = 2}^{n} g(i) = -n + 1$。  
筛两次加起来即可得到 $F_{\mathrm{prime}}$ 的所有 $O(\sqrt{n})$ 个所需点值。

## [「LOJ #6053」简单的函数](https://loj.ac/problem/6053)

给定 $f(n)$：

$$
f(n) = \begin{cases}
    1 & n = 1 \\\\
    p \operatorname{xor} c & n = p^{c} \\\\
    f(a)f(b) & n = ab \land a \perp b
\end{cases}
$$

易知 $f(p) = p - 1 + 2[p = 2]$。则按照筛 $\varphi$ 的方法筛，对 $2$ 讨论一下即可。  
此处给出一种 C++ 实现（也是我的min25筛模板）：

```cpp
/* 「LOJ #6053」简单的函数 */
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int maxs = 200000, mod = 1000000007, inv2 = (mod+1)>>1;  // 2sqrt(n)
int pri[maxs / 7], lpf[maxs + 1], spri[maxs + 1], pcnt, lim, G[maxs + 1][2], Fprime[maxs + 1], cnt;
ll lis[maxs + 1], global_n;

#define sqr(x) (1ll*(x)*(x))
#define idx(v) (v <= lim ? le[v] : ge[global_n / v])  // n/i的值为v时，i在数论分块中是第几块 
int le[maxs + 1],  // x <= \sqrt{n}
	ge[maxs + 1];  // x > \sqrt{n}
	
int sum(int x,int y){x+=y;if(x>=mod)x-=mod;return x;}
int sub(int x,int y){x-=y;if(x<0)x+=mod;return x;}
void inc(int &x,int y){x+=y;if(x>=mod)x-=mod;}
void dec(int &x,int y){x-=y;if(x<0)x+=mod;}
void sieve(int n) {
	for (int i = 2; i <= n; ++i) {
		if (lpf[i] == 0) {  // 记录质数
			lpf[i] = ++pcnt; // i的最小质因数编号 
			pri[lpf[i]] = i; // 第i个质数 
			spri[pcnt] = sum(spri[pcnt - 1], i);  // 前缀和
		}
		for (int j = 1, v; j <= lpf[i] && (v = i * pri[j]) <= n; ++j) lpf[v] = j;
	}
}

void init(ll n) {
	for (ll i = 1, j, v; i <= n; i = n / j + 1) {
		j = n / i;
		v = j % mod;
		lis[++cnt] = j;
		idx(j) = cnt;
		G[cnt][0] = sub(v, 1ll); // \sum_2^v i^0
		G[cnt][1] = 1ll * (v + 2ll) * (v - 1ll) % mod * inv2 % mod; // \sum_2^v i^1
	}
}

void calcFprime() {//求Fprime
	for (int k = 1; k <= pcnt; ++k) {
		int p = pri[k];
		ll sqrp = sqr(p);
		for (int i = 1; lis[i] >= sqrp; ++i) {
			ll v = lis[i] / p;
			int id = idx(v);
			dec(G[i][0], sub(G[id][0], k - 1));
			dec(G[i][1], 1ll* p * sub(G[id][1], spri[k - 1]) % mod); // g(p_k)=p, G_{k-1}(p_{k-1})=sum_pri
		}
	}
	/* F_prime = G_1 - G_0 */
	for (int i = 1; i <= cnt; ++i) Fprime[i] = sub(G[i][1], G[i][0]); // F(p)=p-1
}

int f_p(int p, int c) {
	/* f(p^{c}) = p xor c */
	return p xor c;
}

int F(int k, ll n) {//已知Fprime求F
	if (n < pri[k] || n <= 1) return 0;
	int id = idx(n);
	ll ans = Fprime[id] - (spri[k - 1] - (k - 1)); // Fprime(n)-Fprime(p_{k-1})
	if (k == 1) ans += 2;
	for (int i = k; i <= pcnt && sqr(pri[i]) <= n; ++i) {
		ll pw = pri[i], pw2 = sqr(pw);
		for (int c = 1; pw2 <= n; ++c, pw = pw2, pw2 *= pri[i])
			ans += (1ll * f_p(pri[i], c) * F(i + 1, n / pw) + f_p(pri[i], c + 1)) % mod;
	}
	return ans % mod;
}

int main() {
	scanf("%lld", &global_n);
	lim = sqrt(global_n);  // 上限
	sieve(lim + 1000);  // 预处理
	init(global_n);
	calcFprime();
	printf("%lld\n", (F(1, global_n) + 1ll + mod) % mod); // F(1)单独加上 
}
```

## [ICPC2024昆明 F. Flowers](https://contest.ucup.ac/contest/1871/problem/9867)

转化后题意：给定$n,p$，求$\prod_{i=1}^n f(i)\ mod\ p$，其中 $f(i)$ 为 $i$ 不同质因子个数

### 方法1（min25+拉插+CRT，麻烦）：

令 $g(n)=x^{f(n)}$，则 $g$ 为积性函数。

由于 $f(n)$ 的值不会超过 $10$ ，故枚举 $x=2...10$ ，依次求出 $F(x)\ = \ \sum_{i=1}^n g_x(i)$。

$F(x)$ 可表示为 $a_0+a_1x+a_2x^2+...+a_{10}x^{10}$，通过拉格朗日插值求出 $a_0...a_{10}$ 后，即可求出答案：

$\Pi_{i=1}^n f(i) = \Pi_{x=2}^{10} x^{a_x}$，通过这个式子也可以发现$a_0,a_1$不需要求。

因为拉格朗日插值中会遇到很大的数，所以需要对两个大质数分别取模后CRT合并。

```cpp
#include<bits/stdc++.h>
using namespace std;
#define pb push_back
#define int long long
const int maxs = 200000,M1=998244353,M2=1e9+7;
int pri[maxs / 7], lpf[maxs + 1], spri[maxs + 1], pcnt;

void sieve(const int &n) {
      for (int i = 2; i <= n; ++i) {
        if (lpf[i] == 0) {  // 记录质数
          lpf[i] = ++pcnt;
          pri[lpf[i]] = i;
          spri[pcnt] = spri[pcnt - 1]+i;  // 前缀和
        }
        for (int j = 1, v; j <= lpf[i] && (v = i * pri[j]) <= n; ++j) lpf[v] = j;
      }
}

long long global_n;
int lim;
int le[maxs + 1],  // x <= \sqrt{n}
    ge[maxs + 1];  // x > \sqrt{n}
#define idx(v) (v <= lim ? le[v] : ge[global_n / v])

int G[maxs + 1][2], Fprime[maxs + 1];
long long lis[maxs + 1];
int cnt;

void init(const long long &n,int x) {
      for (long long i = 1, j, v; i <= n; i = n / j + 1) {
        j = n / i;
        v = j;
        lis[++cnt] = j;
        (j <= lim ? le[j] : ge[global_n / j]) = cnt;
		G[cnt][0]=(v-1)*x;//tip1
      }
}

void calcFprime(int x) {
      for (int k = 1; k <= pcnt; ++k) {
        const int p = pri[k];
        const long long sqrp = p*p;
        for (int i = 1; lis[i] >= sqrp; ++i) {
          const long long v = lis[i] / p;
          const int id = idx(v);
            G[i][0]-=G[id][0]-(k-1)*x;//tip2
        }
      }
      for (int i = 1; i <= cnt; ++i) Fprime[i] = G[i][0];//tip3
}

int f_p(const int &p, const int &c,int x) {
        return x;
}

int F(const int &k, const long long &n,int x) {
      if (n < pri[k] || n <= 1) return 0;
      const int id = idx(n);
      long long ans = Fprime[id] - (k-1)*x;//tip4
      for (int i = k; i <= pcnt && pri[i]*pri[i] <= n; ++i) {
        long long pw = pri[i], pw2 =pw*pw;
        for (int c = 1; pw2 <= n; ++c, pw = pw2, pw2 *= pri[i])
          ans +=
          ((long long)f_p(pri[i], c,x) * F(i + 1, n / pw,x) + f_p(pri[i], c + 1,x));
      }
  return ans;
}

int mod;
int inv(int k,int MOD) {
  int res = 1;
  for (int e = MOD - 2; e; e /= 2) {
    if (e & 1) res = res * k % MOD;
    k = k * k % MOD;
  }
  return res;
}
int Pw(int k,int y,int MOD) {
  int res = 1;
  for (int e = y; e; e /= 2) {
    if (e & 1) res = res * k % MOD;
    k = k * k % MOD;
  }
  return res;
}
std::vector<int> lagrange_interpolation(const std::vector<int> &x,
                                        const std::vector<int> &y,int MOD) {
  const int n = x.size();
  std::vector<int> M(n + 1), xx(n), f(n);
  M[0] = 1;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j >= 0; --j) {
      M[j + 1] = (M[j] + M[j + 1]) % MOD;
      M[j] = M[j] * (MOD - x[i]) % MOD;
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    for (int j = 0; j < n; ++j) {
      xx[j] = (xx[j] * x[j] + M[i + 1] * (i + 1)) % MOD;
    }
  }
  for (int i = 0; i < n; ++i) {
    int t = y[i]%MOD * inv(xx[i],MOD) % MOD, k = M[n];
    for (int j = n - 1; j >= 0; --j) {
      f[j] = (f[j] + k * t) % MOD;
      k = (M[j] + k * x[i]) % MOD;
    }
  }
  return f;
}
void exgcd(int a, int b, int& x, int& y) {
      if (b == 0) {
            x = 1, y = 0;
            return;
      }
      exgcd(b, a % b, y, x);
      y -= a / b * x;
}
signed main() {
      scanf("%lld%lld", &global_n,&mod);
      lim = sqrt(global_n);  // 上限
      sieve(lim + 1000);  // 预处理
      vector<int>A,B;
      A.pb(0),B.pb(0);
      A.pb(1),B.pb(global_n);
      for (int x=2;x<=10;x++){
            cnt=0;
            init(global_n,x);
            calcFprime(x);
            B.pb(F(1, global_n,x) + x);
            A.pb(x);
        }
        auto f1 = lagrange_interpolation(A,B,M1);
        auto f2 = lagrange_interpolation(A,B,M2);
        vector<int>f;
        int M=M1*M2;
        for (int i=0;i<f1.size();i++){
			int b,y;
            exgcd(M2, M1, b, y);
            int tmp=(__int128)f1[i]*M2*b%M;//处理爆long long 
            exgcd(M1, M2, b, y);
            tmp=((tmp+(__int128)f2[i]*M1*b)%M+M)%M;//模完才是算好的值，才能模mod-1 
            f.pb(tmp);
        }
        int ans=1;
        for (int i=2;i<f.size();i++) ans=(ans*Pw(A[i],f[i],mod))%mod;
        cout<<ans<<endl;
}
```

### 方法2（需充分理解min25，巧妙）：

观察转移 $\sum_{\substack{k \le i \\ p_{i}^{2} \le n}} \sum_{\substack{c \ge 1 \\ p_{i}^{c + 1} \le n}} \left(f\left(p_{i}^{c}\right) F_{i + 1}\left(n / p_{i}^{c}\right) + f\left(p_{i}^{c + 1}\right)\right) + F_{\mathrm{prime}}(n) - F_{\mathrm{prime}}(p_{k - 1})$

发现包含质数的个数已经在转移里一目了然了

$F_{\mathrm{prime}}(n) - F_{\mathrm{prime}}(p_{k - 1})$ 对应：

```cpp
ans[w + 1] += Fprime[id] - (k - 1);
```

$f\left(p_{i}^{c}\right) F_{i + 1}\left(n / p_{i}^{c}\right) + f\left(p_{i}^{c + 1}\right)$ 对应：

```cpp
ans[w + 1]++;
F(i + 1, n / pw, w + 1);
```

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int maxs = 200000, mod = 1000000007, inv2 = (mod+1)>>1;  // 2sqrt(n)
int pri[maxs / 7], lpf[maxs + 1], spri[maxs + 1], pcnt, lim, G[maxs + 1][2], Fprime[maxs + 1], cnt;
ll lis[maxs + 1], global_n;

ll ans[99];

#define sqr(x) (1ll*(x)*(x))
#define idx(v) (v <= lim ? le[v] : ge[global_n / v])  // n/i的值为v时，i在数论分块中是第几块 
int le[maxs + 1],  // x <= \sqrt{n}
	ge[maxs + 1];  // x > \sqrt{n}
	
int sum(int x,int y){x+=y;if(x>=mod)x-=mod;return x;}
int sub(int x,int y){x-=y;if(x<0)x+=mod;return x;}
void inc(int &x,int y){x+=y;if(x>=mod)x-=mod;}
void dec(int &x,int y){x-=y;if(x<0)x+=mod;}
void sieve(int n) {
	for (int i = 2; i <= n; ++i) {
		if (lpf[i] == 0) {  // 记录质数
			lpf[i] = ++pcnt; // i的最小质因数编号 
			pri[lpf[i]] = i; // 第i个质数 
			spri[pcnt] = sum(spri[pcnt - 1], i);  // 前缀和
		}
		for (int j = 1, v; j <= lpf[i] && (v = i * pri[j]) <= n; ++j) lpf[v] = j;
	}
}

void init(ll n) {
	for (ll i = 1, j, v; i <= n; i = n / j + 1) {
		j = n / i;
		v = j % mod;
		lis[++cnt] = j;
		idx(j) = cnt;
		G[cnt][0] = sub(v, 1ll); // \sum_2^v i^0
	}
}

void calcFprime() {//求Fprime
	for (int k = 1; k <= pcnt; ++k) {
		int p = pri[k];
		ll sqrp = sqr(p);
		for (int i = 1; lis[i] >= sqrp; ++i) {
			ll v = lis[i] / p;
			int id = idx(v);
			dec(G[i][0], sub(G[id][0], k - 1));
		}
	}
	for (int i = 1; i <= cnt; ++i) Fprime[i] = G[i][0];
}

void F(int k, ll n, int w) {//已知Fprime求F
	if (n < pri[k] || n <= 1) return;
	int id = idx(n);
	ans[w + 1] += Fprime[id] - (k - 1);
	for (int i = k; i <= pcnt && sqr(pri[i]) <= n; ++i) {
		ll pw = pri[i], pw2 = sqr(pw);
		for (int c = 1; pw2 <= n; ++c, pw = pw2, pw2 *= pri[i]){
			ans[w + 1]++;
			F(i + 1, n / pw, w + 1);
		}
//			ans += (1ll * f_p(pri[i], c) * F(i + 1, n / pw) + f_p(pri[i], c + 1)) % mod;
	}
}

int M;
int pw(int x, ll y, int M){
	int z = 1;
	for (; y; y >>= 1, x = 1ll * x * x % M)
		if (y & 1) z = 1ll * z * x % M;
	return z;
}
int main() {
	scanf("%lld%d", &global_n, &M);
	lim = sqrt(global_n);  // 上限
	sieve(lim + 1000);  // 预处理
	init(global_n);
	calcFprime();
	F(1, global_n, 0);
	int pi = 1;
	for (int i = 2; i <= 10; i++) pi = 1ll * pi * pw(i, ans[i], M) % M;
	cout << pi << '\n';
}
```

## [a*b problem（我出的题）](https://acm.hdu.edu.cn/showproblem.php?pid=7448)

题意：$x_1^{t_1}x_2^{t_2}...x_k^{t_k}\leq n$ 且 $gcd(x_1,x_2,...,x_k)=1$ ，求有序数对 $(x_1,x_2,...,x_k)$ 数量，答案对1e9+7取模

令$f(n)$表示$x_1^{t_1}x_2^{t_2}...x_k^{t_k}=n$ 且 $gcd(x_1,x_2,...,x_k)=1$ 数对数量，$F(n)=\sum_{i=1}^{n} f(i)$，则答案为$F(n)$

$gcd(x_1,x_2,...,x_k)=1$等价于每个质因子在$x_1,x_2,...,x_k$中次数的最小值都是 $0$ ，易知各个质因子之间独立，$f$为积性函数，问题在于$f(p^c)$怎么求

### 方法1（我的，背包+dfs+组合数+min25）

$c>1$时，$f(p^c)$等价于将$c$划分给$k$个数，第 $i$ 个数必须是 $t_i$ 的倍数，且$k$个数里至少有一个为$0$的方案数

用至多不超过$33^k$的时间 $dfs$（$n=10^{10}$，$p^c$ 中c的取值小于等于33）预处理，可以得出结果

$c=1$时，$f(p^c)=num_1$，其中$num_1$是$t_1,...,t_k$中1的数量，p只有1次，只能是1的倍数

此时发现$f(p)$为关于$p$的零次多项式，$f(p^c)$可快速求，故答案可用min25筛快速求得

时间复杂度$O(\frac{n^\frac{3}{4}}{logn}+dfs)$，此时在$k\leq 8$情况下都没有问题，$k$更大的时候复杂度瓶颈的 $dfs$ 需要优化

考虑枚举某个集合的元素均为0，剩余部分形如 $2w+2x+3y+4z=c（w,x,y,z\geq 1）$

如果已知 $2x+3y+4z$ 在每个c处的方案数，那么可以$O(maxc^2)$的时间内推出$2w+2x+3y+4z=c$在每个c处的方案数

发现对于值相同的$t_i$，性质是一样的，可以直接用$t_i$这个值有几个来代替，不需要$2^k$枚举，加上一些剪枝，可以通过$k<=60$的数据

我的代码：

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int maxs = 200000,mod = 1e9+7,maxc=33,maxk=1e5+5;
void inc(int &x, int y) {
	x += y;
	(mod <= x) && (x -= mod);
}
void dec(int &x, int y) {
	x -= y;
	(x < 0) && (x += mod);
}
int sub(int x, int y) {
	return x < y ? x - y + mod : (x - y);
}
int div2(int x) {
	return ((x & 1) ? x + mod : x) >> 1;
}
ll sqrll(int x){return 1ll*x*x;}
int pw(int x,int y){int z=1;for(;y;y>>=1,x=1ll*x*x%mod)if(y&1)z=1ll*z*x%mod;return z;}
int pri[maxs / 7], lpf[maxs + 1], pcnt;

void sieve(const int &n) {
	for (int i = 2; i <= n; ++i) {
		if (lpf[i] == 0) {	// 记录质数
			lpf[i] = ++pcnt;
			pri[lpf[i]] = i;
		}
		for (int j = 1, v; j <= lpf[i] && (v = i * pri[j]) <= n; ++j) lpf[v] = j;
	}
}

typedef vector<int> V;
map<V,int>mp;
queue<V>q;
const int TMP=400000;//上限239496 
int fun[TMP][maxc+3];//fun[i][j]，id为i的表达式和为j的方案数，如2x+2y+3z=10 
int sum[TMP];//sum为表达式左边有几个数，必须小于K
int Sum[TMP];//Sum为当前所有数的和，必须<=maxc 
int f[maxc+3];//F[c]表示f(p^c)的答案 
ll global_n;
int lim;
int le[maxs + 1],	// x <= \sqrt{n}
		ge[maxs + 1];	// x > \sqrt{n}
#define idx(v) (v <= lim ? le[v] : ge[global_n / v])

V num(maxc+1);
int G[maxs + 1], Fprime[maxs + 1],fac[maxk],ifac[maxk];
ll lis[maxs + 1];
int cnt;
int K,t[102];

void init(ll n) {
	for (ll i = 1, j, v; i <= n; i = n / j + 1) {
		j = n / i;
		v = j % mod;
		lis[++cnt] = j;
		(j <= lim ? le[j] : ge[global_n / j]) = cnt;
		G[cnt]=sub(v, 1ll);
	}
}

void calcFprime() {
	for (int k = 1; k <= pcnt; ++k) {
		int p = pri[k];
		ll sqrp = sqrll(p);
		for (int i = 1; lis[i] >= sqrp; ++i) {
			ll v = lis[i] / p;
			int id = idx(v);
			dec(G[i], sub(G[id], (k - 1)));
		}
	}
	for (int i = 1; i <= cnt; ++i) Fprime[i]=G[i]*num[1];
}
int tot;
void bfs(){
	V u(maxc+1);
	mp[u]=++tot;
	sum[1]=0;
	Sum[1]=0;
	fun[1][0]=1;
	q.push(u);
	while (!q.empty()){
		u=q.front(),q.pop();
		int id=mp[u];
		if (sum[id]+1==K) break;
		for (int i=1;i<=maxc && Sum[id]+i<=maxc;i++)
			if (u[i]<num[i]){
				u[i]++;
				if (!mp.count(u)){
					mp[u]=++tot;
					sum[tot]=sum[id]+1;
					Sum[tot]=Sum[id]+i;
					for (int j=i;j<=maxc;j++)
						fun[tot][j]=(fun[id][j-i]+fun[tot][j-i])%mod;
//					for (int j=0;j<=maxc;j++)
//						for (int k=1;k*i<=j;k++) inc(fun[tot][j],fun[id][j-k*i]);
					int tmp=1;
					for (int j=1;j<=maxc;j++) tmp=1ll*tmp*fac[num[j]]%mod*ifac[u[j]]%mod*ifac[num[j]-u[j]]%mod;
					for (int j=0;j<=maxc;j++)
						inc(f[j],1ll*fun[tot][j]*tmp%mod);
					q.push(u);
				}
				u[i]--;
			}
	}
}
int f_p(int p, int c) {return f[c];}

int F(const int &k, const long long &n) {
	if (n < pri[k] || n <= 1) return 0;
	int id = idx(n);
	ll ans = Fprime[id] - (k-1)*num[1];
	
	for (int i = k; i <= pcnt && sqrll(pri[i]) <= n; ++i) {
		ll pw = pri[i], pw2 = sqrll(pw);
		for (int c = 1; pw2 <= n; ++c, pw = pw2, pw2 *= pri[i])
			ans +=(1ll*f_p(pri[i], c) * F(i + 1, n / pw) + f_p(pri[i], c + 1)) % mod;
	}
	return ans % mod;
}
int main() {
	scanf("%lld%d", &global_n,&K);
	fac[0]=1;
	for (int i=1;i<=K;i++) fac[i]=1ll*fac[i-1]*i%mod;
	ifac[K]=pw(fac[K],mod-2);
	for (int i=K;i;i--) ifac[i-1]=1ll*ifac[i]*i%mod;
	for (int i=1;i<=K;i++) scanf("%d",&t[i]),num[t[i]]++;
	bfs();
	lim = sqrt(global_n);	// 上限
	sieve(lim + 1000);	// 预处理
	init(global_n);
	calcFprime();
	printf("%lld\n", (F(1, global_n) + 1ll + mod) % mod);
}
```



### 方法2（更优秀，背包+组合数+min25）

由前文可知，关键在于预处理 $f$。$f(p^c)$等价于将$c$划分给$k$个数，

第 $i$ 个数必须是 $t_i$ 的倍数，且$k$个数里至少有一个为 $0$ 的方案数。

事实上，计算所有方案减去 $k$ 个数都不是 $0$ 的方案即可。

jiangly代码：

```cpp
#include <bits/stdc++.h>

using u32 = unsigned;
using i64 = long long;
using u64 = unsigned long long;

// TODO: Dynamic ModInt

template<typename T>
constexpr T power(T a, u64 b) {
    T res {1};
    for (; b != 0; b /= 2, a *= a) {
        if (b % 2 == 1) {
            res *= a;
        }
    }
    return res;
}

template<u32 P>
constexpr u32 mulMod(u32 a, u32 b) {
    return 1ULL * a * b % P;
}

template<u64 P>
constexpr u64 mulMod(u64 a, u64 b) {
    u64 res = a * b - u64(1.L * a * b / P - 0.5L) * P;
    res %= P;
    return res;
}

template<typename U, U P>
requires std::unsigned_integral<U>
struct ModIntBase {
public:
    constexpr ModIntBase() : x {0} {}
    
    template<typename T>
    requires std::integral<T>
    constexpr ModIntBase(T x_) : x {norm(x_ % T {P})} {}
    
    constexpr static U norm(U x) {
        if ((x >> (8 * sizeof(U) - 1) & 1) == 1) {
            x += P;
        }
        if (x >= P) {
            x -= P;
        }
        return x;
    }
    
    constexpr U val() const {
        return x;
    }
    
    constexpr ModIntBase operator-() const {
        ModIntBase res;
        res.x = norm(P - x);
        return res;
    }
    
    constexpr ModIntBase inv() const {
        return power(*this, P - 2);
    }
    
    constexpr ModIntBase &operator*=(const ModIntBase &rhs) & {
        x = mulMod<P>(x, rhs.val());
        return *this;
    }
    
    constexpr ModIntBase &operator+=(const ModIntBase &rhs) & {
        x = norm(x + rhs.x);
        return *this;
    }
    
    constexpr ModIntBase &operator-=(const ModIntBase &rhs) & {
        x = norm(x - rhs.x);
        return *this;
    }
    
    constexpr ModIntBase &operator/=(const ModIntBase &rhs) & {
        return *this *= rhs.inv();
    }
    
    friend constexpr ModIntBase operator*(ModIntBase lhs, const ModIntBase &rhs) {
        lhs *= rhs;
        return lhs;
    }
    
    friend constexpr ModIntBase operator+(ModIntBase lhs, const ModIntBase &rhs) {
        lhs += rhs;
        return lhs;
    }
    
    friend constexpr ModIntBase operator-(ModIntBase lhs, const ModIntBase &rhs) {
        lhs -= rhs;
        return lhs;
    }
    
    friend constexpr ModIntBase operator/(ModIntBase lhs, const ModIntBase &rhs) {
        lhs /= rhs;
        return lhs;
    }
    
    friend constexpr std::ostream &operator<<(std::ostream &os, const ModIntBase &a) {
        return os << a.val();
    }
    
    friend constexpr bool operator==(ModIntBase lhs, ModIntBase rhs) {
        return lhs.val() == rhs.val();
    }
    
    friend constexpr bool operator!=(ModIntBase lhs, ModIntBase rhs) {
        return lhs.val() != rhs.val();
    }
    
    friend constexpr bool operator<(ModIntBase lhs, ModIntBase rhs) {
        return lhs.val() < rhs.val();
    }
    
private:
    U x;
};

template<u32 P>
using ModInt = ModIntBase<u32, P>;

template<u64 P>
using ModInt64 = ModIntBase<u64, P>;

constexpr u32 P = 1000000007;
using Z = ModInt<P>;

constexpr int K = 33;

struct Comb {
    int n;
    std::vector<Z> _fac;
    std::vector<Z> _invfac;
    std::vector<Z> _inv;
    
    Comb() : n{0}, _fac{1}, _invfac{1}, _inv{0} {}
    Comb(int n) : Comb() {
        init(n);
    }
    
    void init(int m) {
        m = std::min<i64>(m, P - 1);
        if (m <= n) return;
        _fac.resize(m + 1);
        _invfac.resize(m + 1);
        _inv.resize(m + 1);
        
        for (int i = n + 1; i <= m; i++) {
            _fac[i] = _fac[i - 1] * i;
        }
        _invfac[m] = _fac[m].inv();
        for (int i = m; i > n; i--) {
            _invfac[i - 1] = _invfac[i] * i;
            _inv[i] = _invfac[i] * _fac[i - 1];
        }
        n = m;
    }
    
    Z fac(int m) {
        if (m > n) init(2 * m);
        return _fac[m];
    }
    Z invfac(int m) {
        if (m > n) init(2 * m);
        return _invfac[m];
    }
    Z inv(int m) {
        if (m > n) init(2 * m);
        return _inv[m];
    }
    Z binom(int n, int m) {
        if (n < m || m < 0) return 0;
        return fac(n) * invfac(m) * invfac(n - m);
    }
} comb;
std::vector<int> minp, primes;

void sieve(int n) {
    minp.assign(n + 1, 0);
    primes.clear();
    
    for (int i = 2; i <= n; i++) {
        if (minp[i] == 0) {
            minp[i] = i;
            primes.push_back(i);
        }
        
        for (auto p : primes) {
            if (i * p > n) {
                break;
            }
            minp[i * p] = p;
            if (p == minp[i]) {
                break;
            }
        }
    }
}

void solve() {
    i64 n;
    int k;
    std::cin >> n >> k;
    
    std::vector<int> cnt(K + 1);
    for (int i = 0; i < k; i++) {
        int t;
        std::cin >> t;
        cnt[t]++;
    }
    
    std::vector<Z> dp(K + 1), dp2(K + 1);
    dp[0] = 1;
    dp2[0] = 1;
    for (int i = 1; i <= K; i++) {
        if (cnt[i] == 0) {
            continue;
        }
        for (int s = K; s >= 0; s--) {
            for (int j = 1; s + j * i <= K; j++) {
                Z w = comb.binom(j - 1 + cnt[i], cnt[i] - 1);
                dp[s + j * i] += dp[s] * w;
                w = comb.binom(j - 1, cnt[i] - 1);
                dp2[s + j * i] += dp2[s] * w;
            }
            dp2[s] = 0;
        }
    }
    for (int i = 0; i <= K; i++) {
        dp[i] -= dp2[i];
        // std::cerr << dp[i] << " \n"[i == K];
    }
    
    const int sqrtn = std::sqrt(n);
    std::vector<i64> v;
    for (int i = 1; i <= sqrtn; i++) {
        v.push_back(n / i);
    }
    for (int i = n / sqrtn - 1; i >= 1; i--) {
        v.push_back(i);
    }
    
    const int m = v.size();
    
    auto idx = [&](i64 x) {
        if (x <= sqrtn) {
            return m - x;
        } else {
            return n / x - 1;
        }
    };
    
    std::vector<Z> f(m);
    for (int i = 0; i < m; i++) {
        f[i] = dp[1] * (v[i] - 1);
    }
    
    for (int i = 0; i < primes.size(); i++) {
        if (1LL * primes[i] * primes[i] > n) {
            break;
        }
        for (int j = 0; j < m && v[j] >= 1LL * primes[i] * primes[i]; j++) {
            Z w = f[idx(v[j] / primes[i])] - i * dp[1];
            f[j] -= w;
        }
    }
    for (int i = primes.size() - 1; i >= 0; i--) {
        if (1LL * primes[i] * primes[i] > n) {
            continue;
        }
        for (int j = 0; j < m && v[j] >= 1LL * primes[i] * primes[i]; j++) {
            int t = 1;
            i64 c = v[j] / primes[i];
            while (c >= primes[i]) {
                f[j] += (f[idx(c)] - (i + 1) * dp[1]) * dp[t] + dp[t + 1];
                t++;
                c /= primes[i];
            }
        }
    }
    std::cout << f[0] + 1 << "\n";
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    sieve(1E6);
    
    int T;
    std::cin >> T;
    
    while (T--) {
        solve();
    }
    
    return 0;
}

```



