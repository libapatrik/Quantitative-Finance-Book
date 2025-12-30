# COS Method for American Options: A Model Validation Perspective

This document provides comprehensive study material on the COS (Fourier-cosine series expansion) method for pricing American options, with emphasis on model validation and comparison with alternative approaches.

**Reference:** Fang & Oosterlee (2009), "Pricing early-exercise and discrete barrier options by Fourier-cosine series expansions", _Numerische Mathematik_ 114(1):27-62

---

## Table of Contents

1. [The American Option Pricing Problem](#1-the-american-option-pricing-problem)
2. [Overview of Numerical Methods](#2-overview-of-numerical-methods)
3. [The COS Method: Mathematical Foundation](#3-the-cos-method-mathematical-foundation)
4. [Algorithm for American Options](#4-algorithm-for-american-options)
5. [Model Validation Framework](#5-model-validation-framework)
6. [Comparison of Methods](#6-comparison-of-methods)
7. [Implementation Pitfalls](#7-implementation-pitfalls)
8. [Summary and Recommendations](#8-summary-and-recommendations)

---

## 1. The American Option Pricing Problem

### 1.1 Mathematical Formulation

An **American option** grants the holder the right to exercise at any time $t \in [0, T]$ up to maturity. This creates an **optimal stopping problem**:

$$
V(S, t) = \sup_{\tau \in [t, T]} \mathbb{E}\left[e^{-r(\tau - t)} g(S_\tau) \,\Big|\, S_t = S\right]
$$

where:

-   $\tau$ is a stopping time (the exercise decision)
-   $g(S)$ is the payoff: $\max(K - S, 0)$ for a put, $\max(S - K, 0)$ for a call
-   The expectation is under the risk-neutral measure

### 1.2 The Dynamic Programming Principle

At any time, the option value satisfies:

$$
V(S, t) = \max\Big(\underbrace{g(S)}_{\text{immediate exercise}}, \quad \underbrace{e^{-r\Delta t}\,\mathbb{E}[V(S_{t+\Delta t}, t+\Delta t) \mid S_t = S]}_{\text{continuation value}}\Big)
$$

This recursive structure is the foundation for all numerical methods.

### 1.3 Why No Closed-Form Solution?

The American option creates a **free boundary problem**: there exists a critical stock price $S^*(t)$ (the _early exercise boundary_) below which (for puts) immediate exercise is optimal. This boundary is part of the solution, not given a priori — making analytical solutions impossible except in special cases.

---

## 2. Overview of Numerical Methods

### 2.1 Method Comparison Table

| Method                | Time Complexity | Space   | Convergence Rate           | Best For                      |
| --------------------- | --------------- | ------- | -------------------------- | ----------------------------- |
| **Binomial Tree**     | $O(N^2)$        | $O(N)$  | $O(1/N)$                   | Validation, teaching          |
| **Trinomial Tree**    | $O(N^2)$        | $O(N)$  | $O(1/N)$                   | Slightly better than binomial |
| **Finite Difference** | $O(NM)$         | $O(M)$  | $O(\Delta t + \Delta x^2)$ | Production systems            |
| **LSMC**              | $O(NMP)$        | $O(NM)$ | $O(1/\sqrt{N})$            | High dimensions, exotics      |
| **COS Method**        | $O(NM \log N)$  | $O(N)$  | Spectral                   | Fast pricing, Greeks          |

where $N$ = time steps, $M$ = spatial grid points, $P$ = paths (Monte Carlo)

### 2.2 Binomial Tree

**Algorithm:**

1. Build a recombining tree of stock prices: $S_{i,j} = S_0 u^j d^{i-j}$
2. At maturity, set option values to payoff
3. Work backward: $V_{i,j} = \max(g(S_{i,j}), e^{-r\Delta t}[pV_{i+1,j+1} + (1-p)V_{i+1,j}])$

**Convergence:** The CRR binomial tree converges to the true American option price:

$$
|V_N^{\text{binom}} - V^{\text{true}}| = O(1/N)
$$

**Strengths:**

-   Provably convergent
-   Intuitive interpretation
-   Easy to implement and debug

**Weaknesses:**

-   Slow convergence ($O(1/N)$ means doubling accuracy requires 4× computation)
-   $O(N^2)$ complexity becomes expensive for high accuracy

### 2.3 Finite Difference Methods

Transform the option pricing PDE to a grid and solve numerically:

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
$$

with the constraint $V \geq g(S)$ (early exercise).

**Variants:**

-   **Explicit:** Simple but requires $\Delta t \leq \frac{\Delta x^2}{2\sigma^2}$ (stability)
-   **Implicit:** Unconditionally stable, requires solving linear systems
-   **Crank-Nicolson:** Second-order in time, but can oscillate near exercise boundary

**Strengths:**

-   Very flexible (handles complex boundaries, variable coefficients)
-   Well-understood theory
-   Industry standard

**Weaknesses:**

-   Grid design requires expertise
-   Boundary conditions need care
-   Doesn't directly extend to high dimensions

### 2.4 Least-Squares Monte Carlo (LSMC)

Longstaff-Schwartz algorithm:

1. Simulate $P$ paths of the underlying
2. At each exercise date (backward), estimate continuation value via regression
3. Compare continuation vs. exercise to determine optimal stopping

**Strengths:**

-   Works for path-dependent options
-   Scales to high dimensions
-   Flexible for complex payoffs

**Weaknesses:**

-   $O(1/\sqrt{P})$ convergence (very slow)
-   Regression introduces bias
-   Computational cost for accurate results

### 2.5 COS Method

Uses Fourier-cosine expansions to efficiently compute the conditional expectation in the dynamic programming equation.

**Key idea:** The transition density $f(y|x)$ can be approximated as a cosine series, with coefficients determined by the characteristic function.

**Strengths:**

-   Very fast: $O(NM \log N)$ with FFT optimization
-   Spectral convergence for smooth problems
-   Works whenever characteristic function is known

**Weaknesses:**

-   Requires analytical characteristic function
-   Less intuitive than tree methods
-   Truncation and discretization need care

---

## 3. The COS Method: Mathematical Foundation

### 3.1 Fourier-Cosine Series

Any function $f(x)$ on $[a, b]$ can be expanded in a cosine series:

$$
f(x) = \sum_{k=0}^{\infty} A_k \cos\left(k\pi \frac{x-a}{b-a}\right)
$$

where the coefficients are:

$$
A_k = \frac{2}{b-a} \int_a^b f(x) \cos\left(k\pi \frac{x-a}{b-a}\right) dx
$$

with $A_0$ having an implicit factor of $\frac{1}{2}$ (the "half-weight" convention).

**Why cosines instead of sines?**

-   Cosines are symmetric: $\cos(-x) = \cos(x)$
-   They naturally satisfy Neumann boundary conditions
-   Faster convergence for functions that don't vanish at boundaries

### 3.2 Characteristic Functions

The **characteristic function** of a random variable $X$ is:

$$
\varphi_X(\omega) = \mathbb{E}[e^{i\omega X}] = \int_{-\infty}^{\infty} e^{i\omega x} f_X(x) dx
$$

This is the Fourier transform of the density. The key insight: **characteristic functions are often known analytically even when densities are not**.

**For Geometric Brownian Motion:**

If $X_t = \ln(S_t/S_0)$ follows GBM with drift $\mu = r - \frac{1}{2}\sigma^2$:

$$
\varphi(\omega; t) = \exp\left(i\omega\mu t - \frac{1}{2}\sigma^2\omega^2 t\right)
$$

This is a **complex number** with:

-   Real part: $\text{Re}\{\varphi\} = e^{-\frac{1}{2}\sigma^2\omega^2 t} \cos(\omega\mu t)$
-   Imaginary part: $\text{Im}\{\varphi\} = e^{-\frac{1}{2}\sigma^2\omega^2 t} \sin(\omega\mu t)$

### 3.3 COS Approximation of the Density

The transition density $f(y|x)$ from $x$ at time $t$ to $y$ at time $t + \Delta t$ can be approximated:

$$
f(y|x) \approx \frac{2}{b-a} \sum_{k=0}^{N-1} \text{Re}\left\{\varphi_k \cdot e^{ik\pi\frac{x-a}{b-a}}\right\} \cos\left(k\pi\frac{y-a}{b-a}\right)
$$

where $\varphi_k = \varphi\left(\frac{k\pi}{b-a}; \Delta t\right)$.

### 3.4 Critical Formula: Complex Exponential

When evaluating $\text{Re}\{\varphi_k \cdot e^{i\theta}\}$ where $\theta = k\pi\frac{x-a}{b-a}$:

$$
\text{Re}\{\varphi_k \cdot e^{i\theta}\} = \text{Re}\{\varphi_k\}\cos(\theta) - \text{Im}\{\varphi_k\}\sin(\theta)
$$

**This is crucial!** Using only $\text{Re}\{\varphi_k\} \cdot \cos(\theta)$ (ignoring the imaginary part) causes ~20% pricing errors.

### 3.5 Domain Truncation

The infinite real line must be truncated to a finite interval $[a, b]$. Using the cumulants of the log-return distribution:

-   First cumulant (mean): $c_1 = \mu T$
-   Second cumulant (variance): $c_2 = \sigma^2 T$

The truncation bounds are:

$$
a = c_1 - L\sqrt{c_2}, \quad b = c_1 + L\sqrt{c_2}
$$

where $L \approx 10$ captures more than 99.99% of the probability mass.

---

## 4. Algorithm for American Options

### 4.1 Variable Transformation

Work in log-moneyness: $x = \ln(S/K)$

The put payoff becomes: $g(x) = K \cdot \max(1 - e^x, 0)$

This is non-zero for $x < 0$ (in-the-money region).

### 4.2 Backward Induction Algorithm

**Initialization** at maturity $t_M = T$:

$$
V_k(T) = \frac{2}{b-a} \int_a^{\min(0,b)} (1 - e^x) \cos\left(k\pi\frac{x-a}{b-a}\right) dx
$$

These integrals have closed-form solutions (the $\chi_k$ and $\psi_k$ functions in the paper).

**Backward iteration** for $m = M-1, \ldots, 0$:

1. **Compute continuation value** on a spatial grid $\{x_j\}$:

    $$
    c(x_j) = e^{-r\Delta t} \sum_{k=0}^{N-1} \text{Re}\{\varphi_k \cdot e^{ik\pi(x_j-a)/(b-a)}\} \cdot V_k(t_{m+1}) \cdot w_k
    $$

2. **Apply early exercise**:

    $$
    V(x_j, t_m) = \max(c(x_j), g(x_j))
    $$

3. **Project back to Fourier space**:

    $$
    V_k(t_m) = \frac{2}{b-a} \int_a^b V(x, t_m) \cos\left(k\pi\frac{x-a}{b-a}\right) dx
    $$

    (In practice, use numerical integration on the grid)

**Final evaluation** at $x_0 = \ln(S_0/K)$:

$$
V(S_0, 0) = K \cdot e^{-r\Delta t} \sum_{k=0}^{N-1} \text{Re}\{\varphi_k \cdot e^{ik\pi(x_0-a)/(b-a)}\} \cdot V_k(t_1) \cdot w_k
$$

### 4.3 Implementation Variants

**Grid reconstruction (this notebook):**

-   Reconstruct $V(x)$ on a dense grid at each step
-   Apply max pointwise
-   Project back via numerical integration
-   Simple, robust, $O(N \cdot n_{\text{grid}})$ per step

**FFT-based with boundary detection (paper's Algorithm 2):**

-   Find early exercise boundary $x^*$ analytically (Newton's method)
-   Split $V_k = G_k(\text{exercise region}) + C_k(\text{continuation region})$
-   Use FFT for efficient $C_k$ computation
-   More complex, but $O(N \log N)$ per step

---

## 5. Model Validation Framework

### 5.1 The Validation Challenge

For American options, there is **no closed-form solution** to compare against. We must validate numerically:

```
                    ┌─────────────────────────┐
                    │   True American Price   │
                    │      (unknown)          │
                    └───────────┬─────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
  │  Binomial   │       │   Finite    │       │    COS      │
  │  N → ∞      │       │  Difference │       │   Method    │
  │  O(1/N)     │       │  O(Δx²,Δt)  │       │  Spectral   │
  └─────────────┘       └─────────────┘       └─────────────┘
```

### 5.2 Using Binomial as Benchmark

The binomial tree is an ideal benchmark because:

1. **Provable convergence:** $V_N^{\text{binom}} \to V^{\text{true}}$ as $N \to \infty$
2. **Known rate:** Error $\approx C/N$ for some constant $C$
3. **No hidden parameters:** Just the number of steps $N$
4. **Easy to verify:** Simple enough to check by hand

**Convergence table:**

| Binomial$N$ | Approx. Error   | Decimal Accuracy |
| ----------- | --------------- | ---------------- |
| 100         | ~0.01 (0.1%)    | 2-3 digits       |
| 500         | ~0.002 (0.03%)  | 3-4 digits       |
| 1000        | ~0.001 (0.02%)  | 4 digits         |
| 2000        | ~0.0005 (0.01%) | 4-5 digits       |

**Validation principle:** If binomial with $N=1000$ has error ~0.02% and COS has error ~0.5% vs. binomial, we can confidently say:

-   COS error vs. true price ≈ 0.5% (binomial error is negligible)

### 5.3 Validation Test Suite

A robust validation should include:

**1. Convergence tests:**

-   COS error vs. $N$ (Fourier terms): should decrease
-   COS error vs. $M$ (time steps): should decrease
-   Binomial error vs. $N$: should follow $O(1/N)$

**2. Cross-method validation:**

-   COS vs. Binomial (different $N$)
-   COS vs. Finite Difference (if available)
-   All methods should converge to same value

**3. Parameter sensitivity:**

-   Moneyness: OTM, ATM, ITM
-   Volatility: low (10%), medium (20%), high (40%)
-   Interest rate: low (1%), medium (5%), high (10%)
-   Maturity: short (0.25y), medium (1y), long (3y)

**4. Boundary cases:**

-   Deep ITM: should approach intrinsic value
-   Deep OTM: should approach zero
-   American call (no dividends): should equal European call

**5. Stress tests:**

-   Very short maturity ($T = 0.01$)
-   Very high volatility ($\sigma = 1.0$)
-   Near-zero interest rate

### 5.4 Error Analysis

**COS method error sources:**

| Source              | Parameter         | Effect            | Typical Value |
| ------------------- | ----------------- | ----------------- | ------------- |
| Fourier truncation  | $N$               | Smooth decay      | $N = 128$     |
| Time discretization | $M$               | Bermudan approx   | $M = 50-100$  |
| Domain truncation   | $L$               | Tail probability  | $L = 10$      |
| Grid resolution     | $n_{\text{grid}}$ | Integration error | 2000+ points  |

**Typical accuracy achieved:**

-   $N = 128$, $M = 50$: ~0.5% error
-   $N = 256$, $M = 100$: ~0.2% error
-   Higher parameters: diminishing returns

---

## 6. Comparison of Methods

### 6.1 Accuracy vs. Computation Time

For a typical American put ($S_0 = K = 100$, $r = 5\%$, $\sigma = 20\%$, $T = 1$):

| Method   | Parameters       | Error vs. Reference | Time   |
| -------- | ---------------- | ------------------- | ------ |
| Binomial | $N = 200$        | 0.06%               | 20 ms  |
| Binomial | $N = 1000$       | 0.01%               | 300 ms |
| COS      | $N=128$, $M=50$  | 0.5%                | 100 ms |
| COS      | $N=256$, $M=100$ | 0.2%                | 500 ms |
| LSMC     | $10^5$ paths     | 1-2%                | 2 sec  |

### 6.2 When to Use Each Method

**Use Binomial when:**

-   You need a trusted benchmark
-   Implementation simplicity matters
-   Moderate accuracy is sufficient
-   Teaching/understanding the problem

**Use Finite Difference when:**

-   High accuracy needed
-   Complex boundary conditions
-   Variable coefficients (local volatility)
-   Production pricing systems

**Use COS when:**

-   Characteristic function is known analytically
-   Fast repricing needed (calibration, Greeks)
-   Multiple strikes/maturities at once
-   Model has complex dynamics but known CF (Heston, VG, etc.)

**Use LSMC when:**

-   High-dimensional problems (baskets)
-   Path-dependent payoffs
-   Model simulation is easier than CF
-   Moderate accuracy is acceptable

### 6.3 Extension to Complex Models

A key advantage of COS: it works for any model with a known characteristic function.

| Model                 | Characteristic Function | COS Applicable? |
| --------------------- | ----------------------- | --------------- |
| Black-Scholes/GBM     | Closed-form             | Yes             |
| Heston stochastic vol | Closed-form             | Yes             |
| Variance Gamma        | Closed-form             | Yes             |
| CGMY/Tempered Stable  | Closed-form             | Yes             |
| Local Volatility      | Not available           | No              |
| SABR                  | Approximation only      | Partially       |

---

## 7. Implementation Pitfalls

### 7.1 The Imaginary Part Bug

**Symptom:** Prices are ~20% too high

**Cause:** Using $\text{Re}\{\varphi_k\} \cdot \cos(\theta)$ instead of $\text{Re}\{\varphi_k \cdot e^{i\theta}\}$

**Fix:** Always use the full complex formula:

```python
phi_exp_re = phi_re * np.cos(theta) - phi_im * np.sin(theta)
```

### 7.2 Half-Weight on k=0

**Symptom:** Small systematic bias

**Cause:** The cosine series convention requires $w_0 = 0.5$, $w_k = 1$ for $k > 0$

**Fix:**

```python
weights = np.ones(N)
weights[0] = 0.5
```

### 7.3 Domain Too Narrow

**Symptom:** Prices wrong for extreme strikes or long maturities

**Cause:** Truncation bounds $[a, b]$ don't capture enough of the density

**Fix:** Use $L = 10$ or more standard deviations:

```python
a = c1 - L * np.sqrt(c2)
b = c1 + L * np.sqrt(c2)
```

### 7.4 Grid Too Coarse

**Symptom:** Oscillating or unstable results

**Cause:** Numerical integration (for projecting back to Fourier space) is inaccurate

**Fix:** Use sufficient grid points (2000+) and proper integration (trapezoid rule)

### 7.5 Not Enough Time Steps

**Symptom:** Prices consistently too low

**Cause:** Bermudan option (discrete exercise) < American option (continuous)

**Fix:** Increase $M$ until convergence (typically $M = 50-100$ sufficient)

---

## 8. Summary and Recommendations

### 8.1 Key Takeaways

1. **American options have no closed-form solution** — numerical methods are essential
2. **The COS method** leverages characteristic functions for efficient computation:

    - Fast: $O(NM \log N)$ complexity
    - Accurate: spectral convergence for smooth problems
    - Flexible: works for any model with known CF

3. **Model validation requires care:**

    - Use binomial tree as trusted benchmark (provably convergent)
    - Test across parameter ranges (moneyness, volatility, maturity)
    - Verify convergence behavior

4. **Common implementation bugs:**

    - Missing imaginary part of characteristic function (~20% error)
    - Wrong weights on Fourier coefficients
    - Insufficient domain truncation or grid resolution

### 8.2 Recommended Settings

For most applications:

| Parameter            | Recommended | Notes                               |
| -------------------- | ----------- | ----------------------------------- |
| $N$ (Fourier terms)  | 128         | Increase to 256 for higher accuracy |
| $M$ (exercise dates) | 50-100      | More for longer maturities          |
| $L$ (truncation)     | 10          | Standard deviations                 |
| $n_{\text{grid}}$    | 2000        | For grid reconstruction variant     |

### 8.3 Further Reading

-   **Original paper:** Fang & Oosterlee (2009) — full algorithm details
-   **European options:** Fang & Oosterlee (2008) — simpler case, same methodology
-   **Extensions:** Barrier options, Bermudan swaptions, Greeks computation
-   **Alternative methods:** Finite difference (Wilmott), LSMC (Longstaff-Schwartz)

---

## 9. Quant Interview Q&A: Rapid Fire

Test your understanding with these interview-style questions. Answers are hidden — try to answer before revealing.

### Fundamentals

**Q1: Why can't we use Black-Scholes for American puts?**

<details>
<summary>Answer</summary>

Black-Scholes assumes European exercise (only at maturity). American options have an early exercise feature that creates a free boundary problem — the optimal exercise boundary is part of the solution, not known in advance. This makes the problem fundamentally different and analytically intractable.

</details>

**Q2: What is the early exercise premium? When is it largest?**

<details>
<summary>Answer</summary>

The early exercise premium is: American Price − European Price. It's always ≥ 0.

For puts: Largest when deep ITM and interest rates are high (time value of receiving $K$ now vs. later).

For calls on non-dividend stocks: Zero! Early exercise is never optimal because you lose the time value of the strike payment.

</details>

**Q3: Why does the binomial tree converge to the true American option price?**

<details>
<summary>Answer</summary>

As $N \to \infty$:

1. The discrete stock price process converges to GBM (central limit theorem)
2. The backward induction implements the dynamic programming principle exactly
3. The limiting value satisfies the continuous-time optimal stopping problem

The convergence rate is $O(1/N)$ for the CRR parameterization.

</details>

### COS Method Specifics

**Q4: What is a characteristic function and why is it useful?**

<details>
<summary>Answer</summary>

The characteristic function is $\varphi(\omega) = \mathbb{E}[e^{i\omega X}]$, the Fourier transform of the density.

Useful because:

1. Often known analytically even when density is not (Heston, VG, etc.)
2. Convolutions become multiplications in Fourier space
3. Moments can be extracted via derivatives at $\omega = 0$

</details>

**Q5: Write the characteristic function for GBM log-returns.**

<details>
<summary>Answer</summary>

For $X_t = \ln(S_t/S_0)$ under GBM with drift $\mu = r - \frac{1}{2}\sigma^2$:

$$
\varphi(\omega; t) = \exp\left(i\omega\mu t - \frac{1}{2}\sigma^2\omega^2 t\right)
$$

This is the CF of a normal distribution with mean $\mu t$ and variance $\sigma^2 t$.

</details>

**Q6: What's the critical formula when reconstructing values in the COS method?**

<details>
<summary>Answer</summary>

$$
\text{Re}\{\varphi_k \cdot e^{i\theta}\} = \text{Re}\{\varphi_k\}\cos(\theta) - \text{Im}\{\varphi_k\}\sin(\theta)
$$

You must include BOTH the real and imaginary parts of $\varphi_k$. Using only $\text{Re}\{\varphi_k\} \cdot \cos(\theta)$ causes ~20% errors!

</details>

**Q7: Why do we use cosine series instead of sine series?**

<details>
<summary>Answer</summary>

1. Cosines are symmetric: naturally handle reflection at boundaries
2. Cosines don't require the function to vanish at endpoints (Neumann vs. Dirichlet)
3. Better convergence for smooth, non-vanishing functions
4. The density function typically doesn't vanish at the truncation boundaries

</details>

**Q8: How do you choose the truncation range [a, b]?**

<details>
<summary>Answer</summary>

Using cumulants of the log-return distribution:

$$
a = c_1 - L\sqrt{c_2}, \quad b = c_1 + L\sqrt{c_2}
$$

where:

-   $c_1 = \mu T$ (first cumulant, mean)
-   $c_2 = \sigma^2 T$ (second cumulant, variance)
-   $L \approx 10$ (captures >99.99% of probability mass)

For more complex models, use higher cumulants if available.

</details>

### Implementation & Validation

**Q9: What are the three main error sources in the COS method for American options?**

<details>
<summary>Answer</summary>

1. **Fourier truncation** ($N$): Using finite number of cosine terms
2. **Time discretization** ($M$): Bermudan approximation to continuous exercise
3. **Domain truncation** ($L$): Finite interval instead of infinite real line

Secondary: grid resolution for numerical integration in the reconstruction variant.

</details>

**Q10: How would you validate a COS implementation if you don't have another COS code to compare against?**

<details>
<summary>Answer</summary>

1. **European option test:** COS should match Black-Scholes exactly (no early exercise)
2. **Binomial benchmark:** Use high-$N$ binomial as reference (provably convergent)
3. **Convergence test:** Error should decrease as $N$, $M$ increase
4. **American call test:** Should equal European call (no dividends)
5. **Deep ITM test:** Should approach intrinsic value
6. **Cross-parameter tests:** Various $\sigma$, $r$, $T$ combinations

</details>

**Q11: Your COS prices are ~20% higher than binomial. What's likely wrong?**

<details>
<summary>Answer</summary>

Almost certainly the **imaginary part bug**: using $\text{Re}\{\varphi_k\} \cdot \cos(\theta)$ instead of the correct $\text{Re}\{\varphi_k\}\cos(\theta) - \text{Im}\{\varphi_k\}\sin(\theta)$.

This is the #1 implementation mistake.

</details>

**Q12: Your prices are consistently slightly low. What might be the issue?**

<details>
<summary>Answer</summary>

Likely **insufficient time steps** ($M$). The COS method prices a Bermudan option (discrete exercise dates), which is always worth less than a true American option (continuous exercise).

As $M \to \infty$, Bermudan → American.

</details>

### Comparisons & Trade-offs

**Q13: When would you use COS over binomial?**

<details>
<summary>Answer</summary>

Use COS when:

-   Speed matters (calibration, real-time pricing)
-   Model has known CF but complex dynamics (Heston, VG)
-   Pricing many strikes/maturities at once
-   Computing Greeks via finite differences

Use binomial when:

-   Simplicity and transparency matter
-   Validating other methods
-   Teaching/understanding the problem

</details>

**Q14: Can you use COS for local volatility models?**

<details>
<summary>Answer</summary>

**No**, because local volatility models don't have a known characteristic function. The CF depends on the entire local vol surface, which is path-dependent.

For local vol, use finite differences or Monte Carlo.

</details>

**Q15: What's the complexity of COS vs. binomial for American options?**

<details>
<summary>Answer</summary>

-   **Binomial:** $O(N^2)$ where $N$ = time steps
-   **COS (grid variant):** $O(M \cdot N \cdot n_{\text{grid}})$ where $M$ = exercise dates, $N$ = Fourier terms
-   **COS (FFT variant):** $O(M \cdot N \log N)$

COS is faster when high accuracy is needed, since binomial's $O(1/N)$ convergence requires large $N$ for accuracy.

</details>

### Advanced

**Q16: How would you extend COS to the Heston model?**

<details>
<summary>Answer</summary>

The algorithm is identical! Just replace the GBM characteristic function with the Heston CF:

$$
\varphi(\omega; t) = \exp(C(\omega,t) + D(\omega,t)v_0 + i\omega x_0)
$$

where $C$ and $D$ are known functions involving the Heston parameters $(\kappa, \theta, \xi, \rho)$.

The rest of the backward induction works exactly the same.

</details>

**Q17: What is the early exercise boundary and how does COS handle it?**

<details>
<summary>Answer</summary>

The early exercise boundary $S^*(t)$ is the critical stock price where continuation value equals exercise value.

Two approaches in COS:

1. **Grid reconstruction** (this notebook): Don't explicitly track boundary; just take max pointwise
2. **Boundary detection** (paper's Algorithm 2): Find $x^*$ via Newton's method, split integrals analytically

The grid approach is simpler; boundary detection is faster.

</details>

**Q18: How would you compute Delta using COS?**

<details>
<summary>Answer</summary>

Two approaches:

1. **Finite difference:** $\Delta \approx \frac{V(S_0 + h) - V(S_0 - h)}{2h}$ (just reprice twice)
2. **Pathwise/Likelihood ratio:** Differentiate through the COS formula w.r.t. $x_0 = \ln(S_0/K)$

The first is simpler but requires two price calculations. COS is fast enough that this is usually fine.

</details>

**Q19: What happens to COS accuracy for very short maturities?**

<details>
<summary>Answer</summary>

Can be problematic because:

1. Domain $[a,b]$ becomes very narrow ($\sigma\sqrt{T}$ small)
2. Early exercise boundary may be outside the domain
3. Payoff discontinuity (at $S = K$) not well-resolved by smooth cosine basis

Fixes: Increase $L$, use more Fourier terms near maturity, or switch to a different method for $T < 0.1$.

</details>

**Q20: Bonus: Why is it called the "COS" method?**

<details>
<summary>Answer</summary>

**COS** = **CO**sine **S**eries expansion

Named after the Fourier-cosine basis functions used in the approximation. The original paper by Fang & Oosterlee (2008) introduced it for European options; the 2009 follow-up extended it to American/Bermudan options.

</details>

---

_Document prepared as study material for the COS method implementation in the Quantitative Finance Book._
