"""
COS method for European option Greeks.
Fang & Oosterlee (2008) - analytical Delta/Gamma, FD for Theta/Vega/Rho.
"""
import os
import sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import cos_price, _cos_payoff_coeffs, _compute_domain


def cos_greeks(S0, K, T, r, cf, N=128, L=10, opt_type='call', std=None):
    """
    Analytical COS Greeks. Delta/Gamma from derivatives of COS formula, Theta via FD.

    Δ = -(Ke^{-rT}/S₀) Σ' ωₖ Im[Hₖ]Vₖ
    Γ = (Ke^{-rT}/S₀²) Σ' (ωₖ Im[Hₖ] - ωₖ² Re[Hₖ])Vₖ
    """
    x0 = np.log(S0 / K)
    a, b = _compute_domain(x0, T, L, std, r)
    bma = b - a

    k = np.arange(N)
    w = k * np.pi / bma
    H = cf(w) * np.exp(1j * w * (x0 - a))
    Re_H, Im_H = np.real(H), np.imag(H)

    V = _cos_payoff_coeffs(k, a, b, opt_type)
    V[0] /= 2

    disc = np.exp(-r * T)

    price = K * disc * np.sum(Re_H * V)
    delta = -(K * disc / S0) * np.sum(w * Im_H * V)
    gamma = (K * disc / S0**2) * np.sum((w * Im_H - w**2 * Re_H) * V)

    # Theta via finite difference: θ = ∂C/∂T ≈ (C(T-ε) - C(T)) / (-ε)
    # Note: We compute ∂C/∂τ where τ = T - t (time to maturity decreases)
    # θ = -∂C/∂τ, so θ = (C(T) - C(T-ε)) / ε when τ decreases
    eps = max(1e-5, min(0.001, T/100))
    if T > eps:
        p_dn = cos_price(S0, K, T - eps, r, cf, N, L, opt_type, std)
        theta = (p_dn - price) / eps  # As T decreases, option loses value
    else:
        theta = 0.0

    return {'price': max(0, price), 'delta': delta, 'gamma': gamma, 'theta': theta}


def cos_greeks_full(S0, K, T, r, sigma, model='bs', N=128, L=10, opt_type='call',
                    heston_cf=None, **params):
    """
    Full Greeks: price, delta, gamma, theta, vega, rho.
    model='bs' or 'heston'. For Heston, sigma=v0 and pass heston_cf + params.
    """
    eps_sigma = 0.0001
    eps_r = 0.0001
    eps_T = max(1e-5, min(0.001, T/100))

    if model == 'bs':
        # For BS, std = sigma
        std = sigma

        def make_cf(tau, rate=r):
            def cf(u):
                drift = rate - 0.5 * sigma**2
                return np.exp(1j * u * drift * tau - 0.5 * sigma**2 * tau * u**2)
            return cf

        g = cos_greeks(S0, K, T, r, make_cf(T), N, L, opt_type, std)

        # Vega: ∂C/∂σ via central difference
        def cf_up(u):
            s = sigma + eps_sigma
            return np.exp(1j*u*(r - 0.5*s**2)*T - 0.5*s**2*T*u**2)
        def cf_dn(u):
            s = sigma - eps_sigma
            return np.exp(1j*u*(r - 0.5*s**2)*T - 0.5*s**2*T*u**2)

        p_up = cos_price(S0, K, T, r, cf_up, N, L, opt_type, sigma + eps_sigma)
        p_dn = cos_price(S0, K, T, r, cf_dn, N, L, opt_type, sigma - eps_sigma)
        g['vega'] = (p_up - p_dn) / (2 * eps_sigma)

        # Rho: ∂C/∂r via central difference
        p_r_up = cos_price(S0, K, T, r + eps_r, make_cf(T, r + eps_r), N, L, opt_type, std)
        p_r_dn = cos_price(S0, K, T, r - eps_r, make_cf(T, r - eps_r), N, L, opt_type, std)
        g['rho'] = (p_r_up - p_r_dn) / (2 * eps_r)

        # Theta: CF must also change with T
        if T > eps_T:
            p_T_dn = cos_price(S0, K, T - eps_T, r, make_cf(T - eps_T), N, L, opt_type, std)
            g['theta'] = (p_T_dn - g['price']) / eps_T
        else:
            g['theta'] = 0.0

    elif model == 'heston':
        if heston_cf is None:
            raise ValueError("Must provide heston_cf function for Heston model")

        kappa = params.get('kappa', 2.0)
        theta_h = params.get('theta', 0.04)
        xi = params.get('xi', 0.3)
        rho_h = params.get('rho', -0.7)
        v0 = sigma  # sigma parameter is v0 for Heston

        # For Heston, use long-term vol as approximate std
        std = np.sqrt(theta_h)

        def make_cf(tau, rate=r, vol0=v0):
            def cf(u):
                # Heston CF gives E[exp(iu*ln(S_T))]
                # We need E[exp(iu*ln(S_T/S_0))] = Heston_CF / S0^{iu}
                return heston_cf(u, S0, tau, rate, kappa, vol0, theta_h, xi, rho_h) / S0**(1j*u)
            return cf

        g = cos_greeks(S0, K, T, r, make_cf(T), N, L, opt_type, std)

        # Vega w.r.t. v0 (initial variance)
        def cf_v_up(u):
            return heston_cf(u, S0, T, r, kappa, v0 + eps_sigma, theta_h, xi, rho_h) / S0**(1j*u)
        def cf_v_dn(u):
            return heston_cf(u, S0, T, r, kappa, v0 - eps_sigma, theta_h, xi, rho_h) / S0**(1j*u)

        p_v_up = cos_price(S0, K, T, r, cf_v_up, N, L, opt_type, std)
        p_v_dn = cos_price(S0, K, T, r, cf_v_dn, N, L, opt_type, std)
        vega_v0 = (p_v_up - p_v_dn) / (2 * eps_sigma)

        # Convert to per-σ units: if σ = √v0, then ∂C/∂σ = ∂C/∂v0 × 2√v0
        g['vega'] = vega_v0 * 2 * np.sqrt(v0)

        # Rho: ∂C/∂r
        p_r_up = cos_price(S0, K, T, r + eps_r, make_cf(T, r + eps_r), N, L, opt_type, std)
        p_r_dn = cos_price(S0, K, T, r - eps_r, make_cf(T, r - eps_r), N, L, opt_type, std)
        g['rho'] = (p_r_up - p_r_dn) / (2 * eps_r)

        # Theta
        if T > eps_T:
            p_T_dn = cos_price(S0, K, T - eps_T, r, make_cf(T - eps_T), N, L, opt_type, std)
            g['theta'] = (p_T_dn - g['price']) / eps_T
        else:
            g['theta'] = 0.0

    else:
        raise ValueError(f"Unknown model: {model}")

    return g


def bs_greeks_analytical(S0, K, T, r, sigma, opt_type='call'):
    """Closed-form Black-Scholes Greeks for validation."""
    if T <= 0:
        intrinsic = max(S0-K, 0) if opt_type == 'call' else max(K-S0, 0)
        return {'price': intrinsic,
                'delta': 1.0 if (opt_type == 'call' and S0 > K) else (-1.0 if (opt_type == 'put' and S0 < K) else 0.0),
                'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

    sqT = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqT)
    d2 = d1 - sigma*sqT

    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)

    # Greeks that are the same for calls and puts
    gamma = pdf_d1 / (S0 * sigma * sqT)
    vega = S0 * pdf_d1 * sqT

    if opt_type == 'call':
        price = S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -S0 * pdf_d1 * sigma / (2*sqT) - r * K * disc * norm.cdf(d2)
        rho = K * T * disc * norm.cdf(d2)
    else:
        price = K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = -S0 * pdf_d1 * sigma / (2*sqT) + r * K * disc * norm.cdf(-d2)
        rho = -K * T * disc * norm.cdf(-d2)

    return {'price': price, 'delta': delta, 'gamma': gamma,
            'theta': theta, 'vega': vega, 'rho': rho}


def validate_cos_greeks():
    """Test COS Greeks against BS analytical formulas. Returns True if all pass."""
    test_cases = [
        # (S0, K, T, r, sigma, opt_type)
        (100, 100, 1.0, 0.05, 0.2, 'call'),   # ATM call
        (100, 100, 1.0, 0.05, 0.2, 'put'),    # ATM put
        (100, 110, 0.5, 0.03, 0.25, 'call'),  # OTM call
        (100, 90, 0.5, 0.03, 0.25, 'put'),    # OTM put
        (100, 80, 2.0, 0.08, 0.3, 'call'),    # ITM call, long maturity
        (100, 120, 2.0, 0.08, 0.3, 'put'),    # ITM put, long maturity
        # Long-dated options (tests drift correction in domain)
        (100, 100, 5.0, 0.05, 0.2, 'call'),   # ATM call, T=5y
        (100, 100, 5.0, 0.05, 0.2, 'put'),    # ATM put, T=5y
        (100, 100, 10.0, 0.05, 0.2, 'call'),  # ATM call, T=10y
    ]

    print("COS Greeks Validation")
    print("=" * 80)

    all_passed = True
    tolerances = {'price': 1e-4, 'delta': 1e-4, 'gamma': 1e-4,
                  'theta': 1e-2, 'vega': 1e-2, 'rho': 1e-2}

    for S0, K, T, r, sigma, opt_type in test_cases:
        bs = bs_greeks_analytical(S0, K, T, r, sigma, opt_type)
        cos = cos_greeks_full(S0, K, T, r, sigma, model='bs', N=256, L=12, opt_type=opt_type)

        print(f"\nTest: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}, {opt_type}")
        print("-" * 80)
        print(f"{'Greek':<10} {'BS Analytical':>15} {'COS Method':>15} {'Error':>15} {'Status':>10}")
        print("-" * 80)

        for greek in ['price', 'delta', 'gamma', 'theta', 'vega', 'rho']:
            bs_val = bs[greek]
            cos_val = cos[greek]

            # Relative error for non-zero values, absolute for near-zero
            if abs(bs_val) > 1e-6:
                error = abs(cos_val - bs_val) / abs(bs_val)
            else:
                error = abs(cos_val - bs_val)

            passed = error < tolerances[greek]
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False

            print(f"{greek:<10} {bs_val:>15.6f} {cos_val:>15.6f} {error:>15.2e} {status:>10}")

    print("\n" + "=" * 80)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    validate_cos_greeks()