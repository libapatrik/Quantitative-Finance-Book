"""
COS European Option Pricing & Greeks
Reference: Fang & Oosterlee (2008)

Analytical derivatives of the COS pricing formula for Delta and Gamma.
"""
import numpy as np
from scipy.stats import norm


def _cos_chi(k, c, d, a, b):
    """χ_k(c,d): Eq. (22) - cosine coefficient for e^x."""
    bma = b - a
    k = np.atleast_1d(k).astype(float)
    chi = np.zeros_like(k)
    for i, ki in enumerate(k):
        if ki == 0:
            chi[i] = np.exp(d) - np.exp(c)
        else:
            w = ki * np.pi / bma
            chi[i] = (1 / (1 + w**2)) * (
                np.exp(d) * (np.cos(w*(d-a)) + w*np.sin(w*(d-a))) -
                np.exp(c) * (np.cos(w*(c-a)) + w*np.sin(w*(c-a)))
            )
    return chi


def _cos_psi(k, c, d, a, b):
    """ψ_k(c,d): Eq. (23) - cosine coefficient for constant 1."""
    bma = b - a
    k = np.atleast_1d(k).astype(float)
    psi = np.zeros_like(k)
    for i, ki in enumerate(k):
        if ki == 0:
            psi[i] = d - c
        else:
            w = ki * np.pi / bma
            psi[i] = (np.sin(w*(d-a)) - np.sin(w*(c-a))) / w
    return psi


def _cos_payoff_coeffs(k, a, b, opt_type='call'):
    """V_k payoff coefficients. Call: [0,b], Put: [a,0]."""
    bma = b - a
    if opt_type == 'call':
        return (2/bma) * (_cos_chi(k, 0, b, a, b) - _cos_psi(k, 0, b, a, b))
    else:
        return (2/bma) * (-_cos_chi(k, a, 0, a, b) + _cos_psi(k, a, 0, a, b))


def cos_price(S0, K, T, r, cf, N=128, L=10, opt_type='call'):
    """COS European option price. cf = characteristic function of log-returns."""
    x0 = np.log(S0 / K)
    a, b = x0 - L*np.sqrt(T), x0 + L*np.sqrt(T)
    bma = b - a

    k = np.arange(N)
    w = k * np.pi / bma
    H = cf(w) * np.exp(1j * w * (x0 - a))

    V = _cos_payoff_coeffs(k, a, b, opt_type)
    V[0] /= 2

    return max(0, K * np.exp(-r*T) * np.sum(np.real(H) * V))


def cos_greeks(S0, K, T, r, cf, N=128, L=10, opt_type='call'):
    """
    Analytical COS Greeks. Derivation from Fang & Oosterlee (2008):

    x₀ = ln(S₀/K), ωₖ = kπ/(b-a), Hₖ = φ(ωₖ)·exp(iωₖ(x₀-a))

    Price: C = Ke^{-rT} Σ' Re[Hₖ]Vₖ
    Delta: Δ = -Ke^{-rT}/S₀ Σ' ωₖ Im[Hₖ]Vₖ
    Gamma: Γ = Ke^{-rT}/S₀² Σ' (ωₖ Im[Hₖ] - ωₖ² Re[Hₖ])Vₖ

    Note: Theta computed here assumes cf is time-independent. For models where
    the CF depends on T (like BS or Heston), use cos_greeks_full() which
    correctly recomputes the CF at T-ε for theta calculation.
    """
    x0 = np.log(S0 / K)
    a, b = x0 - L*np.sqrt(T), x0 + L*np.sqrt(T)
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

    # Theta via finite difference: θ = (C(T-ε) - C(T)) / ε
    eps = min(0.001, T/100)
    if T > eps:
        p_dn = cos_price(S0, K, T-eps, r, cf, N, L, opt_type)
        theta = (p_dn - price) / eps
    else:
        theta = 0.0

    return {'price': max(0, price), 'delta': delta, 'gamma': gamma, 'theta': theta}


def cos_greeks_full(S0, K, T, r, sigma, model='bs', N=128, L=10, opt_type='call',
                    heston_cf=None, **params):
    """Full Greeks with Vega. model='bs' or 'heston'. For Heston, sigma=v0."""
    eps_sigma = 0.001
    eps_T = min(0.001, T/100)

    if model == 'bs':
        def make_cf(tau):
            def cf(u):
                return np.exp(1j*u*(r - 0.5*sigma**2)*tau - 0.5*sigma**2*tau*u**2)
            return cf

        g = cos_greeks(S0, K, T, r, make_cf(T), N, L, opt_type)

        # Vega
        def cf_up(u):
            s = sigma + eps_sigma
            return np.exp(1j*u*(r - 0.5*s**2)*T - 0.5*s**2*T*u**2)
        def cf_dn(u):
            s = sigma - eps_sigma
            return np.exp(1j*u*(r - 0.5*s**2)*T - 0.5*s**2*T*u**2)
        g['vega'] = (cos_price(S0,K,T,r,cf_up,N,L,opt_type) -
                     cos_price(S0,K,T,r,cf_dn,N,L,opt_type)) / (2*eps_sigma)

        # Theta (CF must also change with T)
        if T > eps_T:
            p_dn = cos_price(S0, K, T-eps_T, r, make_cf(T-eps_T), N, L, opt_type)
            g['theta'] = (p_dn - g['price']) / eps_T
        else:
            g['theta'] = 0.0

    elif model == 'heston':
        if heston_cf is None:
            raise ValueError("Must provide heston_cf function for Heston model")

        kappa = params.get('kappa', 2.0)
        theta_h = params.get('theta', 0.04)
        xi = params.get('xi', 0.3)
        rho = params.get('rho', -0.7)
        v0 = sigma

        def make_cf(tau):
            def cf(u):
                return heston_cf(u, S0, tau, r, kappa, v0, theta_h, xi, rho) / S0**(1j*u)
            return cf

        g = cos_greeks(S0, K, T, r, make_cf(T), N, L, opt_type)

        # Vega w.r.t. σ (not v0). Chain rule: ∂C/∂σ = ∂C/∂v₀ × 2σ
        def cf_up(u):
            return heston_cf(u, S0, T, r, kappa, v0+eps_sigma, theta_h, xi, rho) / S0**(1j*u)
        def cf_dn(u):
            return heston_cf(u, S0, T, r, kappa, v0-eps_sigma, theta_h, xi, rho) / S0**(1j*u)
        vega_v0 = (cos_price(S0,K,T,r,cf_up,N,L,opt_type) -
                   cos_price(S0,K,T,r,cf_dn,N,L,opt_type)) / (2*eps_sigma)
        g['vega'] = vega_v0 * 2 * np.sqrt(v0)  # Convert to per-σ units

        # Theta
        if T > eps_T:
            p_dn = cos_price(S0, K, T-eps_T, r, make_cf(T-eps_T), N, L, opt_type)
            g['theta'] = (p_dn - g['price']) / eps_T
        else:
            g['theta'] = 0.0
    else:
        raise ValueError(f"Unknown model: {model}")

    return g


def bs_greeks_analytical(S0, K, T, r, sigma, opt_type='call'):
    """Closed-form BS Greeks for validation."""
    if T <= 0:
        intrinsic = max(S0-K, 0) if opt_type == 'call' else max(K-S0, 0)
        return {'price': intrinsic, 'delta': 1.0 if S0 > K else 0.0,
                'gamma': 0, 'theta': 0, 'vega': 0}

    sqT = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqT)
    d2 = d1 - sigma*sqT

    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S0 * sigma * sqT)
    vega = S0 * pdf_d1 * sqT

    if opt_type == 'call':
        price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -S0*pdf_d1*sigma/(2*sqT) - r*K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = -S0*pdf_d1*sigma/(2*sqT) + r*K*np.exp(-r*T)*norm.cdf(-d2)

    return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}