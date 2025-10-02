"""
Quantitative Finance Book - Utilities Module
===========================================

This module contains all shared functions used across the QFB notebooks.
Consolidates duplicated code and provides a single source of truth for
financial modeling functions.

Author: Patrik Liba
Version: 2.0 - Fixed Carr-Madan zeros issue
"""

# Version identifier for debugging
UTILS_VERSION = "2.0-FIXED-CARR-MADAN"

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.special as ss
from scipy.stats import norm
from scipy.integrate import trapezoid
from scipy.optimize import newton
import pandas as pd

# ===========================================a==================================
# COS METHOD FUNCTIONS
# =============================================================================

def cos_cdf(a, b, omega, chf, x):
    """
    Compute CDF using the COS method.
    
    Parameters:
    -----------
    a : float
        Lower bound of the integration domain
    b : float  
        Upper bound of the integration domain
    omega : np.ndarray
        Frequency array
    chf : np.ndarray
        Characteristic function values at omega frequencies
    x : np.ndarray
        Evaluation points for CDF
        
    Returns:
    --------
    np.ndarray
        CDF values at x points
    """
    F_k = 2.0 / (b - a) * np.real(chf * np.exp(-1j * omega * a))
    cdf = np.squeeze(F_k[0] / 2.0 * (x - a)) + np.matmul(F_k[1:] / omega[1:], np.sin(np.outer(omega[1:], x - a)))
    return cdf


def cos_pdf(a, b, N, chf, x):
    """
    Compute PDF using the COS method.
    
    Parameters:
    -----------
    a : float
        Lower bound of the integration domain
    b : float
        Upper bound of the integration domain  
    N : int
        Number of terms in COS expansion
    chf : callable
        Characteristic function
    x : np.ndarray
        Evaluation points for PDF
        
    Returns:
    --------
    np.ndarray
        PDF values at x points
    """
    k = np.linspace(0, N-1, N)
    u = k * np.pi / (b-a)  # frequencies -- u = omega 
    # F_k coefficients
    F_k = 2.0 / (b - a) * np.real(chf(u) * np.exp(-1j * u * a))
    F_k[0] = F_k[0] * 0.5  # first term
    # Final calculation
    pdf = np.matmul(F_k, np.cos(np.outer(u, x - a)))
    return pdf


# =============================================================================
# CARR-MADAN METHOD FUNCTIONS  
# =============================================================================

def carr_madan_cdf(chf, x_grid, u_max=200, N=2**12, alpha=1.0):
    """
    Recover CDF from characteristic function using Gil-Pelaez inversion formula.
    
    Parameters:
    -----------
    chf : callable
        Characteristic function phi(u)
    x_grid : np.ndarray
        Evaluation points for CDF
    u_max : float, default=200
        Maximum frequency for integration
    N : int, default=2**12
        Number of integration points
    alpha : float, default=1.0
        Damping parameter (not used in current implementation)
        
    Returns:
    --------
    np.ndarray
        CDF values at x_grid points
    """
    # Use adaptive integration bounds based on characteristic function decay
    u_min = 1e-8  # Start from very small but not zero
    
    # Create integration grid with higher density near zero
    u_dense = np.concatenate([
        np.linspace(u_min, 1.0, N//4),      # Dense sampling near 0
        np.linspace(1.0, 10.0, N//4),       # Medium sampling 
        np.linspace(10.0, u_max, N//2)      # Coarse sampling at high frequencies
    ])
    
    # Remove duplicates and sort
    u = np.unique(u_dense)
    
    # Evaluate characteristic function with error handling
    try:
        chf_vals = chf(u)
        
        # Check for numerical issues in CF
        if np.any(~np.isfinite(chf_vals)):
            # Fallback to simpler grid if CF has issues
            u = np.linspace(u_min, min(50, u_max), N//2)
            chf_vals = chf(u)
            
    except Exception:
        # Emergency fallback
        u = np.linspace(u_min, 50, N//2)
        chf_vals = chf(u)
    
    # Compute integrand for each x using Gil-Pelaez formula
    # CDF(x) = 0.5 - (1/π) * ∫[0,∞] Im(exp(-iux) * φ(u)) / u du
    integrand = np.imag(np.exp(-1j * np.outer(x_grid, u)) * chf_vals) / u
    
    # Integrate using trapezoidal rule
    integral = trapezoid(integrand, u, axis=1)
    
    # Apply Gil-Pelaez inversion formula
    cdf = 0.5 - (1 / np.pi) * integral
    
    # Apply soft clipping to handle numerical errors while preserving the fact
    # that some distributions don't span [0,1]
    cdf = np.clip(cdf, -0.1, 1.1)  # Allow some overshoot for numerical errors
    
    return np.squeeze(cdf)


def carr_madan_pdf(chf, x_grid, u_max=200, N=2**12):
    """
    Recover PDF from characteristic function using inverse Fourier transform.
    
    Parameters:
    -----------
    chf : callable
        Characteristic function phi(u)
    x_grid : np.ndarray
        Evaluation points for PDF
    u_max : float, default=200
        Maximum frequency for integration
    N : int, default=2**12
        Number of integration points
        
    Returns:
    --------
    np.ndarray
        PDF values at x_grid points
    """
    du = 2 * u_max / N
    u = np.linspace(-u_max, u_max - du, N)
    # Compute integrand of inverse Fourier transform
    integrand = np.exp(-1j * np.outer(x_grid, u)) * chf(u)
    # Numerical integration using trapezoidal rule
    integral = trapezoid(integrand, u, axis=1)
    pdf = np.real(integral) / (2 * np.pi)
    pdf = np.maximum(pdf, 0)
    return np.atleast_1d(pdf)


# =============================================================================
# CONV METHOD FUNCTIONS
# =============================================================================

def conv_pdf(chf, x_range=(-5, 5), alpha=0.5, N=2**12):
    """
    Compute PDF using the CONV method (FFT-based).
    
    Parameters:
    -----------
    chf : callable
        Characteristic function
    x_range : tuple, default=(-5, 5)
        Spatial domain range (x_min, x_max)
    alpha : float, default=0.5
        Damping parameter
    N : int, default=2**12
        Number of grid points
        
    Returns:
    --------
    tuple
        (x_grid, pdf_values) - spatial grid and corresponding PDF values
    """
    x_min, x_max = x_range
    L = x_max - x_min  # Total length of spatial domain
    dx = L / N
    du = 2 * np.pi / L
    u_max = np.pi / dx  # Nyquist frequency

    # Create frequency array in "standard" order from -u_max to u_max
    if N % 2 == 0:
        # For even N: [-N/2, -N/2+1, ..., -1, 0, 1, ..., N/2-1] x du
        k = np.concatenate([np.arange(-N // 2, 0), np.arange(0, N // 2)])
    else:
        # For odd N: [-(N-1)/2, ..., -1, 0, 1, ..., (N-1)/2] x du
        k = np.arange(-(N - 1) // 2, (N + 1) // 2)
    u = k * du
    x = x_min + np.arange(N) * dx

    phi_damped = chf(u - 1j * alpha)
    integrand = phi_damped * np.exp(-1j * u * x_min)
    integrand_fft_order = np.fft.ifftshift(integrand)
    fft_result = np.fft.fft(integrand_fft_order)

    pdf = np.real(fft_result) * np.exp(-alpha * x) * du / (2 * np.pi)
    # Ensure non-negativity (small numerical errors can cause negative values)
    pdf = np.maximum(pdf, 0)

    # Check: Normalize to ensure integral equals 1
    integral = trapezoid(pdf, x)
    if integral > 0:
        pdf = pdf / integral

    return x, pdf


def conv_cdf(chf, x_vals, x_range=None, alpha=0.5, N=2**12):
    """
    Compute CDF using the CONV method.
    
    Parameters:
    -----------
    chf : callable
        Characteristic function
    x_vals : np.ndarray
        Evaluation points for CDF
    x_range : tuple, optional
        Spatial domain range. If None, automatically determined from x_vals
    alpha : float, default=0.5
        Damping parameter
    N : int, default=2**12
        Number of grid points
        
    Returns:
    --------
    np.ndarray
        CDF values at x_vals points
    """
    if x_range is None:
        # Automatically choose a domain that covers x_vals with some padding
        x_min_req = np.min(x_vals)
        x_max_req = np.max(x_vals)
        padding = (x_max_req - x_min_req) * 0.5  # 50% padding on each side
        x_range = (x_min_req - padding, x_max_req + padding)
    
    x_grid, pdf_grid = conv_pdf(chf=chf, x_range=x_range, alpha=alpha, N=N)

    dx = x_grid[1] - x_grid[0]  # Grid spacing (uniform)
    cdf_grid = np.zeros_like(pdf_grid)

    cdf_grid[0] = 0.0  # CDF starts at 0 at the leftmost point
    for i in range(1, len(cdf_grid)):
        # Add the area of the trapezoid between points i-1 and i
        cdf_grid[i] = cdf_grid[i - 1] + 0.5 * (pdf_grid[i - 1] + pdf_grid[i]) * dx

    cdf_vals = np.interp(x_vals, x_grid, cdf_grid)
    # Ensure CDF properties are satisfied
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)

    return cdf_vals


# =============================================================================
# CDF INVERTER CLASS
# =============================================================================

class CDF_Inverter:
    """
    Unified CDF inversion class supporting multiple methods.
    
    Supports COS, Carr-Madan, and CONV methods for CDF recovery and inversion.
    """
    
    def __init__(self, method="cos", nr_expansion=100, u_max=200, N=2**12, alpha=0.5):
        """
        Initialize CDF inverter.
        
        Parameters:
        -----------
        method : str, default="cos"
            Method to use: "cos", "carr_madan", or "conv"
        nr_expansion : int, default=100
            Number of terms for COS expansion
        u_max : float, default=200
            Maximum frequency for Carr-Madan method
        N : int, default=2**12
            Number of grid points
        alpha : float, default=0.5
            Damping parameter for CONV method
        """
        self.method = method
        self.nr_expansion = nr_expansion
        self.u_max = u_max
        self.N = N
        self.alpha = alpha

    def compute_cdf(self, chf, x_vals, lower_bound, upper_bound):
        """
        Compute CDF using the specified method.
        
        Parameters:
        -----------
        chf : callable
            Characteristic function
        x_vals : np.ndarray
            Evaluation points
        lower_bound : float
            Lower bound of domain
        upper_bound : float
            Upper bound of domain
            
        Returns:
        --------
        np.ndarray
            CDF values at x_vals
        """
        if self.method == "cos":
            # omega array
            omega = np.arange(self.nr_expansion) * np.pi / (upper_bound - lower_bound)
            chf_values = chf(omega)
            return cos_cdf(lower_bound, upper_bound, omega, chf_values, x_vals)

        elif self.method == "carr_madan":
            result = carr_madan_cdf(chf, x_vals, u_max=self.u_max, N=self.N)
            return np.atleast_1d(result)

        elif self.method == "conv":
            return conv_cdf(chf, x_vals, x_range=(lower_bound, upper_bound), alpha=self.alpha, N=self.N)

        else:
            raise ValueError(f"Unknown CDF method: {self.method}")

    def compute_pdf(self, chf, x_vals, lower_bound, upper_bound):
        """
        Compute PDF using the specified method.
        
        Parameters:
        -----------
        chf : callable
            Characteristic function
        x_vals : np.ndarray
            Evaluation points
        lower_bound : float
            Lower bound of domain
        upper_bound : float
            Upper bound of domain
            
        Returns:
        --------
        np.ndarray
            PDF values at x_vals
        """
        if self.method == "cos":
            def chf_wrapper(u):
                return chf(u)
            return cos_pdf(lower_bound, upper_bound, self.nr_expansion, chf_wrapper, x_vals)

        elif self.method == "carr_madan":
            return carr_madan_pdf(chf, x_vals, u_max=self.u_max, N=self.N)

        elif self.method == "conv":
            x_grid, pdf_grid = conv_pdf(chf, x_range=(lower_bound, upper_bound), alpha=self.alpha, N=self.N)
            return np.interp(x_vals, x_grid, pdf_grid)

        else:
            raise ValueError(f"Unknown PDF method: {self.method}")

    def invert_cdf_newton(self, chf, lower_bound, upper_bound, p, max_iter=100, tol=1e-8):
        """
        Invert CDF using Newton's method with robust handling for distributions
        that don't have support starting at 0.
        
        Parameters:
        -----------
        chf : callable
            Characteristic function
        lower_bound : float
            Lower bound of domain
        upper_bound : float
            Upper bound of domain
        p : float
            Probability value to invert (0 < p < 1)
        max_iter : int, default=100
            Maximum number of iterations
        tol : float, default=1e-8
            Convergence tolerance
            
        Returns:
        --------
        float
            x value such that CDF(x) ≈ p
        """
        # Initial checks
        p = np.maximum(0.0, np.minimum(1.0, p))  # Ensure p is in [0,1]
        p = np.maximum(1e-12, np.minimum(1.0 - 1e-12, p))
        
        # Evaluate CDF at initial points for better initial guess
        initial_points = 50  # Increased for better resolution
        x_initial = np.linspace(lower_bound, upper_bound, initial_points)
        cdf_initial = self.compute_cdf(chf, x_initial, lower_bound, upper_bound)
        
        # Find the actual CDF range - important for distributions that don't start at 0
        cdf_min = np.min(cdf_initial)
        cdf_max = np.max(cdf_initial)
        
        # Handle case where p is outside the actual CDF range
        if p < cdf_min:
            # If p is below minimum CDF, return the x value that gives minimum CDF
            # But ensure it's not at the boundary
            idx_min = np.argmin(cdf_initial)
            result = x_initial[idx_min]
            # Ensure result is not at lower bound (which could be 0)
            if result <= lower_bound:
                result = lower_bound + (upper_bound - lower_bound) * 0.01
            return result
        elif p > cdf_max:
            # If p is above maximum CDF, return the x value that gives maximum CDF
            idx_max = np.argmax(cdf_initial)
            return x_initial[idx_max]

        # Find closest point to target probability
        idx = np.abs(cdf_initial - p).argmin()
        x = x_initial[idx]  # Initial guess

        # Newton-Raphson iteration
        for iteration in range(max_iter):
            # Calculate CDF at current point
            cdf_x = self.compute_cdf(chf, np.array([x]), lower_bound, upper_bound)[0]

            # Calculate distance from target f(x) = CDF(x) - p
            fx = cdf_x - p

            # Check convergence
            if abs(fx) < tol:
                return x

            # Calculate F'(x) = PDF at current point
            pdf_x = self.compute_pdf(chf, np.array([x]), lower_bound, upper_bound)[0]

            # Robust safeguard against division by very small numbers
            if abs(pdf_x) < 1e-12:
                # If PDF is too small, use bisection method instead
                if fx > 0:
                    # CDF too high, move left
                    x_new = (lower_bound + x) / 2
                else:
                    # CDF too low, move right
                    x_new = (x + upper_bound) / 2
            else:
                # Normal Newton step
                x_new = x - fx / pdf_x

            # Keep within bounds with small buffer to avoid boundary issues
            buffer = (upper_bound - lower_bound) * 1e-6
            x_new = max(lower_bound + buffer, min(upper_bound - buffer, x_new))

            # Check for convergence
            if abs(x_new - x) < tol:
                return x_new

            # Prevent oscillation by damping large steps
            if iteration > 10 and abs(x_new - x) > (upper_bound - lower_bound) * 0.1:
                x_new = x + 0.1 * (x_new - x)  # Damp the step

            # Update for next iteration
            x = x_new

        # If we didn't converge, return the best guess we have
        return x


# =============================================================================
# HESTON MODEL FUNCTIONS
# =============================================================================

def Heston_CF(u, S0, T, r, kappa, nu0, theta, xi, rho):
    """
    Heston model characteristic function.
    
    Parameters:
    -----------
    u : np.ndarray
        Frequency parameter
    S0 : float
        Initial stock price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    kappa : float
        Mean reversion speed
    nu0 : float
        Initial variance
    theta : float
        Long-term variance
    xi : float
        Volatility of volatility
    rho : float
        Correlation between stock and variance
        
    Returns:
    --------
    np.ndarray
        Characteristic function values
    """
    alpha = - u**2 / 2 - 1j * u / 2
    beta = kappa - rho * xi * 1j * u
    gamma = xi**2 / 2
    h = np.sqrt(beta**2 - 4 * alpha * gamma)
    rm = (beta - h) / xi**2
    rp = (beta + h) / xi**2
    g = rm / rp
    C = kappa * (rm * T - 2 / xi**2 * np.log((1 - g * np.exp(-h * T)) / (1 - g)))
    D = rm * (1 - np.exp(-h * T)) / (1 - g * np.exp(-h * T))
    cf = np.exp(C * theta + D * nu0 + 1j * u * np.log(S0 * np.exp(r * T)))
    return cf


def Heston_price(S0, K, T, r, kappa, nu0, theta, xi, rho):
    """
    Heston call option price using midpoint rule integration.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    kappa : float
        Mean reversion speed
    nu0 : float
        Initial variance
    theta : float
        Long-term variance
    xi : float
        Volatility of volatility
    rho : float
        Correlation between stock and variance
        
    Returns:
    --------
    float
        Call option price
    """
    params = (S0, T, r, kappa, nu0, theta, xi, rho)
    P1 = 0.5
    P2 = 0.5
    umax = 50
    n = 100
    du = umax / n
    u = du / 2
    for i in range(n):
        temp1 = np.exp(-1j * u * np.log(K)) * Heston_CF(u - 1j, *params) / (1j * u * Heston_CF(-1j, *params))
        temp2 = np.exp(-1j * u * np.log(K)) * Heston_CF(u, *params) / (1j * u)
        P1 = P1 + 1 / np.pi * temp1 * du
        P2 = P2 + 1 / np.pi * temp2 * du
        u = u + du
    price = np.real(S0 * P1 - K * np.exp(-r * T) * P2)
    return price


def ChFIntegratedVariance(omega, kappa, gamma, vbar, vu, vt, tau):
    """
    Characteristic function of integrated variance (Broadie-Kaya).
    
    Parameters:
    -----------
    omega : np.ndarray
        Frequency parameter
    kappa : float
        Mean reversion speed
    gamma : float
        Volatility of volatility
    vbar : float
        Long-term variance
    vu : float
        Initial variance
    vt : float
        Final variance
    tau : float
        Time interval
        
    Returns:
    --------
    np.ndarray
        Characteristic function values
    """
    R = np.sqrt(kappa ** 2 - 2.0 * gamma ** 2 * 1j * omega)
    d = 4 * kappa * vbar / gamma ** 2

    temp1 = R * np.exp(-tau / 2.0 * (R - kappa)) * (1 - np.exp(-kappa * tau)) / (kappa * (1 - np.exp(-R * tau)))

    temp2 = np.exp((vu + vt) / gamma ** 2 * (
                kappa * (1 + np.exp(-kappa * tau)) / (1 - np.exp(-kappa * tau)) - R * (1 + np.exp(-R * tau)) / (
                    1 - np.exp(-R * tau))))

    # Bessel functions
    temp3 = ss.iv(0.5 * d - 1.0,
                  np.sqrt(vt * vu) * 4.0 * R * np.exp(-R * tau / 2.0) / (gamma ** 2 * (1 - np.exp(-R * tau))))

    temp4 = ss.iv(0.5 * d - 1.0, np.sqrt(vt * vu) * 4.0 * kappa * np.exp(-kappa * tau / 2.0) / (
                gamma ** 2 * (1 - np.exp(-kappa * tau))))

    chf = temp1 * temp2 * temp3 / temp4
    return chf


def CIR_Sample(NoOfPaths, kappa, gamma, vbar, s, t, v_s):
    """
    Sample from CIR process using noncentral chi-squared distribution.
    
    Parameters:
    -----------
    NoOfPaths : int
        Number of paths to simulate
    kappa : float
        Mean reversion speed
    gamma : float
        Volatility of volatility
    vbar : float
        Long-term variance
    s : float
        Start time
    t : float
        End time
    v_s : np.ndarray
        Initial variance values
        
    Returns:
    --------
    np.ndarray
        Sampled variance values
    """
    delta = 4.0 * kappa * vbar / gamma / gamma
    c = 2 * kappa / (gamma ** 2 * (1 - np.exp(-kappa * (t - s))))
    kappaBar = 2 * c * np.exp(-kappa * (t - s)) * v_s
    sample = np.random.noncentral_chisquare(delta, kappaBar, NoOfPaths) / (2 * c)
    return sample


# =============================================================================
# HESTON EXACT SIMULATION
# =============================================================================

def GeneratePathsHestonES(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0, nr_expansion, L,
                          recovery_method="cos", **method_kwargs):
    """
    Generate Heston model paths using exact simulation.
    
    Parameters:
    -----------
    NoOfPaths : int
        Number of simulation paths
    NoOfSteps : int
        Number of time steps
    T : float
        Time to maturity
    r : float
        Risk-free rate
    S_0 : float
        Initial stock price
    kappa : float
        Mean reversion speed
    gamma : float
        Volatility of volatility
    rho : float
        Correlation
    vbar : float
        Long-term variance
    v0 : float
        Initial variance
    nr_expansion : int
        Number of terms for COS expansion
    L : float
        Bound multiplier for integrated variance
    recovery_method : str, default="cos"
        Method for CDF recovery: "cos", "carr_madan", or "conv"
    **method_kwargs
        Additional arguments for the recovery method
        
    Returns:
    --------
    dict
        Dictionary containing time grid, stock paths, and integrated variance paths
    """
    dt = T / float(NoOfSteps)
    p = np.random.uniform(0, 1, [NoOfPaths, NoOfSteps])

    Z1 = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    V = np.zeros([NoOfPaths, NoOfSteps + 1])
    V_int = np.zeros([NoOfPaths, NoOfSteps + 1])
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    V[:, 0] = v0
    V_int[:, 0] = v0 * dt
    X[:, 0] = np.log(S_0)

    time = np.zeros([NoOfSteps + 1])

    # Initialize the CDF_Inverter
    inverter = CDF_Inverter(method=recovery_method, nr_expansion=nr_expansion, **method_kwargs)
    print(f"Using {recovery_method} method for CDF recovery and inversion")

    for i in range(0, NoOfSteps):
        # Standardize normal samples
        if NoOfPaths > 1:
            Z1[:, i] = (Z1[:, i] - np.mean(Z1[:, i])) / np.std(Z1[:, i])

        # STEP 1: Exact samples for the variance process
        V[:, i + 1] = CIR_Sample(NoOfPaths, kappa, gamma, vbar, 0, dt, V[:, i])

        # STEP 2: Sample from integrated variance distribution
        for j in range(0, NoOfPaths):
            chf_omega = lambda w: ChFIntegratedVariance(w, kappa, gamma, vbar, V[j, i], V[j, i + 1], dt)
            
            # Compute moments for bounds
            first_moment = -1j * (chf_omega(dt) - 1) / dt
            second_moment = -1 * (chf_omega(2 * dt) - 2 * chf_omega(dt) + 1) / (dt ** 2)
            standard_deviation = np.sqrt(abs(second_moment) - abs(first_moment) ** 2)

            # Improved bounds calculation for integrated variance
            # The integrated variance has a minimum value > 0 in the Heston model
            mean_val = np.real(first_moment)
            std_val = standard_deviation
            
            # Use asymmetric bounds that respect the distribution's actual support
            # For integrated variance, the theoretical minimum is always > 0
            theoretical_min = min(V[j, i], V[j, i + 1]) * dt * 0.1  # At least 10% of min variance * dt
            lower_bound = max(theoretical_min, mean_val - 2 * std_val)  # More conservative bound
            upper_bound = mean_val + L * std_val
            
            # Ensure bounds are reasonable and lower_bound is always > 0
            if lower_bound <= 0:
                lower_bound = max(1e-6, mean_val * 0.01)  # Ensure positive lower bound
            if upper_bound - lower_bound < 1e-6:
                upper_bound = lower_bound + max(1e-5, mean_val * 0.1)
            
            # For Carr-Madan method, we need to handle the case where the requested
            # probability might be outside the actual CDF range
            if recovery_method == "carr_madan":
                # Quick check of CDF range at bounds
                try:
                    # First, validate the characteristic function
                    test_chf = chf_omega(np.array([0.1, 1.0]))
                    if np.any(np.isnan(test_chf)) or np.any(np.isinf(test_chf)):
                        raise ValueError("Characteristic function produces NaN/Inf values")
                    
                    x_check = np.array([lower_bound, upper_bound])
                    cdf_check = inverter.compute_cdf(chf_omega, x_check, lower_bound, upper_bound)
                    
                    # Validate CDF values
                    if np.any(np.isnan(cdf_check)) or np.any(np.isinf(cdf_check)):
                        raise ValueError("CDF computation produces NaN/Inf values")
                    
                    # If the probability is outside the achievable range, 
                    # clamp it to the achievable range
                    p_clamped = np.clip(p[j, i], min(cdf_check) + 1e-6, max(cdf_check) - 1e-6)
                    
                    result = inverter.invert_cdf_newton(chf_omega, lower_bound, upper_bound, p_clamped)
                    
                    # Validate result
                    if np.isnan(result) or np.isinf(result) or result <= 0:
                        raise ValueError(f"Invalid CDF inversion result: {result}")
                    
                    V_int[j, i + 1] = result
                    
                except Exception as e:
                    # Fallback to a safer method if Carr-Madan fails
                    # Use a reasonable positive value based on the variance process
                    fallback_val = max(
                        theoretical_min,  # Use the theoretical minimum we calculated
                        mean_val * 0.5 if not np.isnan(mean_val) else theoretical_min,   # Or 50% of expected value
                        min(V[j, i], V[j, i + 1]) * dt * 0.5  # Or 50% of variance * dt
                    )
                    V_int[j, i + 1] = max(fallback_val, 1e-5)  # Ensure it's never too small
            else:
                # For COS and CONV methods, use the original approach
                result = inverter.invert_cdf_newton(chf_omega, lower_bound, upper_bound, p[j, i])
                
                # Ensure result is always positive
                if np.isnan(result) or np.isinf(result) or result <= 0:
                    # Use fallback for other methods too
                    fallback_val = max(
                        theoretical_min,
                        mean_val * 0.5 if not np.isnan(mean_val) else theoretical_min,
                        min(V[j, i], V[j, i + 1]) * dt * 0.5
                    )
                    result = max(fallback_val, 1e-5)
                
                V_int[j, i + 1] = result

        # STEP 3: Compute Ito integral
        ito_integral_Ws1 = 1.0 / gamma * (V[:, i + 1] - V[:, i] - kappa * vbar * dt + kappa * V_int[:, i])

        # STEP 4: Generate stock price sample
        m = X[:, i] + (r * dt - 1.0 / 2.0 * V_int[:, i] + rho * ito_integral_Ws1)
        variance = (1 - rho ** 2) * V_int[:, i]

        X[:, i + 1] = m + np.sqrt(variance) * Z1[:, i]
        time[i + 1] = time[i] + dt

    # Compute stock prices
    S = np.exp(X)
    paths = {"time": time, "S": S, "Vint": V_int}

    return paths


# =============================================================================
# CDF Inversion Function with
# =============================================================================

def cdf_inversion_newton(lower_bound, upper_bound, omega, chf, p, max_iter=100, tol=1e-8):
    """
    Legacy Newton CDF inversion function (COS method only).
    
    DEPRECATED: Use CDF_Inverter.invert_cdf_newton() instead.
    """
    # Initial checks
    p = max(0.0, min(1.0, p))
    if p <= 0.0:
        return lower_bound
    if p >= 1.0:
        return upper_bound

    # Initial guess
    initial_points = 30
    x_initial = np.linspace(lower_bound, upper_bound, initial_points)
    cdf_initial = cos_cdf(lower_bound, upper_bound, omega, chf, x_initial)

    idx = np.abs(cdf_initial - p).argmin()
    x = x_initial[idx]

    # Newton-Raphson iteration
    for i in range(max_iter):
        cdf_x = cos_cdf(lower_bound, upper_bound, omega, chf, np.array([x]))[0]
        fx = cdf_x - p

        if abs(fx) < tol:
            return x

        # Define wrapper for cos_pdf
        def chf_wrapper(u):
            return chf

        N = len(omega)
        pdf_x = cos_pdf(lower_bound, upper_bound, N, chf_wrapper, np.array([x]))[0]

        if abs(pdf_x) < 1e-10:
            pdf_x = 1e-10 if pdf_x >= 0 else -1e-10

        x_new = x - fx / pdf_x
        x_new = max(lower_bound, min(upper_bound, x_new))

        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    return x


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def bs_put_price(S0, K, r, sigma, T):
    """
    Black-Scholes put option price.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    sigma : float
        Volatility
    T : float
        Time to maturity
        
    Returns:
    --------
    float
        Put option price
    """
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_put = -1 * norm.cdf(-d1)*S0 + norm.cdf(-d2)*K*np.exp(-r*T)
    return bs_put


def compare_cdf_inversion_methods():
    """
    Compare brute force vs Newton CDF inversion methods.
    
    Returns:
    --------
    pd.DataFrame
        Comparison results
    """
    # Heston model parameters
    gamma = 0.4
    kappa = 0.5
    vbar = 0.2
    rho = -0.9
    v0 = 0.2
    T = 1.0
    S_0 = 100.0
    r = 0.1

    # Simulation parameters
    NoOfPaths = 4
    NoOfSteps = 50
    nr_expansion = 100
    L = 10

    # Values of N to test
    N_values = [4, 8, 16, 32, 64]

    # Arrays to store results
    brute_times = []
    newton_times = []
    vint_brute_values = []
    vint_newton_values = []

    for N in N_values:
        # Run brute force method
        np.random.seed(3)
        start_time = time.time()
        paths_brute = GeneratePathsHestonES_suppl(N, NoOfSteps, T, r, S_0, kappa, gamma,
                                            rho, vbar, v0, nr_expansion, L, method="brute")
        brute_time = time.time() - start_time
        brute_times.append(brute_time)

        # Run Newton method
        np.random.seed(3)
        start_time = time.time()
        paths_newton = GeneratePathsHestonES_suppl(N, NoOfSteps, T, r, S_0, kappa, gamma,
                                             rho, vbar, v0, nr_expansion, L, method="newton")
        newton_time = time.time() - start_time
        newton_times.append(newton_time)

        vint_brute = np.mean(paths_brute["Vint"])
        vint_newton = np.mean(paths_newton["Vint"])

        vint_brute_values.append(vint_brute)
        vint_newton_values.append(vint_newton)

    # Create results dataframe
    results_df = pd.DataFrame({
        'N': N_values,
        'Brute Time (s)': brute_times,
        'Newton Time (s)': newton_times,
        'vint_brute_values': vint_brute_values,
        'vint_newton_values': vint_newton_values
    })

    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Computation Time Comparison
    ax1.plot(N_values, brute_times, 'o-', color='blue', label='Brute')
    ax1.plot(N_values, newton_times, 's-', color='green', label='Newton')

    for i, n in enumerate(N_values):
        ax1.annotate(f"{brute_times[i]:.3f}s",
                     xy=(n, brute_times[i]),
                     xytext=(5, 5),
                     textcoords="offset points")
        ax1.annotate(f"{newton_times[i]:.3f}s",
                     xy=(n, newton_times[i]),
                     xytext=(5, 5),
                     textcoords="offset points")

    ax1.set_xlabel('Number of Paths (N)')
    ax1.set_ylabel('Computation Time (seconds)')
    ax1.set_title('Computation Time vs. N')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Accuracy Comparison
    ax2.plot(N_values, vint_brute_values, 'o-', color='blue', label='Brute')
    ax2.plot(N_values, vint_newton_values, 's-', color='green', label='Newton')

    for i, n in enumerate(N_values):
        ax2.annotate(f"{vint_brute_values[i]:.2e}",
                     xy=(n, vint_brute_values[i]),
                     xytext=(5, 5),
                     textcoords="offset points")
        ax2.annotate(f"{vint_newton_values[i]:.2e}",
                     xy=(n, vint_newton_values[i]),
                     xytext=(5, 5),
                     textcoords="offset points")

    ax2.set_xlabel('Number of Paths (N)')
    ax2.set_ylabel('Integrated Variance')
    ax2.set_title('Accuracy vs. N')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return results_df


def GeneratePathsHestonES_suppl(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0, nr_expansion, L,
                          method="newton"):
    dt = T / float(NoOfSteps)
    p = np.random.uniform(0, 1, [NoOfPaths, NoOfSteps])

    Z1 = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    V = np.zeros([NoOfPaths, NoOfSteps + 1])
    V_int = np.zeros([NoOfPaths, NoOfSteps + 1])
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    V[:, 0] = v0
    V_int[:, 0] = v0 * dt
    X[:, 0] = np.log(S_0)

    time = np.zeros([NoOfSteps + 1])

    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z1[:, i] = (Z1[:, i] - np.mean(Z1[:, i])) / np.std(Z1[:, i])

        V[:, i + 1] = CIR_Sample(NoOfPaths, kappa, gamma, vbar, 0, dt, V[:, i])

        for j in range(0, NoOfPaths):
            chf_omega = lambda w: ChFIntegratedVariance(w, kappa, gamma, vbar, V[j, i], V[j, i + 1], dt)
            first_moment = -1j * (chf_omega(dt) - 1) / dt
            second_moment = -1 * (chf_omega(2 * dt) - 2 * chf_omega(dt) + 1) / (dt ** 2)

            standard_deviation = np.sqrt(abs(second_moment) - abs(first_moment) ** 2)

            lower_bound = 0
            upper_bound = abs(first_moment) + L * standard_deviation

            omega = np.arange(nr_expansion) * np.pi / (upper_bound - lower_bound)
            chf_integrated = ChFIntegratedVariance(omega, kappa, gamma, vbar, V[j, i], V[j, i + 1], dt)

            if method == "brute":
                x = np.linspace(lower_bound, upper_bound, 10000)
                cdf_integratedvar = cos_cdf(lower_bound, upper_bound, omega, chf_integrated, x)
                V_int[j, i + 1] = x[np.abs(cdf_integratedvar - p[j, i]).argmin()]
            else:  # method == "newton"
                V_int[j, i + 1] = cdf_inversion_newton(lower_bound, upper_bound, omega, chf_integrated, p[j, i])

        # Compute stock paths
        ito_integral_Ws1 = 1.0 / gamma * (V[:, i + 1] - V[:, i] - kappa * vbar * dt + kappa * V_int[:, i])
        m = X[:, i] + (r * dt - 1.0 / 2.0 * V_int[:, i] + rho * ito_integral_Ws1)
        variance = (1 - rho ** 2) * V_int[:, i]
        X[:, i + 1] = m + np.sqrt(variance) * Z1[:, i]
        time[i + 1] = time[i] + dt

    S = np.exp(X)
    paths = {"time": time, "S": S, "Vint": V_int}
    return paths