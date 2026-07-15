# Quantitative Finance Book

A Jupyter Book covering option pricing, stochastic volatility, and hedging. Implementations use characteristic function methods (COS, Carr-Madan, CONV), Monte Carlo simulation, and finite differences. Some notebooks use [cppfm](https://github.com/libapatrik/Cpp-Option-Pricing), a personal C++ option pricing library with Python bindings.

**Read online:** [libapatrik.github.io/Quantitative-Finance-Book](https://libapatrik.github.io/Quantitative-Finance-Book/)

## Highlights

| Chapter | Result |
|---------|--------|
| [COS Density Recovery](https://libapatrik.github.io/Quantitative-Finance-Book/char_functions/01_pdf_cdf_recovery.html) | Exponential convergence — machine precision (3.6e-16) at N=64 vs N=256+ for Carr-Madan/CONV |
| [Heston Exact Simulation](https://libapatrik.github.io/Quantitative-Finance-Book/char_functions/02_heston_exact_simulation.html) | Broadie-Kaya scheme with branch-cut fix (Albrecher 2007); CDF inversion via COS |
| [Heston Calibration](https://libapatrik.github.io/Quantitative-Finance-Book/calibration/01_heston_calibration.html) | LM exploits residual structure — 6 iterations / 0.08s in C++ vs 25 / 0.37s for SLSQP |
| [SSVI Calibration](https://libapatrik.github.io/Quantitative-Finance-Book/calibration/02_ssvi.html) | Fit to live SPX (5,661 strikes, 40 maturities); all butterfly + calendar arbitrage checks pass |
| [HestonSLV Calibration](https://libapatrik.github.io/Quantitative-Finance-Book/calibration/03_heston_slv_calibration.html) | Full pipeline: SSVI → Heston → leverage function → 200k-path MC in 4.5s |
| [COS American Pricing](https://libapatrik.github.io/Quantitative-Finance-Book/american_options/04_cos_american.html) | FFT acceleration O(N²) → O(N log N); 16× speedup at N=1024, spectral convergence in N |
| [Finite Difference Pricing](https://libapatrik.github.io/Quantitative-Finance-Book/american_options/03_finite_difference.html) | Implicit scheme with sparse LU — 40× faster than dense (0.5s vs 20s per option) |
| [Delta Hedging under Heston](https://libapatrik.github.io/Quantitative-Finance-Book/trading/02_heston_hedging_pnl.html) | BS vs Heston deltas nearly identical — unhedgeable vol-of-vol dominates hedge P&L |

## Contents

**Characteristic Function Methods**
- [CDF/PDF Recovery: COS, Carr-Madan, CONV](https://libapatrik.github.io/Quantitative-Finance-Book/char_functions/01_pdf_cdf_recovery.html)
- [Exact Simulation of the Heston Model](https://libapatrik.github.io/Quantitative-Finance-Book/char_functions/02_heston_exact_simulation.html)

**Calibration**
- [Heston Calibration](https://libapatrik.github.io/Quantitative-Finance-Book/calibration/01_heston_calibration.html)
- [SSVI Implied Volatility Surface](https://libapatrik.github.io/Quantitative-Finance-Book/calibration/02_ssvi.html)
- [HestonSLV Calibration and Simulation](https://libapatrik.github.io/Quantitative-Finance-Book/calibration/03_heston_slv_calibration.html)

**Pricing American Options**
- [Intuition behind Longstaff-Schwartz](https://libapatrik.github.io/Quantitative-Finance-Book/american_options/01_lsmc_intuition.html)
- [Least-Square Monte Carlo](https://libapatrik.github.io/Quantitative-Finance-Book/american_options/02_lsmc_pricing.html)
- [Finite Difference Methods](https://libapatrik.github.io/Quantitative-Finance-Book/american_options/03_finite_difference.html)
- [COS Method for American Options](https://libapatrik.github.io/Quantitative-Finance-Book/american_options/04_cos_american.html)
- [COS Method for Barrier Options](https://libapatrik.github.io/Quantitative-Finance-Book/american_options/05_cos_barrier.html)

**Hedging and Greeks**
- [Greeks in Black-Scholes](https://libapatrik.github.io/Quantitative-Finance-Book/greeks/01_greeks_black_scholes.html)
- [Greeks via COS Method](https://libapatrik.github.io/Quantitative-Finance-Book/greeks/02_greeks_cos.html)

**Trading and Hedging**
- [Hedging P&L under Black-Scholes](https://libapatrik.github.io/Quantitative-Finance-Book/trading/01_bs_hedging_pnl.html)
- [Hedging P&L under Heston](https://libapatrik.github.io/Quantitative-Finance-Book/trading/02_heston_hedging_pnl.html)
- [Hedging P&L under HestonSLV](https://libapatrik.github.io/Quantitative-Finance-Book/trading/03_hestonSLV_hedging_pnl.html)

**Volatility Forecasting**
- [Predicting Realized Volatility](https://libapatrik.github.io/Quantitative-Finance-Book/volatility/01_pred_realized_vol.html)

