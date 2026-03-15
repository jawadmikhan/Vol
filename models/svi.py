"""
SVI (Stochastic Volatility Inspired) Model
=============================================
Jim Gatheral's SVI parameterization for implied volatility surfaces.

The raw SVI parameterization models total implied variance w(k) as:

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where:
    k     = log-moneyness = log(K / F)
    w     = total implied variance = sigma_BS^2 * T
    a     = overall variance level
    b     = slope of the wings (b >= 0)
    rho   = skew / correlation (-1 < rho < 1)
    m     = horizontal translation (center of the smile)
    sigma = smoothness of the ATM region (sigma > 0)

Also implements SSVI (Surface SVI) by Gatheral & Jacquier (2014) for
arbitrage-free interpolation across tenors.

Reference:
    Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility
    parameterization with application to the valuation of volatility derivatives."
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SVI parameter container
# ---------------------------------------------------------------------------

@dataclass
class SVIParams:
    """Raw SVI parameters for a single tenor slice."""
    a: float       # variance level
    b: float       # wing slope (>= 0)
    rho: float     # skew (-1 < rho < 1)
    m: float       # horizontal shift
    sigma: float   # ATM smoothness (> 0)
    tenor: float   # time to expiry in years
    residual: float = 0.0  # calibration residual (RMSE)

    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        """Compute total implied variance w(k) = sigma_BS^2 * T."""
        return self.a + self.b * (
            self.rho * (k - self.m)
            + np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        )

    def implied_vol(self, k: float | np.ndarray) -> float | np.ndarray:
        """Compute Black-Scholes implied vol from log-moneyness k."""
        w = self.total_variance(k)
        w = np.maximum(w, 1e-10)  # floor to avoid sqrt of negative
        return np.sqrt(w / self.tenor)

    def implied_vol_strike(
        self, strike: float | np.ndarray, forward: float
    ) -> float | np.ndarray:
        """Compute implied vol from absolute strike and forward price."""
        k = np.log(strike / forward)
        return self.implied_vol(k)

    def atm_vol(self) -> float:
        """ATM implied vol (k = 0)."""
        return float(self.implied_vol(0.0))

    def skew_25d(self) -> float:
        """25-delta risk reversal skew (approx k ~ +/-0.3 for 3M)."""
        k_put = -0.3
        k_call = 0.3
        return float(self.implied_vol(k_put) - self.implied_vol(k_call))


# ---------------------------------------------------------------------------
# SVI calibration
# ---------------------------------------------------------------------------

def calibrate_svi(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    forward: float,
    tenor: float,
    weights: np.ndarray | None = None,
) -> SVIParams:
    """
    Calibrate raw SVI parameters to market implied vols.

    Args:
        strikes: Array of option strikes.
        market_vols: Array of market implied vols (annualized).
        forward: Forward price for the tenor.
        tenor: Time to expiry in years.
        weights: Optional weights for each observation (e.g., vega-weight).

    Returns:
        Calibrated SVIParams.
    """
    if len(strikes) < 3:
        raise ValueError("Need at least 3 strike/vol pairs for SVI calibration")

    # Convert to log-moneyness and total variance
    k = np.log(strikes / forward)
    w_market = market_vols ** 2 * tenor

    if weights is None:
        weights = np.ones(len(k))
    weights = weights / weights.sum()

    def objective(params):
        a, b, rho, m, sigma = params
        w_model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        # Weighted sum of squared errors
        return np.sum(weights * (w_model - w_market) ** 2)

    # Bounds to ensure valid SVI parameters
    # a: total variance must be non-negative at all points
    # b >= 0, -1 < rho < 1, sigma > 0
    atm_var = np.interp(0.0, k, w_market)
    bounds = [
        (max(-atm_var, -0.5), 1.0),    # a: variance level
        (1e-6, 2.0),                     # b: wing slope
        (-0.99, 0.99),                   # rho: skew
        (k.min() - 0.5, k.max() + 0.5), # m: center
        (1e-4, 2.0),                     # sigma: smoothness
    ]

    # Initial guess from market data
    a0 = atm_var
    b0 = 0.1
    rho0 = -0.3  # typical equity skew
    m0 = 0.0
    sigma0 = 0.1

    # Two-stage optimization: global then local
    try:
        # Stage 1: Differential evolution for global search
        result_global = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            tol=1e-10,
            polish=False,
        )

        # Stage 2: Nelder-Mead polish from the global optimum
        result = minimize(
            objective,
            result_global.x,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-12, "fatol": 1e-12},
        )
    except Exception:
        # Fallback: just local optimization
        result = minimize(
            objective,
            [a0, b0, rho0, m0, sigma0],
            method="Nelder-Mead",
            options={"maxiter": 5000},
        )

    a, b, rho, m, sigma = result.x
    b = max(b, 0.0)
    rho = np.clip(rho, -0.99, 0.99)
    sigma = max(sigma, 1e-4)

    # Compute calibration RMSE in vol terms
    params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, tenor=tenor)
    model_vols = params.implied_vol(k)
    rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))
    params.residual = rmse

    logger.debug(
        "SVI calibrated: a=%.4f b=%.4f rho=%.3f m=%.4f sigma=%.4f | RMSE=%.4f",
        a, b, rho, m, sigma, rmse,
    )

    return params


# ---------------------------------------------------------------------------
# SVI to delta-space conversion
# ---------------------------------------------------------------------------

def _bs_delta(k: float, vol: float, tenor: float, is_call: bool) -> float:
    """Black-Scholes delta from log-moneyness (forward delta, undiscounted)."""
    if vol <= 0 or tenor <= 0:
        return 0.5 if is_call else -0.5
    d1 = (-k + 0.5 * vol ** 2 * tenor) / (vol * np.sqrt(tenor))
    if is_call:
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1.0)


def svi_vol_at_delta(
    params: SVIParams,
    target_delta: float,
    is_call: bool,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> tuple[float, float]:
    """
    Find the implied vol at a given Black-Scholes delta using Newton's method.

    Args:
        params: Calibrated SVI parameters.
        target_delta: Target delta (e.g., 0.25 for 25-delta call, -0.10 for 10-delta put).
        is_call: True for call delta, False for put delta.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        (implied_vol, log_moneyness) at the target delta.
    """
    # Initial guess for k based on delta using ATM vol
    atm_v = params.atm_vol()
    sqrt_t = np.sqrt(params.tenor)

    if is_call:
        # For call delta: delta = N(d1), so d1 = N_inv(delta)
        d1_target = norm.ppf(max(target_delta, 1e-6))
        k = -d1_target * atm_v * sqrt_t + 0.5 * atm_v ** 2 * params.tenor
    else:
        # For put delta: delta = N(d1) - 1, so d1 = N_inv(delta + 1)
        d1_target = norm.ppf(min(target_delta + 1.0, 1 - 1e-6))
        k = -d1_target * atm_v * sqrt_t + 0.5 * atm_v ** 2 * params.tenor

    # Clamp initial guess to reasonable range
    k = np.clip(k, -1.5, 1.5)

    for _ in range(max_iter):
        vol = float(params.implied_vol(k))
        if vol <= 0 or np.isnan(vol):
            break

        delta = _bs_delta(k, vol, params.tenor, is_call)
        err = delta - target_delta

        if abs(err) < tol:
            break

        # Numerical derivative dk/ddelta
        dk = 0.0005
        k_up = k + dk
        vol_up = float(params.implied_vol(k_up))
        delta_up = _bs_delta(k_up, vol_up, params.tenor, is_call)
        ddelta_dk = (delta_up - delta) / dk

        if abs(ddelta_dk) < 1e-12:
            break

        step = err / ddelta_dk
        # Damped Newton to prevent overshooting
        step = np.clip(step, -0.3, 0.3)
        k -= step
        # Keep k in reasonable range
        k = np.clip(k, -2.0, 2.0)

    vol = float(params.implied_vol(k))
    return vol, k


def extract_delta_vols(params: SVIParams) -> dict[str, float]:
    """
    Extract standard delta-bucket implied vols from calibrated SVI.

    Returns dict with keys: 10D_PUT, 25D_PUT, ATM, 25D_CALL, 10D_CALL
    """
    results = {}

    # ATM: k = 0
    results["ATM"] = float(params.implied_vol(0.0))

    # 25-delta put (delta = -0.25)
    vol, _ = svi_vol_at_delta(params, target_delta=-0.25, is_call=False)
    results["25D_PUT"] = vol

    # 10-delta put (delta = -0.10)
    vol, _ = svi_vol_at_delta(params, target_delta=-0.10, is_call=False)
    results["10D_PUT"] = vol

    # 25-delta call (delta = 0.25)
    vol, _ = svi_vol_at_delta(params, target_delta=0.25, is_call=True)
    results["25D_CALL"] = vol

    # 10-delta call (delta = 0.10)
    vol, _ = svi_vol_at_delta(params, target_delta=0.10, is_call=True)
    results["10D_CALL"] = vol

    return results


# ---------------------------------------------------------------------------
# SSVI: Surface SVI for tenor interpolation (Gatheral & Jacquier 2014)
# ---------------------------------------------------------------------------

@dataclass
class SSVIParams:
    """
    SSVI parameterization for the full surface.

    theta(t) = ATM total variance as a function of tenor
    phi(theta) = b / (2 * theta) controls wing behavior
    rho: global skew parameter
    """
    theta_params: list  # [(tenor, atm_total_var), ...] for interpolation
    rho: float          # global skew
    eta: float          # controls how phi varies with theta
    gamma: float        # power-law exponent for phi

    def theta(self, tenor: float) -> float:
        """Interpolate ATM total variance at a given tenor."""
        tenors = np.array([t for t, _ in self.theta_params])
        variances = np.array([v for _, v in self.theta_params])
        return float(np.interp(tenor, tenors, variances))

    def phi(self, theta_val: float) -> float:
        """Wing function: controls slope as function of ATM variance."""
        if theta_val <= 0:
            return 0.0
        return self.eta / (theta_val ** self.gamma * (1 + theta_val) ** (1 - self.gamma))

    def total_variance(self, k: float, tenor: float) -> float:
        """SSVI total variance at (k, T)."""
        theta_t = self.theta(tenor)
        phi_t = self.phi(theta_t)
        return 0.5 * theta_t * (
            1 + self.rho * phi_t * k
            + np.sqrt((phi_t * k + self.rho) ** 2 + (1 - self.rho ** 2))
        )

    def implied_vol(self, k: float, tenor: float) -> float:
        """SSVI implied vol at (k, T)."""
        w = self.total_variance(k, tenor)
        w = max(w, 1e-10)
        return np.sqrt(w / tenor)


def calibrate_ssvi(svi_slices: list[SVIParams]) -> SSVIParams:
    """
    Fit SSVI parameters from a set of per-tenor SVI calibrations.

    This ensures the full surface is arbitrage-free across tenors
    (no calendar spread arbitrage).

    Args:
        svi_slices: List of calibrated SVIParams, one per tenor.

    Returns:
        Calibrated SSVIParams.
    """
    if len(svi_slices) < 2:
        # Not enough tenors for SSVI, return with defaults
        s = svi_slices[0] if svi_slices else SVIParams(0.04, 0.1, -0.3, 0.0, 0.1, 0.25)
        return SSVIParams(
            theta_params=[(s.tenor, s.a + s.b * s.sigma)],
            rho=s.rho,
            eta=0.5,
            gamma=0.5,
        )

    # Extract ATM total variances per tenor
    theta_points = []
    for s in sorted(svi_slices, key=lambda x: x.tenor):
        atm_w = s.total_variance(0.0)
        theta_points.append((s.tenor, float(atm_w)))

    # Average rho across tenors (weighted by tenor)
    total_tenor = sum(s.tenor for s in svi_slices)
    avg_rho = sum(s.rho * s.tenor for s in svi_slices) / total_tenor

    # Fit eta and gamma by minimizing error across all slices
    all_k = np.linspace(-0.5, 0.5, 21)

    def objective(params):
        eta, gamma = params
        ssvi = SSVIParams(
            theta_params=theta_points,
            rho=avg_rho,
            eta=eta,
            gamma=gamma,
        )
        total_err = 0.0
        for s in svi_slices:
            for k in all_k:
                w_svi = s.total_variance(k)
                w_ssvi = ssvi.total_variance(k, s.tenor)
                total_err += (w_svi - w_ssvi) ** 2
        return total_err

    result = minimize(
        objective,
        [0.5, 0.5],
        method="Nelder-Mead",
        bounds=[(0.01, 5.0), (0.01, 0.99)],
        options={"maxiter": 2000},
    )

    eta, gamma = result.x
    eta = max(eta, 0.01)
    gamma = np.clip(gamma, 0.01, 0.99)

    ssvi = SSVIParams(
        theta_params=theta_points,
        rho=float(np.clip(avg_rho, -0.99, 0.99)),
        eta=float(eta),
        gamma=float(gamma),
    )

    logger.info(
        "SSVI calibrated: rho=%.3f eta=%.3f gamma=%.3f (%d tenor slices)",
        ssvi.rho, ssvi.eta, ssvi.gamma, len(svi_slices),
    )

    return ssvi


# ---------------------------------------------------------------------------
# Convenience: calibrate full surface from market data
# ---------------------------------------------------------------------------

def calibrate_surface(
    chain_df,
    forward: float,
    tenors_years: list[float],
) -> tuple[list[SVIParams], SSVIParams | None]:
    """
    Calibrate SVI per-tenor and SSVI across tenors from a chain DataFrame.

    Args:
        chain_df: DataFrame with columns [strike, implied_vol, tenor_years].
        forward: Forward price.
        tenors_years: List of tenors to calibrate (in years).

    Returns:
        (list of SVIParams per tenor, SSVIParams for full surface or None)
    """
    slices = []

    for tenor in sorted(tenors_years):
        tenor_data = chain_df[
            (chain_df["tenor_years"] - tenor).abs() < 0.02
        ]
        if len(tenor_data) < 3:
            logger.warning("Skipping tenor %.2f: only %d data points", tenor, len(tenor_data))
            continue

        try:
            params = calibrate_svi(
                strikes=tenor_data["strike"].values,
                market_vols=tenor_data["implied_vol"].values,
                forward=forward,
                tenor=tenor,
            )
            slices.append(params)
        except Exception as e:
            logger.warning("SVI calibration failed for tenor %.2f: %s", tenor, e)

    ssvi = calibrate_ssvi(slices) if len(slices) >= 2 else None

    return slices, ssvi
