"""
Improved Regime Classifier
=============================
Multi-model regime detection replacing the simple threshold-based approach.

Models:
  1. Rules-Based (original): VIX thresholds + term structure + VVIX
  2. Hidden Markov Model (HMM): learns regime transitions from data
  3. Composite: weighted blend of rules + HMM for robustness

Regimes:
  - LOW_VOL_HARVESTING: low vol, contango, calm markets
  - TRANSITIONAL: moderate vol, mixed signals
  - CRISIS: high vol, backwardation, elevated VVIX

Calibrated to historical VIX distribution:
  - ~60-70% of time in LOW_VOL or TRANSITIONAL
  - ~10-15% of time in CRISIS
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

REGIMES = ["LOW_VOL_HARVESTING", "TRANSITIONAL", "CRISIS"]


# ---------------------------------------------------------------------------
# 1. Rules-Based Classifier (improved with calibrated thresholds)
# ---------------------------------------------------------------------------

@dataclass
class RulesConfig:
    """Calibrated thresholds for the rules-based classifier."""
    # VIX thresholds (calibrated to historical distribution)
    vix_low: float = 15.0          # Below = low vol
    vix_high: float = 22.0         # Above = crisis candidate

    # Term structure (front-to-second month ratio)
    contango_threshold: float = 0.02   # > 2% = contango (normal)
    backwardation_threshold: float = -0.01  # < -1% = backwardation (stress)

    # VVIX (vol of vol)
    vvix_low: float = 80.0
    vvix_high: float = 110.0

    # RV/IV spread
    rv_iv_crisis_threshold: float = 0.02    # RV > IV by 2% = vol breakout

    # Scoring weights
    weight_vix: float = 2.0
    weight_term: float = 1.5
    weight_vvix: float = 1.0
    weight_rv_iv: float = 1.0


def classify_rules(
    vix: float,
    term_slope: float,
    vvix: float | None,
    rv_iv_spread: float,
    config: RulesConfig | None = None,
) -> tuple[str, dict]:
    """
    Improved rules-based classifier with scoring.

    Returns:
        (regime, scores_dict)
    """
    if config is None:
        config = RulesConfig()

    crisis_score = 0.0
    low_vol_score = 0.0

    # VIX contribution
    if vix > config.vix_high:
        crisis_score += config.weight_vix * min((vix - config.vix_high) / 10, 2.0)
    elif vix < config.vix_low:
        low_vol_score += config.weight_vix * min((config.vix_low - vix) / 5, 2.0)

    # Term structure
    if term_slope < config.backwardation_threshold:
        crisis_score += config.weight_term
    elif term_slope > config.contango_threshold:
        low_vol_score += config.weight_term * 0.5

    # VVIX
    if vvix is not None:
        if vvix > config.vvix_high:
            crisis_score += config.weight_vvix
        elif vvix < config.vvix_low:
            low_vol_score += config.weight_vvix * 0.5

    # RV/IV spread
    if rv_iv_spread > config.rv_iv_crisis_threshold:
        crisis_score += config.weight_rv_iv

    scores = {
        "crisis_score": crisis_score,
        "low_vol_score": low_vol_score,
    }

    if crisis_score >= 2.5:
        return "CRISIS", scores
    elif low_vol_score >= 2.5:
        return "LOW_VOL_HARVESTING", scores
    else:
        return "TRANSITIONAL", scores


# ---------------------------------------------------------------------------
# 2. Hidden Markov Model (Gaussian HMM)
# ---------------------------------------------------------------------------

class GaussianHMM:
    """
    Simple 3-state Gaussian HMM for regime detection.

    Fitted via Baum-Welch (EM algorithm) on VIX returns and term structure.

    States map to: 0=LOW_VOL, 1=TRANSITIONAL, 2=CRISIS
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter = n_iter

        # Initialize with reasonable priors for vol regimes
        self.means = np.array([-0.01, 0.0, 0.03])      # VIX daily return means
        self.stds = np.array([0.03, 0.05, 0.10])        # VIX daily return stds
        self.transition = np.array([                      # Transition matrix
            [0.95, 0.04, 0.01],  # LOW_VOL: sticky, rare jump to crisis
            [0.05, 0.90, 0.05],  # TRANSITIONAL: can go either way
            [0.02, 0.08, 0.90],  # CRISIS: sticky, slow to de-escalate
        ])
        self.initial = np.array([0.40, 0.45, 0.15])     # Prior state probabilities
        self.fitted = False

    def fit(self, observations: np.ndarray):
        """
        Fit HMM parameters using Expectation-Maximization.

        Args:
            observations: 1D array of VIX daily returns (or changes).
        """
        n = len(observations)
        if n < 20:
            logger.warning("Not enough data for HMM fitting (%d obs)", n)
            return

        # EM algorithm (simplified Baum-Welch)
        for iteration in range(self.n_iter):
            # E-step: forward-backward to compute state probabilities
            alpha, scale = self._forward(observations)
            beta = self._backward(observations, scale)
            gamma = alpha * beta
            gamma = gamma / gamma.sum(axis=1, keepdims=True)

            # Compute xi (transition probabilities)
            xi = np.zeros((n - 1, self.n_states, self.n_states))
            for t in range(n - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (
                            alpha[t, i]
                            * self.transition[i, j]
                            * self._emission_prob(observations[t + 1], j)
                            * beta[t + 1, j]
                        )
                xi_sum = xi[t].sum()
                if xi_sum > 0:
                    xi[t] /= xi_sum

            # M-step: update parameters
            # Transition matrix
            for i in range(self.n_states):
                gamma_sum = gamma[:-1, i].sum()
                if gamma_sum > 0:
                    for j in range(self.n_states):
                        self.transition[i, j] = xi[:, i, j].sum() / gamma_sum

            # Emission parameters (mean, std)
            for j in range(self.n_states):
                weight = gamma[:, j]
                total_weight = weight.sum()
                if total_weight > 0:
                    self.means[j] = (weight * observations).sum() / total_weight
                    diff = observations - self.means[j]
                    self.stds[j] = max(
                        np.sqrt((weight * diff ** 2).sum() / total_weight),
                        1e-4,
                    )

            # Initial probabilities
            self.initial = gamma[0]

        # Sort states by mean (ensure LOW_VOL < TRANSITIONAL < CRISIS)
        order = np.argsort(self.means)
        self.means = self.means[order]
        self.stds = self.stds[order]
        self.transition = self.transition[np.ix_(order, order)]
        self.initial = self.initial[order]

        self.fitted = True
        logger.info(
            "HMM fitted: means=[%.4f, %.4f, %.4f] stds=[%.4f, %.4f, %.4f]",
            *self.means, *self.stds,
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding: find the most likely state sequence.

        Returns array of state indices (0=LOW_VOL, 1=TRANSITIONAL, 2=CRISIS).
        """
        n = len(observations)
        delta = np.zeros((n, self.n_states))
        psi = np.zeros((n, self.n_states), dtype=int)

        # Initialization
        for j in range(self.n_states):
            delta[0, j] = np.log(max(self.initial[j], 1e-20)) + np.log(
                max(self._emission_prob(observations[0], j), 1e-20)
            )

        # Recursion
        for t in range(1, n):
            for j in range(self.n_states):
                candidates = delta[t - 1] + np.log(np.maximum(self.transition[:, j], 1e-20))
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + np.log(
                    max(self._emission_prob(observations[t], j), 1e-20)
                )

        # Backtracking
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute posterior state probabilities via forward-backward.

        Returns (n, n_states) array of probabilities.
        """
        alpha, scale = self._forward(observations)
        beta = self._backward(observations, scale)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        return gamma

    def current_regime(self, recent_obs: np.ndarray) -> tuple[str, np.ndarray]:
        """
        Classify the current regime from recent observations.

        Returns (regime_name, probability_vector).
        """
        if len(recent_obs) < 2:
            return "TRANSITIONAL", np.array([0.33, 0.34, 0.33])

        proba = self.predict_proba(recent_obs)
        latest_proba = proba[-1]
        state_idx = np.argmax(latest_proba)

        return REGIMES[state_idx], latest_proba

    def _emission_prob(self, obs: float, state: int) -> float:
        """Gaussian emission probability."""
        return norm.pdf(obs, self.means[state], self.stds[state])

    def _forward(self, observations):
        """Forward algorithm with scaling."""
        n = len(observations)
        alpha = np.zeros((n, self.n_states))
        scale = np.zeros(n)

        for j in range(self.n_states):
            alpha[0, j] = self.initial[j] * self._emission_prob(observations[0], j)
        scale[0] = alpha[0].sum()
        if scale[0] > 0:
            alpha[0] /= scale[0]

        for t in range(1, n):
            for j in range(self.n_states):
                alpha[t, j] = sum(
                    alpha[t - 1, i] * self.transition[i, j]
                    for i in range(self.n_states)
                ) * self._emission_prob(observations[t], j)
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]

        return alpha, scale

    def _backward(self, observations, scale):
        """Backward algorithm with scaling."""
        n = len(observations)
        beta = np.zeros((n, self.n_states))
        beta[-1] = 1.0

        for t in range(n - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = sum(
                    self.transition[i, j]
                    * self._emission_prob(observations[t + 1], j)
                    * beta[t + 1, j]
                    for j in range(self.n_states)
                )
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]

        return beta


# ---------------------------------------------------------------------------
# 3. Composite Classifier
# ---------------------------------------------------------------------------

class CompositeRegimeClassifier:
    """
    Combines rules-based and HMM classifiers.

    The composite approach:
    - Uses HMM posterior probabilities for base regime estimate
    - Overrides with rules when market signals are extreme
    - Provides confidence scores

    Usage:
        clf = CompositeRegimeClassifier()
        clf.fit(vix_returns)  # Train HMM on historical data
        regime, confidence, detail = clf.classify(vix=18.5, term_slope=0.03, ...)
    """

    def __init__(
        self,
        hmm_weight: float = 0.6,
        rules_weight: float = 0.4,
        rules_config: RulesConfig | None = None,
    ):
        self.hmm = GaussianHMM(n_states=3)
        self.hmm_weight = hmm_weight
        self.rules_weight = rules_weight
        self.rules_config = rules_config or RulesConfig()
        self._recent_vix_returns = []

    def fit(self, vix_series: np.ndarray | pd.Series):
        """
        Train the HMM on a VIX time series.

        Args:
            vix_series: Daily VIX levels (will compute returns internally).
        """
        vix = np.asarray(vix_series, dtype=float)
        vix = vix[~np.isnan(vix)]

        if len(vix) < 30:
            logger.warning("Not enough VIX data for HMM training (%d)", len(vix))
            return

        # Compute daily VIX returns
        returns = np.diff(np.log(vix))
        self.hmm.fit(returns)
        self._recent_vix_returns = list(returns[-60:])

    def classify(
        self,
        vix: float,
        term_slope: float,
        vvix: float | None = None,
        rv_iv_spread: float = 0.0,
        prev_vix: float | None = None,
    ) -> tuple[str, float, dict]:
        """
        Classify the current regime.

        Args:
            vix: Current VIX level.
            term_slope: VIX term structure slope.
            vvix: Current VVIX.
            rv_iv_spread: Realized - implied vol spread.
            prev_vix: Previous day's VIX (for HMM update).

        Returns:
            (regime, confidence, detail_dict)
        """
        # Update HMM observations
        if prev_vix is not None and prev_vix > 0:
            vix_return = np.log(vix / prev_vix)
            self._recent_vix_returns.append(vix_return)
            if len(self._recent_vix_returns) > 252:
                self._recent_vix_returns = self._recent_vix_returns[-252:]

        # Rules classification
        rules_regime, rules_scores = classify_rules(
            vix, term_slope, vvix, rv_iv_spread, self.rules_config,
        )

        # HMM classification
        hmm_regime = "TRANSITIONAL"
        hmm_proba = np.array([0.33, 0.34, 0.33])

        if self.hmm.fitted and len(self._recent_vix_returns) >= 5:
            obs = np.array(self._recent_vix_returns[-60:])
            hmm_regime, hmm_proba = self.hmm.current_regime(obs)

        # Composite: weighted probability blend
        rules_proba = np.zeros(3)
        regime_idx = {"LOW_VOL_HARVESTING": 0, "TRANSITIONAL": 1, "CRISIS": 2}
        rules_proba[regime_idx[rules_regime]] = 1.0

        composite_proba = (
            self.hmm_weight * hmm_proba
            + self.rules_weight * rules_proba
        )

        # Override: if rules give extreme signal, use rules
        if rules_scores["crisis_score"] >= 4.0:
            final_regime = "CRISIS"
            confidence = 0.95
        elif rules_scores["low_vol_score"] >= 4.0:
            final_regime = "LOW_VOL_HARVESTING"
            confidence = 0.90
        else:
            final_idx = np.argmax(composite_proba)
            final_regime = REGIMES[final_idx]
            confidence = float(composite_proba[final_idx])

        detail = {
            "rules_regime": rules_regime,
            "rules_scores": rules_scores,
            "hmm_regime": hmm_regime,
            "hmm_proba": {REGIMES[i]: float(hmm_proba[i]) for i in range(3)},
            "composite_proba": {REGIMES[i]: float(composite_proba[i]) for i in range(3)},
            "vix": vix,
            "term_slope": term_slope,
        }

        return final_regime, confidence, detail
