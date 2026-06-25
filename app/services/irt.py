"""Psychometric IRT core (Phase 2) — pure math, no DB.

Everything here follows the Vettalume IRT reference exactly:
  * 3PL:        P(theta) = c + (1-c) / (1 + e^(-a(theta-b)))
  * ability:    EAP — the mean of the posterior belief curve over a theta grid, with a N(0,1) prior
  * uncertainty SE = 1/sqrt(total Fisher information at the estimate)
  * selection:  pick the item with the most information at the current ability
  * Elo/1PL:    theta_new = theta_old + K*(R - P),  P = sigmoid(theta - b)
  * calibration: bootstrap via logit, then Newton-Raphson MLE on the item curve (phased b -> a -> c)

The worked examples in the reference are the unit tests for this module (see tests/test_irt.py).
"""
from __future__ import annotations

import math

# ability grid: -3..+3 in 0.5 steps (the reference grid)
THETA_MIN, THETA_MAX, THETA_STEP = -3.0, 3.0, 0.5
THETA_GRID = [round(THETA_MIN + i * THETA_STEP, 4)
              for i in range(int((THETA_MAX - THETA_MIN) / THETA_STEP) + 1)]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def sigmoid(x: float) -> float:
    # numerically stable logistic
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normal_prior(theta: float) -> float:
    """Standard Normal density N(0,1) — the starting belief that a new learner is average."""
    return _INV_SQRT_2PI * math.exp(-0.5 * theta * theta)


def prob_3pl(theta: float, a: float, b: float, c: float) -> float:
    """3PL probability of a correct answer."""
    return c + (1.0 - c) * sigmoid(a * (theta - b))


def information_3pl(theta: float, a: float, b: float, c: float) -> float:
    """Fisher information a single item carries at this ability: I = a^2 * (1-P)/P * [(P-c)/(1-c)]^2."""
    p = prob_3pl(theta, a, b, c)
    if p <= 0.0 or p >= 1.0 or c >= 1.0:
        return 0.0
    return (a * a) * ((1.0 - p) / p) * (((p - c) / (1.0 - c)) ** 2)


def se_from_info(total_info: float) -> float:
    """SE = 1/sqrt(information). Infinite when nothing is known yet."""
    return math.inf if total_info <= 0.0 else 1.0 / math.sqrt(total_info)


# ---------- EAP ability ----------
def eap_ability(items: list[tuple[float, float, float, int]],
                grid: list[float] | None = None) -> tuple[float, float]:
    """Expected A Posteriori ability over the theta grid.

    items: list of (a, b, c, u) where u is 1 for a correct response, 0 for wrong.
    Returns (theta_hat, se). With no items, returns (0.0, inf) — the prior mean with full uncertainty.
    """
    grid = grid or THETA_GRID
    num = den = 0.0
    for theta in grid:
        w = normal_prior(theta)
        for a, b, c, u in items:
            p = prob_3pl(theta, a, b, c)
            w *= p if u else (1.0 - p)
        num += theta * w
        den += w
    if den <= 0.0:
        return 0.0, math.inf
    theta_hat = num / den
    total_info = sum(information_3pl(theta_hat, a, b, c) for a, b, c, _ in items)
    return theta_hat, se_from_info(total_info)


# ---------- Elo / 1PL running update ----------
def elo_update(theta: float, b: float, correct: bool, k: float = 0.5) -> float:
    """1PL step update (the chess-rating move): theta += K*(R - P), P = sigmoid(theta - b)."""
    p = sigmoid(theta - b)
    return theta + k * ((1.0 if correct else 0.0) - p)


# ---------- bootstrap ----------
def logit(p: float, eps: float = 1e-3) -> float:
    """ln(p/(1-p)), clamped away from 0/1 so all-right / all-wrong rows stay finite."""
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


# ---------- MLE item calibration (the shared Newton engine) ----------
A_MIN, A_MAX = 0.2, 3.0
B_MIN, B_MAX = -4.0, 4.0
C_MIN, C_MAX = 0.0, 0.5


def _derivs(theta: float, a: float, b: float, c: float) -> tuple[float, float, float, float, float]:
    """Returns (P, dP/da, dP/db, dP/dc, L) for the 3PL at one ability."""
    L = sigmoid(a * (theta - b))
    p = c + (1.0 - c) * L
    g = (1.0 - c) * L * (1.0 - L)            # (1-c)*L*(1-L)
    dP_da = g * (theta - b)
    dP_db = -a * g
    dP_dc = 1.0 - L
    return p, dP_da, dP_db, dP_dc, L


def fit_item(thetas: list[float], us: list[int], *, phase: str,
             a0: float = 1.0, b0: float = 0.0, c0: float = 0.2,
             iters: int = 40, tol: float = 1e-4, max_step: float = 0.5,
             prior_a_var: float = 0.5, prior_b_var: float = 16.0, prior_c_var: float = 0.005) -> dict:
    """Regularized (MAP) Newton-Raphson for one item's curve against fixed abilities.

    phase decides which parameters are free (the reference's phased plan):
      'b'   -> fit b only         (a, c fixed)            ~40+ responses
      '2pl' -> fit b, a           (c fixed at prior)      ~500+ responses
      '3pl' -> fit b, a, c                                 ~2000+ responses

    A Gaussian prior is added to each free parameter (MAP, not raw MLE) so the fit stays stable where
    data is thin. The c-prior is deliberately STRONG: c is barely estimable (the reference shows it is
    noise even at 2000 responses), so the 3PL phase only nudges c off the format floor — freely
    fitting it diverges and corrupts a and b. Each Newton step is also clamped to +/-max_step to stop
    overshoot. Priors centre on a=1, b=0, c=c0.

    Per free parameter xi:
      grad = sum (u - P)/(P(1-P)) * dP/dxi   - (xi - mu)/var
      info = sum (dP/dxi)^2 / (P(1-P))        + 1/var
      xi  += clamp(grad / info, +/-max_step)
    """
    a, b, c = a0, b0, c0
    free = {"b"} if phase == "b" else {"a", "b"} if phase == "2pl" else {"a", "b", "c"}
    mu = {"a": 1.0, "b": 0.0, "c": c0}
    var = {"a": prior_a_var, "b": prior_b_var, "c": prior_c_var}
    lo = {"a": A_MIN, "b": B_MIN, "c": C_MIN}
    hi = {"a": A_MAX, "b": B_MAX, "c": C_MAX}
    n = len(us)
    converged = False
    used = 0
    for used in range(1, iters + 1):
        g = {"a": 0.0, "b": 0.0, "c": 0.0}
        h = {"a": 0.0, "b": 0.0, "c": 0.0}
        for theta, u in zip(thetas, us):
            p, dPa, dPb, dPc, _ = _derivs(theta, a, b, c)
            denom = p * (1.0 - p)
            if denom < 1e-9:
                continue
            resid = (u - p) / denom
            if "a" in free:
                g["a"] += resid * dPa; h["a"] += (dPa * dPa) / denom
            if "b" in free:
                g["b"] += resid * dPb; h["b"] += (dPb * dPb) / denom
            if "c" in free:
                g["c"] += resid * dPc; h["c"] += (dPc * dPc) / denom
        cur = {"a": a, "b": b, "c": c}
        step = 0.0
        new = dict(cur)
        for k in free:
            g[k] += -(cur[k] - mu[k]) / var[k]      # MAP prior gradient
            h[k] += 1.0 / var[k]                     # MAP prior information
            if h[k] > 1e-9:
                d = g[k] / h[k]
                d = max(-max_step, min(max_step, d))   # clamp the step
                new[k] = min(hi[k], max(lo[k], cur[k] + d))
                step = max(step, abs(new[k] - cur[k]))
        a, b, c = new["a"], new["b"], new["c"]
        if step < tol:
            converged = True
            break
    return {"a": a, "b": b, "c": c, "phase": phase, "n": n,
            "converged": converged, "iters": used}


def default_c(num_options: int | None, is_type_in: bool = False) -> float:
    """Guessing floor from the answer format: 1/options for MCQ, ~0 for type-in."""
    if is_type_in or not num_options or num_options <= 0:
        return 0.0
    return round(1.0 / num_options, 4)


def phase_for(n_responses: int, *, two_pl_at: int = 500, three_pl_at: int = 2000) -> str:
    """Which parameters the data can support at this response count."""
    if n_responses >= three_pl_at:
        return "3pl"
    if n_responses >= two_pl_at:
        return "2pl"
    return "b"


# ---------- normal CDF / inverse (percentile + call-probability scoring) ----------
def normal_cdf(x: float) -> float:
    """Standard Normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse standard Normal CDF (Acklam's rational approximation)."""
    p = min(1 - 1e-12, max(1e-12, p))
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def marginal_reliability(se: float, prior_var: float = 1.0) -> float:
    """Reliability of a theta estimate on the N(0,1) scale: rho = 1 - SE^2/Var, clamped to [0,1)."""
    if se == math.inf:
        return 0.0
    return max(0.0, min(0.999, 1.0 - (se * se) / prior_var))
