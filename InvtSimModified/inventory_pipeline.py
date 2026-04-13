import numpy as np
import pandas as pd
from tqdm import tqdm

from Invtsim_unified import InvtSim


def save_pickle(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame as pickle."""
    df.to_pickle(path)


def _validate_window_lengths(truth, gap1: int, label: str) -> int:
    total = len(truth)
    if gap1 < 1:
        raise ValueError("gap1 must be >= 1.")
    if total % gap1 != 0:
        raise ValueError(
            f"{label} length {total} is not divisible by gap1={gap1}. "
            "The simulation expects complete horizon windows."
        )
    return total // gap1


def run_base_loop(fcst, truth, residual, NAME: str, gap1: int = 28, gap2: int = 1913, L_: int = 1):
    """
    Direct forecast-based simulation.

    Each truth window of length `gap1` is paired with:
    - the corresponding forecast window of length `gap1`
    - the corresponding residual history window of length `gap2`
    """
    n = _validate_window_lengths(truth, gap1, "truth")
    fcst = np.asarray(fcst, dtype=float)
    truth = np.asarray(truth, dtype=float)
    residual = np.asarray(residual, dtype=float)

    if len(fcst) < len(truth):
        raise ValueError("fcst must be at least as long as truth.")
    if len(residual) < n * gap2:
        raise ValueError(
            f"residual length {len(residual)} is shorter than the required {n * gap2} "
            f"for n={n} windows and gap2={gap2}."
        )

    out = []
    for i in tqdm(range(n)):
        fcst_1 = fcst[i * gap1:(i + 1) * gap1]
        truth_1 = truth[i * gap1:(i + 1) * gap1]
        res_1 = residual[i * gap2:(i + 1) * gap2]
        sim = InvtSim(fcst=fcst_1, truth=truth_1, residual=res_1, name=NAME, L=L_, period=gap1)
        out.append(sim.ob_all_t().reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def run_fixed_loop(truth, forecast_source, fixed_orders, NAME: str, gap1: int = 28, L_: int = 1):
    """
    Fixed-order replay simulation.

    Orders are taken as externally supplied (e.g., BU / TDFP / MinT order reconciliation),
    while the forecast path is still used for the strictly causal initial inventory position.
    """
    n = _validate_window_lengths(truth, gap1, "truth")
    truth = np.asarray(truth, dtype=float)
    forecast_source = np.asarray(forecast_source, dtype=float)
    fixed_orders = pd.DataFrame(fixed_orders).reset_index(drop=True)

    if len(forecast_source) < len(truth):
        raise ValueError("forecast_source must be at least as long as truth.")
    if len(fixed_orders) < len(truth):
        raise ValueError("fixed_orders must have at least as many rows as truth.")

    out = []
    for i in tqdm(range(n)):
        fo = forecast_source[i * gap1:(i + 1) * gap1]
        tr = truth[i * gap1:(i + 1) * gap1]
        od = fixed_orders.iloc[i * gap1:(i + 1) * gap1].reset_index(drop=True)
        sim = InvtSim(fcst=fo, truth=tr, residual=np.zeros(max(2, gap1)), name=NAME, L=L_, period=gap1)
        out.append(sim.ob_all_t_fixedcase(od).reset_index(drop=True))
    return pd.concat(out, ignore_index=True)
