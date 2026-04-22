import os
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_LEVEL_RANGES = {
    "l10": (0, 3049),
    "l11": (3049, 3049 + 9147),
    "l12": (3049 + 9147, 3049 + 9147 + 30490),
}

OUTPUT_COLUMNS = [
    "Levels",
    "Forecast_Methods",
    "Target_Service_Level",
    "Senario",
    "Achieved_Service_Level",
    "Holding_Costs",
    "Backlogging_Costs",
    "Fill_Rate",
]

DEFAULT_METRIC_TRIPLES = [
    ("backlog_90", "ch_90", "cb_90", "90%"),
    ("backlog_95", "ch_95", "cb_95", "95%"),
    ("backlog_99", "ch_99", "cb_99", "99%"),
]


def _validate_level_ranges(level_ranges: Mapping[str, Tuple[int, int]]) -> None:
    last_end = None
    for level_name, (start, end) in level_ranges.items():
        if end <= start:
            raise ValueError(f"Invalid range for {level_name}: {(start, end)}")
        if last_end is not None and start != last_end:
            raise ValueError(
                f"Level ranges must be contiguous and sorted. "
                f"Got previous end {last_end} but {level_name} starts at {start}."
            )
        last_end = end


def total_series_count(level_ranges: Mapping[str, Tuple[int, int]]) -> int:
    _validate_level_ranges(level_ranges)
    return list(level_ranges.values())[-1][1]


def expected_rows(level_ranges: Mapping[str, Tuple[int, int]], horizon: int) -> int:
    return total_series_count(level_ranges) * horizon


def _slice_level_block(df: pd.DataFrame, start_series: int, end_series: int, horizon: int) -> pd.DataFrame:
    start_row = start_series * horizon
    end_row = end_series * horizon
    block = df.iloc[start_row:end_row].reset_index(drop=True)
    expected = (end_series - start_series) * horizon
    if len(block) != expected:
        raise ValueError(
            f"Level block has {len(block)} rows, expected {expected}. "
            f"Check ordering, horizon, or level_ranges."
        )
    return block


def _reshape_series_metric(block: pd.DataFrame, col: str, n_series: int, horizon: int) -> np.ndarray:
    if col not in block.columns:
        raise KeyError(f"Column '{col}' not found in simulation output.")
    values = block[col].to_numpy()
    if len(values) != n_series * horizon:
        raise ValueError(f"Column '{col}' has {len(values)} values, expected {n_series * horizon}.")
    return values.reshape(n_series, horizon)


def summarize_level_metrics(
    block: pd.DataFrame,
    backlog_col: str,
    holding_col: str,
    backlog_cost_col: str,
    demand_col: str = "true_demand",
    horizon: int = 28,
    stockout_tol: float = 1e-12,
) -> Dict[str, float]:
    """
    Mean-based inventory summary.

    Final outputs are means, not totals:
    - Achieved_Service_Level: for each series, proportion of zero-backlog periods; then mean across series
    - Holding_Costs: mean period holding cost over all rows in the level block
    - Backlogging_Costs: mean period backlog cost over all rows in the level block
    - Fill_Rate: for each series, satisfied-demand ratio; then mean across series
    """
    n_series = len(block) // horizon
    backlog = _reshape_series_metric(block, backlog_col, n_series, horizon)
    holding = _reshape_series_metric(block, holding_col, n_series, horizon)
    backlogging_cost = _reshape_series_metric(block, backlog_cost_col, n_series, horizon)
    demand = _reshape_series_metric(block, demand_col, n_series, horizon)

    achieved_service_per_series = (backlog <= stockout_tol).mean(axis=1)

    lagged_backlog = np.concatenate([np.zeros((n_series, 1)), backlog[:, :-1]], axis=1)
    backlog_increase = np.maximum(backlog - lagged_backlog, 0.0)
    satisfied_demand = np.maximum(demand - backlog_increase, 0.0)
    total_demand = demand.sum(axis=1)
    fill_rate_per_series = np.divide(
        satisfied_demand.sum(axis=1),
        total_demand,
        out=np.ones_like(total_demand, dtype=float),
        where=total_demand > 0,
    )

    return {
        "Achieved_Service_Level": float(achieved_service_per_series.mean()),
        "Holding_Costs": float(holding.mean()),
        "Backlogging_Costs": float(backlogging_cost.mean()),
        "Fill_Rate": float(fill_rate_per_series.mean()),
    }


def summarize_inventory_dataframe(
    df: pd.DataFrame,
    level_ranges: Mapping[str, Tuple[int, int]] = DEFAULT_LEVEL_RANGES,
    model_name: str = "",
    scenario_name: str = "",
    metric_triples: Sequence[Tuple[str, str, str, str]] = DEFAULT_METRIC_TRIPLES,
    horizon: int = 28,
    level_labels: Optional[Sequence[int]] = None,
    demand_col: str = "true_demand",
) -> pd.DataFrame:
    _validate_level_ranges(level_ranges)
    expected = expected_rows(level_ranges, horizon)
    if len(df) != expected:
        raise ValueError(
            f"Input dataframe has {len(df)} rows, expected {expected}. "
            f"Filter to one model name before calling this function."
        )

    if level_labels is None:
        level_labels = [int(k[1:]) if k.startswith("l") and k[1:].isdigit() else k for k in level_ranges.keys()]
    if len(level_labels) != len(level_ranges):
        raise ValueError("level_labels length must match level_ranges length.")

    rows = []
    for (_level_key, (start, end)), level_label in zip(level_ranges.items(), level_labels):
        block = _slice_level_block(df, start, end, horizon)
        for backlog_col, holding_col, backlog_cost_col, target_service in metric_triples:
            metrics = summarize_level_metrics(
                block=block,
                backlog_col=backlog_col,
                holding_col=holding_col,
                backlog_cost_col=backlog_cost_col,
                demand_col=demand_col,
                horizon=horizon,
            )
            rows.append(
                {
                    "Levels": level_label,
                    "Forecast_Methods": model_name,
                    "Target_Service_Level": target_service,
                    "Senario": scenario_name,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)[OUTPUT_COLUMNS]


def _existing_path(path_candidates):
    for p in path_candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these files exists: {path_candidates}")


def load_pickle_columns(path_candidates, required_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if isinstance(path_candidates, str):
        path_candidates = [path_candidates]
    path = _existing_path(path_candidates)
    df = pd.read_pickle(path)
    if required_cols is not None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{path} is missing required columns: {missing}")
        df = df[list(required_cols)]
    return df


def summarize_scenario_set(
    full_df: pd.DataFrame,
    names: Sequence[str],
    name_map: Mapping[str, str],
    scenario_name: str,
    level_ranges: Mapping[str, Tuple[int, int]] = DEFAULT_LEVEL_RANGES,
    metric_triples: Sequence[Tuple[str, str, str, str]] = DEFAULT_METRIC_TRIPLES,
    horizon: int = 28,
    level_labels: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    outputs = []
    expected = expected_rows(level_ranges, horizon)
    present_names = set(full_df["name"].unique())
    for name in names:
        if name not in present_names:
            continue
        subset = full_df[full_df["name"] == name].reset_index(drop=True)
        if len(subset) != expected:
            raise ValueError(
                f"Model '{name}' has {len(subset)} rows, expected {expected}. "
                f"Check that the file belongs to one complete simulation run."
            )
        outputs.append(
            summarize_inventory_dataframe(
                df=subset,
                level_ranges=level_ranges,
                model_name=name_map.get(name, name),
                scenario_name=scenario_name,
                metric_triples=metric_triples,
                horizon=horizon,
                level_labels=level_labels,
            )
        )
    if not outputs:
        raise ValueError(f"No matching names found for scenario '{scenario_name}'.")
    return pd.concat(outputs, ignore_index=True)[OUTPUT_COLUMNS]


def collect_base_vs_ir_results(
    lead_time: int,
    level_ranges: Mapping[str, Tuple[int, int]] = DEFAULT_LEVEL_RANGES,
    horizon: int = 28,
    filename_prefix: str = "",
) -> pd.DataFrame:
    required_cols = [
        "name", "true_demand",
        "backlog_90", "ch_90", "cb_90",
        "backlog_95", "ch_95", "cb_95",
        "backlog_99", "ch_99", "cb_99",
    ]

    def _candidates(stem: str):
        cands = []
        if filename_prefix:
            cands.append(f"{filename_prefix}{stem}")
        cands.append(stem)
        # Backward compatibility with the user's earlier 721-prefixed files.
        if not stem.startswith("721"):
            cands.append(f"721{stem}")
        return cands

    lgb = load_pickle_columns(_candidates(f"lgbInvtSim_L{lead_time}.pkl"), required_cols)
    ets = load_pickle_columns(_candidates(f"etsInvtSim_L{lead_time}.pkl"), required_cols)
    base_file = pd.concat([lgb, ets], ignore_index=True)

    bu_file = load_pickle_columns(_candidates(f"BUOrder_L{lead_time}.pkl"), required_cols)
    td_file = load_pickle_columns(_candidates(f"TDFPOrder_L{lead_time}.pkl"), required_cols)
    mint_file = load_pickle_columns(_candidates(f"VarOrder_L{lead_time}.pkl"), required_cols)

    common_names = ["lgb_base", "lgb_bu", "lgb_td", "lgb_mint", "ets_base", "ets_bu", "ets_td", "ets_mint"]
    common_name_map = {
        "lgb_base": "LGBM",
        "lgb_bu": "LGBM_BU",
        "lgb_td": "LGBM_TD",
        "lgb_mint": "LGBM_MinT",
        "ets_base": "ETS",
        "ets_bu": "ETS_BU",
        "ets_td": "ETS_TD",
        "ets_mint": "ETS_MinT",
    }

    results = []
    results.append(
        summarize_scenario_set(
            full_df=base_file,
            names=common_names,
            name_map=common_name_map,
            scenario_name="BASE",
            level_ranges=level_ranges,
            horizon=horizon,
        )
    )
    results.append(
        summarize_scenario_set(
            full_df=bu_file,
            names=common_names,
            name_map=common_name_map,
            scenario_name="BU_OR",
            level_ranges=level_ranges,
            horizon=horizon,
        )
    )
    results.append(
        summarize_scenario_set(
            full_df=td_file,
            names=common_names,
            name_map=common_name_map,
            scenario_name="TDFP_OR",
            level_ranges=level_ranges,
            horizon=horizon,
        )
    )
    results.append(
        summarize_scenario_set(
            full_df=mint_file,
            names= common_names, #["lgb_base", "lgb_mint", "ets_base", "ets_mint"],
            name_map=common_name_map,
            scenario_name="WLS_VAR_OR",
            level_ranges=level_ranges,
            horizon=horizon,
        )
    )
    return pd.concat(results, ignore_index=True)[OUTPUT_COLUMNS]


def collect_mean_inventory_results(*args, **kwargs) -> pd.DataFrame:
    """Alias kept for readability: this collector returns mean-based summaries."""
    return collect_base_vs_ir_results(*args, **kwargs)
