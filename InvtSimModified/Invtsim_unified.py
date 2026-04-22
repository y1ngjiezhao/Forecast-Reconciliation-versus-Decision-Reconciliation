import warnings
from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.simplefilter("ignore")


class InvtSim:
    """
    Inventory simulation under an order-up-to policy.

    Timing convention (lead time L):
    - order placed in period t arrives in period t+L
    - therefore, when L=1, arrival_t = order_{t-1}

    Forecast-based initialization (strictly causal):
    - initial inventory position is the lead-time demand forecast only
    - safety stock enters the order-up-to target, but NOT the initial
      inventory state used for replay
    - therefore, service-level differences can affect generated orders
      instead of being canceled by identical initial safety-stock offsets

    Notes
    -----
    1. ob_all_t():
       - computes safety stock from residuals
       - computes orders from forecasts via the order-up-to rule
       - IMPORTANT: because an order placed at t arrives at t+L, the target
         inventory position at time t must cover future demand starting at
         t+1, i.e. forecast[t+1 : t+1+L]
       - replays those orders through the stock-flow system

    2. ob_all_t_fixedcase():
       - takes fixed orders as given
       - does NOT recompute orders from safety stock
       - uses the forecast sequence for causal initialization
       - preserves sst_* columns for reporting, but replay starts from the
         same forecast-only initial inventory position as ob_all_t()
    """

    def __init__(
        self,
        fcst,
        truth,
        name: str,
        residual=None,
        L: int = 1,
        period: Optional[int] = 28,
        h: float = 1,
        b: Tuple[float, float, float] = (9, 19, 99),
    ):
        self.name = name
        self.h = h
        self.b90, self.b95, self.b99 = b

        self.fcst = np.asarray(fcst)
        self.truth = np.asarray(truth, dtype=float)
        self.residual = None if residual is None else np.asarray(residual, dtype=float)
        self.L = int(L)
        self.period = int(period) if period is not None else len(self.truth)

        if self.L < 1:
            raise ValueError("L must be >= 1.")
        if self.period < 1:
            raise ValueError("period must be >= 1.")
        if len(self.truth) < self.period:
            raise ValueError("truth is shorter than period.")

        self.a_90, self.a_95, self.a_99 = 0.9, 0.95, 0.99
        self._reset_logs()

    def _reset_logs(self) -> None:
        self.name_l = []

        self.sst_90l, self.sst_95l, self.sst_99l = [], [], []
        self.arrival_90l, self.arrival_95l, self.arrival_99l = [], [], []
        self.wip_90l, self.wip_95l, self.wip_99l = [], [], []
        self.ot_90l, self.ot_95l, self.ot_99l = [], [], []
        self.ipt_90l, self.ipt_95l, self.ipt_99l = [], [], []
        self.net_90l, self.net_95l, self.net_99l = [], [], []
        self.ch_90l, self.ch_95l, self.ch_99l = [], [], []
        self.cb_90l, self.cb_95l, self.cb_99l = [], [], []
        self.bkl_90l, self.bkl_95l, self.bkl_99l = [], [], []

    def ob_ss_t(self):
        if self.residual is None or self.residual.size < 2:
            raise ValueError("residual must contain at least two observations for safety-stock estimation.")
        se = np.std(self.residual, ddof=1)
        ss_90 = norm.ppf(self.a_90) * se * np.sqrt(self.L)
        ss_95 = norm.ppf(self.a_95) * se * np.sqrt(self.L)
        ss_99 = norm.ppf(self.a_99) * se * np.sqrt(self.L)
        return (float(ss_90), float(ss_95), float(ss_99))

    @staticmethod
    def _safe_forecast_output(fcst, period: int) -> np.ndarray:
        arr = np.asarray(fcst, dtype=float)
        if arr.ndim == 1 and len(arr) >= period:
            return arr[:period].astype(float)
        raise ValueError("A forecast sequence with length >= period is required.")

    def _lead_time_demand(self, forecast_seq: np.ndarray, t: int) -> float:
        return float(np.sum(forecast_seq[t : t + self.L]))

    def _initial_inventory_position(self, forecast_seq: np.ndarray) -> float:
        return float(np.sum(forecast_seq[: self.L]))

    @staticmethod
    def _arrival_from_orders(order_history: Sequence[float], t: int, L: int) -> float:
        idx = t - L
        return float(order_history[idx]) if idx >= 0 else 0.0

    def _build_output(self, forecast_output: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "name": self.name_l,
                "true_demand": self.truth[: self.period],
                "forecasts": forecast_output,
                "sst_90": self.sst_90l,
                "arrival_90": self.arrival_90l,
                "ot_90": self.ot_90l,
                "wip_90": self.wip_90l,
                "ip_90": self.ipt_90l,
                "net_90": self.net_90l,
                "backlog_90": self.bkl_90l,
                "ch_90": self.ch_90l,
                "cb_90": self.cb_90l,
                "sst_95": self.sst_95l,
                "arrival_95": self.arrival_95l,
                "ot_95": self.ot_95l,
                "wip_95": self.wip_95l,
                "ip_95": self.ipt_95l,
                "net_95": self.net_95l,
                "backlog_95": self.bkl_95l,
                "ch_95": self.ch_95l,
                "cb_95": self.cb_95l,
                "sst_99": self.sst_99l,
                "arrival_99": self.arrival_99l,
                "ot_99": self.ot_99l,
                "wip_99": self.wip_99l,
                "ip_99": self.ipt_99l,
                "net_99": self.net_99l,
                "backlog_99": self.bkl_99l,
                "ch_99": self.ch_99l,
                "cb_99": self.cb_99l,
            }
        )

    def _simulate_from_orders(
        self,
        order_source_90: Sequence[float],
        order_source_95: Sequence[float],
        order_source_99: Sequence[float],
        ss_values: Tuple[float, float, float],
        forecast_output: np.ndarray,
        initial_ss_values: Optional[Tuple[float, float, float]] = None,
    ) -> pd.DataFrame:
        self._reset_logs()
        self.ss90, self.ss95, self.ss99 = ss_values

        if initial_ss_values is None:
            initial_ss_values = (0.0, 0.0, 0.0)
        init_ss90, init_ss95, init_ss99 = initial_ss_values

        init_ip = self._initial_inventory_position(forecast_output)
        self.ip_90t = init_ip + init_ss90
        self.ip_95t = init_ip + init_ss95
        self.ip_99t = init_ip + init_ss99

        self.net_90t, self.net_95t, self.net_99t = self.ip_90t, self.ip_95t, self.ip_99t
        self.wip_90t, self.wip_95t, self.wip_99t = 0.0, 0.0, 0.0
        self.bkl_90t, self.bkl_95t, self.bkl_99t = 0.0, 0.0, 0.0

        for t in range(self.period):
            self.name_l.append(self.name)

            self.o_90t = float(order_source_90[t])
            self.o_95t = float(order_source_95[t])
            self.o_99t = float(order_source_99[t])

            self.arrival_90t = self._arrival_from_orders(self.ot_90l, t, self.L)
            self.arrival_95t = self._arrival_from_orders(self.ot_95l, t, self.L)
            self.arrival_99t = self._arrival_from_orders(self.ot_99l, t, self.L)

            self.net_90t = self.net_90t + self.arrival_90t - self.truth[t]
            self.net_95t = self.net_95t + self.arrival_95t - self.truth[t]
            self.net_99t = self.net_99t + self.arrival_99t - self.truth[t]

            self.bkl_90t = max(0.0, -self.net_90t)
            self.bkl_95t = max(0.0, -self.net_95t)
            self.bkl_99t = max(0.0, -self.net_99t)

            self.wip_90t = self.wip_90t + self.o_90t - self.arrival_90t
            self.wip_95t = self.wip_95t + self.o_95t - self.arrival_95t
            self.wip_99t = self.wip_99t + self.o_99t - self.arrival_99t

            self.ip_90t = self.net_90t + self.wip_90t
            self.ip_95t = self.net_95t + self.wip_95t
            self.ip_99t = self.net_99t + self.wip_99t

            self.ch_90t = self.h * max(0.0, self.net_90t)
            self.ch_95t = self.h * max(0.0, self.net_95t)
            self.ch_99t = self.h * max(0.0, self.net_99t)
            self.cb_90t = self.b90 * self.bkl_90t
            self.cb_95t = self.b95 * self.bkl_95t
            self.cb_99t = self.b99 * self.bkl_99t

            self.sst_90l.append(self.ss90)
            self.sst_95l.append(self.ss95)
            self.sst_99l.append(self.ss99)
            self.arrival_90l.append(self.arrival_90t)
            self.arrival_95l.append(self.arrival_95t)
            self.arrival_99l.append(self.arrival_99t)
            self.ot_90l.append(self.o_90t)
            self.ot_95l.append(self.o_95t)
            self.ot_99l.append(self.o_99t)
            self.wip_90l.append(self.wip_90t)
            self.wip_95l.append(self.wip_95t)
            self.wip_99l.append(self.wip_99t)
            self.ipt_90l.append(self.ip_90t)
            self.ipt_95l.append(self.ip_95t)
            self.ipt_99l.append(self.ip_99t)
            self.net_90l.append(self.net_90t)
            self.net_95l.append(self.net_95t)
            self.net_99l.append(self.net_99t)
            self.bkl_90l.append(self.bkl_90t)
            self.bkl_95l.append(self.bkl_95t)
            self.bkl_99l.append(self.bkl_99t)
            self.ch_90l.append(self.ch_90t)
            self.ch_95l.append(self.ch_95t)
            self.ch_99l.append(self.ch_99t)
            self.cb_90l.append(self.cb_90t)
            self.cb_95l.append(self.cb_95t)
            self.cb_99l.append(self.cb_99t)

        return self._build_output(forecast_output)

    def ob_all_t(self):
        forecast_output = self._safe_forecast_output(self.fcst, self.period)
        self.ss90, self.ss95, self.ss99 = self.ob_ss_t()

        orders_90, orders_95, orders_99 = [], [], []
        base_ip = self._initial_inventory_position(forecast_output)

        # Mirror the same state timing used in _simulate_from_orders:
        # start from forecast-only initial inventory, then for each period
        # process arrival and demand first, and finally place a new order for
        # future demand. Safety stock affects the target, not the initial state.
        net90 = base_ip
        net95 = base_ip
        net99 = base_ip
        wip90 = wip95 = wip99 = 0.0

        for t in range(self.period):
            arr90 = self._arrival_from_orders(orders_90, t, self.L)
            arr95 = self._arrival_from_orders(orders_95, t, self.L)
            arr99 = self._arrival_from_orders(orders_99, t, self.L)

            net90 = net90 + arr90 - self.truth[t]
            net95 = net95 + arr95 - self.truth[t]
            net99 = net99 + arr99 - self.truth[t]

            wip90 = wip90 - arr90
            wip95 = wip95 - arr95
            wip99 = wip99 - arr99

            ip90_preorder = net90 + wip90
            ip95_preorder = net95 + wip95
            ip99_preorder = net99 + wip99

            # Order placed at t arrives at t+L, so the target must cover
            # future demand starting at t+1, not the current period t.
            dtl = self._lead_time_demand(forecast_output, t + 1)
            o90 = max(0.0, dtl + self.ss90 - ip90_preorder)
            o95 = max(0.0, dtl + self.ss95 - ip95_preorder)
            o99 = max(0.0, dtl + self.ss99 - ip99_preorder)

            orders_90.append(o90)
            orders_95.append(o95)
            orders_99.append(o99)

            wip90 = wip90 + o90
            wip95 = wip95 + o95
            wip99 = wip99 + o99

        return self._simulate_from_orders(
            order_source_90=orders_90,
            order_source_95=orders_95,
            order_source_99=orders_99,
            ss_values=(self.ss90, self.ss95, self.ss99),
            forecast_output=forecast_output,
            initial_ss_values=(0.0, 0.0, 0.0),
        )

    def ob_all_t_fixedcase(self, fixed_order: pd.DataFrame):
        fixed_order = pd.DataFrame(fixed_order).reset_index(drop=True)
        if len(fixed_order) < self.period:
            raise ValueError("fixed_order must have at least 'period' rows.")

        # Forecast sequence for strict forecast-based initialization.
        if "forecasts" in fixed_order.columns:
            forecast_output = self._safe_forecast_output(fixed_order["forecasts"].to_numpy(dtype=float), self.period)
        else:
            forecast_output = self._safe_forecast_output(self.fcst, self.period)

        # Preserve sst labels for reporting if present. Replay still starts
        # from the forecast-only initial inventory position, with no initial
        # safety-stock offset.
        if {"sst_90", "sst_95", "sst_99"}.issubset(fixed_order.columns):
            ss_values = (
                float(fixed_order["sst_90"].iloc[0]),
                float(fixed_order["sst_95"].iloc[0]),
                float(fixed_order["sst_99"].iloc[0]),
            )
        else:
            ss_values = (0.0, 0.0, 0.0)

        if {"ot_90", "ot_95", "ot_99"}.issubset(fixed_order.columns):
            a = fixed_order["ot_90"].to_numpy(dtype=float)[: self.period]
            b = fixed_order["ot_95"].to_numpy(dtype=float)[: self.period]
            c = fixed_order["ot_99"].to_numpy(dtype=float)[: self.period]
        else:
            if fixed_order.shape[1] < 3:
                raise ValueError("fixed_order must contain at least three order columns (90/95/99).")
            a = np.asarray(fixed_order.iloc[: self.period, 0], dtype=float)
            b = np.asarray(fixed_order.iloc[: self.period, 1], dtype=float)
            c = np.asarray(fixed_order.iloc[: self.period, 2], dtype=float)

        return self._simulate_from_orders(
            order_source_90=a,
            order_source_95=b,
            order_source_99=c,
            ss_values=ss_values,
            forecast_output=forecast_output,
            initial_ss_values=(0.0, 0.0, 0.0),
        )
