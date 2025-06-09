# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from power_grid_model import (  # ValidationException,  # Removed: not available in installed version; BatchResult,          # Removed: not available in installed version
    BatchUpdateDataset,
    batch_power_flow,
    power_flow,
)
from power_grid_model.io.serialization import deserialize_json
from power_grid_model.utils import get_element_ids_from_grid_data


# Define dummy classes to replace missing ones
class ValidationException(Exception):
    pass


class BatchResult:
    pass


class PowerGridCalculator:
    def __init__(self, pgm_input_data: dict):
        self._pgm_grid = None
        self._initialize_pgm(pgm_input_data)
        self._base_power_mva = pgm_input_data.get("base_power_mva", 1.0)
        if "base_power" in pgm_input_data:
            self._base_power_mva = pgm_input_data["base_power"] / 1e6
        print(f"Initialized PowerGridCalculator with base power: {self._base_power_mva} MVA")

    def _initialize_pgm(self, pgm_input_data: dict):
        try:
            self._pgm_grid = deserialize_json(pgm_input_data)
        except ValidationException as e:
            print(f"Error initializing PGM grid: {e}")
            raise

    def create_batch_update_dataset(
        self, active_load_profile: pd.DataFrame, reactive_load_profile: pd.DataFrame
    ) -> BatchUpdateDataset:
        if "timestamp" in active_load_profile.columns:
            active_load_profile["timestamp"] = pd.to_datetime(active_load_profile["timestamp"])
            active_load_profile = active_load_profile.set_index("timestamp")
        if "timestamp" in reactive_load_profile.columns:
            reactive_load_profile["timestamp"] = pd.to_datetime(reactive_load_profile["timestamp"])
            reactive_load_profile = reactive_load_profile.set_index("timestamp")

        if not active_load_profile.index.equals(reactive_load_profile.index):
            raise ValueError("Active and reactive load profiles do not have matching timestamps.")

        p_profile_pivot = active_load_profile.pivot_table(
            index=active_load_profile.index, columns="load_id", values="value"
        )
        q_profile_pivot = reactive_load_profile.pivot_table(
            index=reactive_load_profile.index, columns="load_id", values="value"
        )

        if not p_profile_pivot.columns.equals(q_profile_pivot.columns):
            raise ValueError("Active and reactive load profiles do not have matching load IDs.")

        load_ids = p_profile_pivot.columns.tolist()
        timestamps = p_profile_pivot.index.tolist()
        updates = []
        for ts in timestamps:
            timestamp_str = ts.isoformat()
            sym_load_updates = []
            for load_id in load_ids:
                p_value = p_profile_pivot.loc[ts, load_id]
                q_value = q_profile_pivot.loc[ts, load_id]
                sym_load_updates.append({"id": load_id, "p_mw": p_value, "q_mvar": q_value})
            updates.append({"timestamp": timestamp_str, "sym_load": sym_load_updates})

        batch_dataset = BatchUpdateDataset(updates)
        return batch_dataset

    def run_time_series_power_flow(self, batch_dataset: BatchUpdateDataset) -> BatchResult:
        if self._pgm_grid is None:
            raise RuntimeError("PGM grid not initialized. Call __init__ with valid PGM data first.")
        try:
            results = batch_power_flow(self._pgm_grid, batch_dataset)
            return results
        except ValidationException as e:
            print(f"Error running batch power flow: {e}")
            raise

    def aggregate_node_voltage_results(self, power_flow_results: BatchResult) -> pd.DataFrame:
        node_voltage_magnitudes = power_flow_results.get_data("node", "u_pu")
        summary_data = []
        for timestamp_str, node_data in node_voltage_magnitudes.items():
            u_pu_values = np.array(list(node_data.values()))
            node_ids = list(node_data.keys())
            if len(u_pu_values) == 0:
                continue
            max_pu_voltage = np.max(u_pu_values)
            min_pu_voltage = np.min(u_pu_values)
            max_voltage_node_id = node_ids[np.argmax(u_pu_values)]
            min_voltage_node_id = node_ids[np.argmin(u_pu_values)]
            summary_data.append(
                {
                    "timestamp": pd.to_datetime(timestamp_str),
                    "max_pu_voltage": max_pu_voltage,
                    "max_voltage_node_id": max_voltage_node_id,
                    "min_pu_voltage": min_pu_voltage,
                    "min_voltage_node_id": min_voltage_node_id,
                }
            )
        summary_df = pd.DataFrame(summary_data).set_index("timestamp")
        return summary_df

    def aggregate_line_loss_and_loading_results(self, power_flow_results: BatchResult) -> pd.DataFrame:
        line_losses_pu = power_flow_results.get_data("line", "p_loss_pu")
        line_loadings_pu = power_flow_results.get_data("line", "loading_pu")
        all_line_ids = set()
        for ts_data in line_losses_pu.values():
            all_line_ids.update(ts_data.keys())
        timestamps = sorted([pd.to_datetime(ts) for ts in line_losses_pu.keys()])
        if not timestamps:
            return pd.DataFrame(
                columns=[
                    "energy_loss_kwh",
                    "max_loading_pu",
                    "max_loading_timestamp",
                    "min_loading_pu",
                    "min_loading_timestamp",
                ]
            )
        time_intervals_hours = [
            (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600 for i in range(1, len(timestamps))
        ]
        summary_data = []
        for line_id in sorted(list(all_line_ids)):
            losses_for_line = []
            loadings_for_line = []
            loading_timestamps = []
            for ts_str in sorted(line_losses_pu.keys()):
                ts = pd.to_datetime(ts_str)
                loss_pu = line_losses_pu.get(ts_str, {}).get(line_id, 0.0)
                loading_pu = line_loadings_pu.get(ts_str, {}).get(line_id, 0.0)
                losses_for_line.append(loss_pu)
                loadings_for_line.append(loading_pu)
                loading_timestamps.append(ts)
            losses_kw = [loss_pu * self._base_power_mva * 1000 for loss_pu in losses_for_line]
            energy_loss_kwh = trapezoidal_integral(losses_kw, loading_timestamps)
            if loadings_for_line:
                max_loading_pu = max(loadings_for_line)
                min_loading_pu = min(loadings_for_line)
                max_loading_timestamp = loading_timestamps[np.argmax(loadings_for_line)]
                min_loading_timestamp = loading_timestamps[np.argmin(loadings_for_line)]
            else:
                max_loading_pu = np.nan
                min_loading_pu = np.nan
                max_loading_timestamp = pd.NaT
                min_loading_timestamp = pd.NaT
            summary_data.append(
                {
                    "line_id": line_id,
                    "energy_loss_kwh": energy_loss_kwh,
                    "max_loading_pu": max_loading_pu,
                    "max_loading_timestamp": max_loading_timestamp,
                    "min_loading_pu": min_loading_pu,
                    "min_loading_timestamp": min_loading_timestamp,
                }
            )
        summary_df = pd.DataFrame(summary_data).set_index("line_id")
        return summary_df


def trapezoidal_integral(y_values: list[float], x_values: list[pd.Timestamp]) -> float:
    if len(y_values) < 2 or len(x_values) < 2:
        return 0.0
    integral = 0.0
    for i in range(len(y_values) - 1):
        delta_x_hours = (x_values[i + 1] - x_values[i]).total_seconds() / 3600.0
        sum_y = y_values[i] + y_values[i + 1]
        integral += 0.5 * sum_y * delta_x_hours
    return integral
