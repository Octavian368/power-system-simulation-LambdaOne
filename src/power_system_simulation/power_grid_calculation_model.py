import pandas as pd
import numpy as np
import json
from typing import Dict, List
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from power_grid_model import (
    PowerGridModel,
    CalculationMethod,
    CalculationType,
    initialize_array,
    DatasetType,
    ComponentType
)
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import (
    validate_input_data,
    validate_batch_data,
    errors_to_string
)


def _convert_to_columnar_format(data: dict) -> dict:
    columnar_data = {}
    int32_fields = {"id", "from_node", "to_node", "node"}
    int8_fields = {"status", "from_status", "to_status", "type"}

    for component, entries in data.items():
        if not entries:
            columnar_data[component] = {}
            continue

        field_map = defaultdict(list)
        for entry in entries:
            for k, v in entry.items():
                field_map[k].append(v)

        converted = {}
        for key, value in field_map.items():
            arr = np.array(value)
            if key in int32_fields:
                arr = arr.astype(np.int32)
            elif key in int8_fields:
                arr = arr.astype(np.int8)
            elif arr.dtype.kind in {"i", "f"}:
                arr = arr.astype(np.float64)
            converted[key] = arr

        columnar_data[component] = converted

    return columnar_data


class PowerGridCalculator:
    def __init__(self, pgm_input_data: Dict):
        row_data = pgm_input_data.get("data", pgm_input_data)
        columnar_data = _convert_to_columnar_format(row_data)

        errors = validate_input_data(
            input_data=columnar_data,
            calculation_type=CalculationType.power_flow,
            symmetric=True
        )
        if errors:
            raise ValueError(errors_to_string(errors, "input_data", details=True))

        self.model = PowerGridModel(columnar_data)
        self.input_data = columnar_data

    def create_batch_update_dataset(self, active_df: pd.DataFrame, reactive_df: pd.DataFrame) -> Dict:
        # Fix for when Timestamp and load_id are in index or columns with wrong capitalization
        active_df = active_df.rename(columns=str.lower).reset_index()
        reactive_df = reactive_df.rename(columns=str.lower).reset_index()

        if 'timestamp' not in active_df.columns or 'load_id' not in active_df.columns:
            raise ValueError(f"Expected 'timestamp' and 'load_id' in active_df, got columns: {list(active_df.columns)}")
        if 'timestamp' not in reactive_df.columns or 'load_id' not in reactive_df.columns:
            raise ValueError(f"Expected 'timestamp' and 'load_id' in reactive_df, got columns: {list(reactive_df.columns)}")

        timestamps = active_df['timestamp'].unique()
        load_ids = active_df['load_id'].unique()


        update = initialize_array(
            DatasetType.update, ComponentType.sym_load, (len(timestamps), len(load_ids))
        )
        update['id'] = load_ids

        for i, t in enumerate(timestamps):
            active_vals = active_df[active_df['timestamp'] == t].sort_values('load_id')['value'].values
            reactive_vals = reactive_df[reactive_df['timestamp'] == t].sort_values('load_id')['value'].values
            update['p_specified'][i] = active_vals
            update['q_specified'][i] = reactive_vals

        return {ComponentType.sym_load: update}

    def run_time_series_power_flow(self, update_data: Dict, symmetric=True,
                                   calculation_method=CalculationMethod.newton_raphson,
                                   error_tolerance=1e-8, max_iterations=20) -> Dict:
        try:
            return self.model.calculate_power_flow(
                update_data=update_data,
                symmetric=symmetric,
                calculation_method=calculation_method,
                error_tolerance=error_tolerance,
                max_iterations=max_iterations
            )
        except RuntimeError as e:
            errors = validate_batch_data(
                input_data=self.input_data,
                update_data=update_data,
                calculation_type=CalculationType.power_flow,
                symmetric=True
            )
            if errors:
                raise ValueError(errors_to_string(errors, "power_flow", details=True))
            raise RuntimeError(str(e))

    def aggregate_voltage_results(self, results: Dict) -> pd.DataFrame:
        node_results = results[ComponentType.node]
        voltage_data = []

        for i, scenario in enumerate(node_results):
            u_pu = scenario['u_pu']
            node_ids = scenario['id']
            max_idx = np.argmax(u_pu)
            min_idx = np.argmin(u_pu)

            voltage_data.append({
                'max_pu_voltage': u_pu[max_idx],
                'max_voltage_node_id': node_ids[max_idx],
                'min_pu_voltage': u_pu[min_idx],
                'min_voltage_node_id': node_ids[min_idx]
            })

        return pd.DataFrame(voltage_data)

    def aggregate_line_results(self, results: Dict, timestamps: List[datetime]) -> pd.DataFrame:
        line_results = results[ComponentType.line]
        n_scenarios = len(line_results)
        n_lines = len(line_results[0]['id'])

        line_ids = line_results[0]['id']
        all_loadings = np.zeros((n_scenarios, n_lines))
        all_losses = np.zeros((n_scenarios, n_lines))

        for i, scenario in enumerate(line_results):
            all_loadings[i] = scenario['loading']
            all_losses[i] = (scenario['p_from'] + scenario['p_to']) * 1e-3  # convert to kW

        energy_losses = []
        time_deltas = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 for i in range(len(timestamps) - 1)]

        for line_idx in range(n_lines):
            losses_kw = all_losses[:, line_idx]
            total_energy = 0.0
            for i in range(len(time_deltas)):
                avg_loss = (losses_kw[i] + losses_kw[i+1]) / 2
                total_energy += avg_loss * time_deltas[i]
            energy_losses.append(total_energy)

        max_loadings = np.max(all_loadings, axis=0)
        min_loadings = np.min(all_loadings, axis=0)
        max_timestamps = [timestamps[np.argmax(all_loadings[:, i])] for i in range(n_lines)]
        min_timestamps = [timestamps[np.argmin(all_loadings[:, i])] for i in range(n_lines)]

        return pd.DataFrame({
            'line_id': line_ids,
            'energy_loss_kwh': energy_losses,
            'max_loading_pu': max_loadings,
            'max_loading_timestamp': max_timestamps,
            'min_loading_pu': min_loadings,
            'min_loading_timestamp': min_timestamps
        }).set_index('line_id')

    @classmethod
    def from_json_file(cls, json_path: str):
        try:
            with open(json_path, 'r') as f:
                data = json_deserialize(f.read())
            return cls(data)
        except FileNotFoundError:
            raise ValueError(f"File not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {json_path}")


def trapezoidal_integral(y_values: List[float], x_values: List[datetime]) -> float:
    integral = 0.0
    for i in range(len(y_values) - 1):
        delta_hours = (x_values[i+1] - x_values[i]).total_seconds() / 3600
        integral += 0.5 * (y_values[i] + y_values[i+1]) * delta_hours
    return integral
