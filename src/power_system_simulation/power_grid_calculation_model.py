import json
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

from power_grid_model import (
    PowerGridModel,
    CalculationMethod,
    CalculationType,
    initialize_array,
    DatasetType,
    ComponentType,
)
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import (
    validate_input_data,
    validate_batch_data,
    errors_to_string,
)


class TimestampMismatchError(Exception):
    """Raised when the timestamps of active and reactive profiles don't match."""
    pass


class LoadIdsDoNotMatchError(Exception):
    """Raised when the IDs in active and reactive profiles differ."""
    pass


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
            symmetric=True,
        )
        if errors:
            raise ValueError(errors_to_string(errors, "input_data", details=True))

        self.model = PowerGridModel(columnar_data)
        self.input_data = columnar_data

    def create_batch_update(self, active_load_profile: pd.DataFrame, reactive_load_profile: pd.DataFrame) -> Dict:
        active_df = active_load_profile.reset_index().melt(id_vars=["Timestamp"], var_name="load_id", value_name="value")
        reactive_df = reactive_load_profile.reset_index().melt(id_vars=["Timestamp"], var_name="load_id", value_name="value")

        active_df.rename(columns={"Timestamp": "timestamp"}, inplace=True)
        reactive_df.rename(columns={"Timestamp": "timestamp"}, inplace=True)

        if 'timestamp' not in active_df.columns or 'timestamp' not in reactive_df.columns:
            raise KeyError("Missing 'timestamp' column in input DataFrames.")

        if not active_df['timestamp'].equals(reactive_df['timestamp']):
            raise TimestampMismatchError("Timestamps in active and reactive profiles do not match.")

        if 'load_id' not in active_df.columns or 'load_id' not in reactive_df.columns:
            raise ValueError("Missing 'load_id' column in input DataFrames.")

        if not active_df['load_id'].equals(reactive_df['load_id']):
            raise LoadIdsDoNotMatchError("Load IDs in active and reactive profiles don't match.")

        timestamps = active_df['timestamp'].unique()
        load_ids = active_df['load_id'].unique()

        sym_load_update = initialize_array(
            DatasetType.update,
            ComponentType.sym_load,
            (len(timestamps), len(load_ids)),
        )
        sym_load_update['id'] = load_ids

        for i, ts in enumerate(timestamps):
            ts_active = active_df[active_df['timestamp'] == ts].sort_values('load_id')
            ts_reactive = reactive_df[reactive_df['timestamp'] == ts].sort_values('load_id')

            sym_load_update['p_specified'][i] = ts_active['value'].values
            sym_load_update['q_specified'][i] = ts_reactive['value'].values

        return {ComponentType.sym_load: sym_load_update}

    def run_time_series_power_flow(self, update_data: Dict, symmetric=True,
                                   calculation_method=CalculationMethod.newton_raphson,
                                   error_tolerance=1e-8, max_iterations=20) -> Dict:
        try:
            return self.model.calculate_power_flow(
                update_data=update_data,
                symmetric=symmetric,
                calculation_method=calculation_method,
                error_tolerance=error_tolerance,
                max_iterations=max_iterations,
            )
        except RuntimeError as e:
            errors = validate_batch_data(
                input_data=self.input_data,
                update_data=update_data,
                calculation_type=CalculationType.power_flow,
                symmetric=True,
            )
            if errors:
                raise ValueError(errors_to_string(errors, "power_flow", details=True))
            raise

    def aggregate_voltage_results(self, results: Dict) -> pd.DataFrame:
        voltage_data = []
        for scenario in results[ComponentType.node]:
            u_pu = scenario['u_pu']
            node_ids = scenario['id']
            voltage_data.append({
                'Max_Voltage': np.max(u_pu),
                'Max_Voltage_Node': node_ids[np.argmax(u_pu)],
                'Min_Voltage': np.min(u_pu),
                'Min_Voltage_Node': node_ids[np.argmin(u_pu)],
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
            all_losses[i] = (scenario['p_from'] + scenario['p_to']) * 1e-3

        energy_losses = []
        time_deltas = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 for i in range(len(timestamps)-1)]

        for line_idx in range(n_lines):
            total_energy = 0.0
            for i in range(len(time_deltas)):
                avg_loss = (all_losses[i][line_idx] + all_losses[i+1][line_idx]) / 2
                total_energy += avg_loss * time_deltas[i]
            energy_losses.append(total_energy)

        max_loadings = np.max(all_loadings, axis=0)
        min_loadings = np.min(all_loadings, axis=0)
        max_timestamps = [timestamps[np.argmax(all_loadings[:, i])] for i in range(n_lines)]
        min_timestamps = [timestamps[np.argmin(all_loadings[:, i])] for i in range(n_lines)]

        return pd.DataFrame({
        'line_id': line_ids,
        'Total_Loss': energy_losses,
        'Max_Loading': max_loadings,
        'Max_Loading_Timestamp': max_timestamps,
        'Min_Loading': min_loadings,
        'Min_Loading_Timestamp': min_timestamps
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
