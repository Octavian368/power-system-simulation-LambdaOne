"""
This module provides the PowerGridCalculator class and utility functions for performing power grid calculations, including power flow, load updates, and voltage aggregation.
It handles the processing of grid data and provides methods to calculate and aggregate power flow results.
"""

import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from power_grid_model import (
    CalculationMethod,
    CalculationType,
    ComponentType,
    DatasetType,
    PowerGridModel,
    initialize_array,
)
from power_grid_model._core.errors import PowerGridSerializationError
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import (
    errors_to_string,
    validate_batch_data,
    validate_input_data,
)


class TimestampMismatchError(Exception):
    """Raised when timestamps in load profiles do not match."""


class LoadIdsDoNotMatchError(Exception):
    """Raised when load IDs in profiles do not match."""


def _convert_to_columnar_format(data: dict) -> dict:
    """
    Converts input data into columnar format for easier processing.

    Args:
        data (dict): The input data to be converted.

    Returns:
        dict: Columnar format of the input data.
    """
    columnar_data = {}
    int32_fields = {"id", "from_node", "to_node", "node"}
    int8_fields = {"status", "from_status", "to_status", "type"}

    for component, entries in data.items():
        # Check if entries is empty or None
        if entries is None or (isinstance(entries, np.ndarray) and entries.size == 0):
            columnar_data[component] = {}
            continue

        # Structured NumPy array from deserialization
        if isinstance(entries, np.ndarray) and entries.dtype.names:
            columnar_data[component] = {
                key: entries[key].astype(
                    np.int32
                    if key in int32_fields
                    else np.int8 if key in int8_fields else np.float64
                )
                for key in entries.dtype.names
            }
            continue

        # Fallback for list-of-dict
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
    """
    Manages power grid calculations, including updates, power flow, and aggregation.
    """

    def __init__(self, pgm_input_data: Dict):
        """
        Initializes the PowerGridCalculator with input data.

        Args:
            pgm_input_data (Dict): Input data for power grid calculations.
        """
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

    def create_batch_update(
        self, active_load_profile: pd.DataFrame, reactive_load_profile: pd.DataFrame
    ) -> Dict:
        """
        Creates a batch update for active and reactive profiles.

        Args:
            active_load_profile (pd.DataFrame): DataFrame containing active load profile data.
            reactive_load_profile (pd.DataFrame): DataFrame containing reactive load profile data.

        Returns:
            Dict: Batch update for active and reactive loads.
        """
        if not isinstance(active_load_profile.index, pd.DatetimeIndex) or not isinstance(
            reactive_load_profile.index, pd.DatetimeIndex
        ):
            raise ValueError("Input DataFrames must have a DatetimeIndex.")

        active_df = active_load_profile.copy()
        reactive_df = reactive_load_profile.copy()
        active_df.index.name = "Timestamp"
        reactive_df.index.name = "Timestamp"

        active_df = active_df.reset_index()
        reactive_df = reactive_df.reset_index()

        if not active_df["Timestamp"].equals(reactive_df["Timestamp"]):
            raise TimestampMismatchError("Timestamps in active and reactive profiles do not match.")
        if not all(active_df.columns == reactive_df.columns):
            raise LoadIdsDoNotMatchError("Load IDs in active and reactive profiles don't match.")

        active_df.rename(columns={"Timestamp": "timestamp"}, inplace=True)
        reactive_df.rename(columns={"Timestamp": "timestamp"}, inplace=True)

        active_df = active_df.melt(id_vars=["timestamp"], var_name="load_id", value_name="value")
        reactive_df = reactive_df.melt(
            id_vars=["timestamp"], var_name="load_id", value_name="value"
        )

        timestamps = active_df["timestamp"].unique()
        load_ids = active_df["load_id"].unique()

        sym_load_update = initialize_array(
            DatasetType.update,
            ComponentType.sym_load,
            (len(timestamps), len(load_ids)),
        )
        sym_load_update["id"] = load_ids

        # Updated loop with enumerate to fix the pylint issue
        for i, ts in enumerate(timestamps):
            ts_active = active_df[active_df["timestamp"] == ts].sort_values("load_id")
            ts_reactive = reactive_df[reactive_df["timestamp"] == ts].sort_values("load_id")
            sym_load_update["p_specified"][i] = ts_active["value"].values
            sym_load_update["q_specified"][i] = ts_reactive["value"].values

        return {ComponentType.sym_load: sym_load_update}

    def run_time_series_power_flow(
        self,
        update_data: Dict,
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1e-8,
        max_iterations=20,
    ) -> Dict:
        """
        Runs the power flow calculation with the provided update data.

        Args:
            update_data (Dict): The data to update the power grid model.
            symmetric (bool, optional): Whether to use symmetric calculations. Defaults to True.
            calculation_method (CalculationMethod, optional): The method for power flow calculations.
            error_tolerance (float, optional): The error tolerance for calculations. Defaults to 1e-8.
            max_iterations (int, optional): The maximum number of iterations. Defaults to 20.

        Returns:
            Dict: Results of the power flow calculation.
        """
        try:
            return self.model.calculate_power_flow(
                update_data=update_data,
                symmetric=symmetric,
                calculation_method=calculation_method,
                error_tolerance=error_tolerance,
                max_iterations=max_iterations,
            )
        except RuntimeError as exc:
            errors = validate_batch_data(
                input_data=self.input_data,
                update_data=update_data,
                calculation_type=CalculationType.power_flow,
                symmetric=True,
            )
            if errors:
                raise ValueError(errors_to_string(errors, "power_flow", details=True)) from exc
            raise

    def aggregate_voltage_results(self, results: Dict) -> pd.DataFrame:
        """
        Aggregates the voltage results from the power flow calculations.

        Args:
            results (Dict): The results of the power flow calculation.

        Returns:
            pd.DataFrame: Aggregated voltage data.
        """
        voltage_data = []
        for scenario in results[ComponentType.node]:
            u_pu = scenario["u_pu"]
            node_ids = scenario["id"]
            voltage_data.append(
                {
                    "Max_Voltage": np.max(u_pu),
                    "Max_Voltage_Node": node_ids[np.argmax(u_pu)],
                    "Min_Voltage": np.min(u_pu),
                    "Min_Voltage_Node": node_ids[np.argmin(u_pu)],
                }
            )
        return pd.DataFrame(voltage_data)

    def aggregate_line_results(self, results: Dict, timestamps: List[datetime]) -> pd.DataFrame:
        """
        Aggregates the line results from the power flow calculations.

        Args:
            results (Dict): The results of the power flow calculations.
            timestamps (List[datetime]): The list of timestamps for the calculations.

        Returns:
            pd.DataFrame: Aggregated line data.
        """
        line_results = results[ComponentType.line]
        n_scenarios = len(line_results)
        n_lines = len(line_results[0]["id"])

        line_ids = line_results[0]["id"]
        all_loadings = np.zeros((n_scenarios, n_lines))
        all_losses = np.zeros((n_scenarios, n_lines))

        for i, scenario in enumerate(line_results):
            all_loadings[i] = scenario["loading"]
            all_losses[i] = (scenario["p_from"] + scenario["p_to"]) * 1e-3

        energy_losses = []  # Ensure energy_losses is initialized
        time_deltas = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
            for i in range(len(timestamps) - 1)
        ]

        for line_idx in range(n_lines):
            total_energy = 0.0
            for i, delta in enumerate(time_deltas):  # Use enumerate for time_deltas
                avg_loss = (all_losses[i][line_idx] + all_losses[i + 1][line_idx]) / 2
                total_energy += avg_loss * delta

            # Append the calculated total energy for the current line to energy_losses
            energy_losses.append(total_energy)

        max_loadings = np.max(all_loadings, axis=0)
        min_loadings = np.min(all_loadings, axis=0)
        max_timestamps = [timestamps[np.argmax(all_loadings[:, i])] for i in range(n_lines)]
        min_timestamps = [timestamps[np.argmin(all_loadings[:, i])] for i in range(n_lines)]

        return pd.DataFrame(
            {
                "line_id": line_ids,
                "Total_Loss": energy_losses,  # Ensure total energy is included here
                "Max_Loading": max_loadings,
                "Max_Loading_Timestamp": max_timestamps,
                "Min_Loading": min_loadings,
                "Min_Loading_Timestamp": min_timestamps,
            }
        ).set_index("line_id")

    @classmethod
    def from_json_file(cls, json_path: str):
        """
        Loads data from a JSON file and returns a PowerGridCalculator instance.

        Args:
            json_path (str): The path to the JSON file.

        Returns:
            PowerGridCalculator: A new instance of the calculator with the loaded data.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json_deserialize(f.read())
            return cls(data)
        except FileNotFoundError as exc:
            raise ValueError(f"File not found: {json_path}") from exc
        except (json.JSONDecodeError, PowerGridSerializationError) as exc:
            raise ValueError(f"Invalid JSON in file: {json_path}") from exc
