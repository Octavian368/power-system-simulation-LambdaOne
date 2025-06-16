import pandas as pd
import numpy as np
import json
from typing import Dict, List
from datetime import datetime
from power_grid_model import (
    PowerGridModel, 
    CalculationMethod,
    CalculationType,  # Added this import
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

class PowerGridCalculator:
    def __init__(self, pgm_input_data: Dict):
        """
        Initialize the PowerGridCalculator with PGM input data.
        
        Args:
            pgm_input_data: Dictionary representing the power grid in PGM input format.
            
        Raises:
            ValueError: If the pgm_input_data is invalid.
        """
        # Validate input data first
        errors = validate_input_data(
            input_data=pgm_input_data,
            calculation_type=CalculationType.power_flow,
            symmetric=True
        )
        if errors:
            error_msg = errors_to_string(errors, "input_data", details=True)
            raise ValueError(error_msg)
        
        # If validation passes, create the model
        self.model = PowerGridModel(pgm_input_data)
        self.input_data = pgm_input_data
    
    @classmethod
    def from_json_file(cls, json_path: str):
        """
        Create calculator from JSON file.
        
        Args:
            json_path: Path to JSON file containing grid data.
            
        Returns:
            PowerGridCalculator instance initialized with the grid data.
            
        Raises:
            ValueError: If file not found or contains invalid JSON.
        """
        try:
            with open(json_path, 'r') as f:
                data = json_deserialize(f.read())
            return cls(data)
        except FileNotFoundError:
            raise ValueError(f"File not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {json_path}")
    
    def create_batch_update(
        self, 
        active_load_profile: pd.DataFrame, 
        reactive_load_profile: pd.DataFrame
    ) -> Dict:
        """
        Create a batch update dataset from load profiles.
        
        Args:
            active_load_profile: DataFrame with columns ['timestamp', 'load_id', 'value']
            reactive_load_profile: DataFrame with columns ['timestamp', 'load_id', 'value']
            
        Returns:
            Dictionary containing the batch update data.
            
        Raises:
            ValueError: If profiles don't match in timestamps or load IDs.
        """
        # Check if timestamps and load IDs match
        if not active_load_profile['timestamp'].equals(reactive_load_profile['timestamp']):
            raise ValueError("Timestamps in active and reactive profiles don't match")
        
        if not active_load_profile['load_id'].equals(reactive_load_profile['load_id']):
            raise ValueError("Load IDs in active and reactive profiles don't match")
        
        # Get unique timestamps and load IDs
        timestamps = active_load_profile['timestamp'].unique()
        load_ids = active_load_profile['load_id'].unique()
        
        # Initialize update array
        sym_load_update = initialize_array(
            DatasetType.update, 
            ComponentType.sym_load, 
            (len(timestamps), len(load_ids))
        )

        # Set IDs for all scenarios
        sym_load_update['id'] = load_ids
        
        # Populate power values
        for i, ts in enumerate(timestamps):
            ts_active = active_load_profile[active_load_profile['timestamp'] == ts]
            ts_reactive = reactive_load_profile[reactive_load_profile['timestamp'] == ts]
            
            # Ensure order matches
            ts_active = ts_active.sort_values('load_id')
            ts_reactive = ts_reactive.sort_values('load_id')
            
            sym_load_update['p_specified'][i] = ts_active['value'].values
            sym_load_update['q_specified'][i] = ts_reactive['value'].values
        
        return {
            ComponentType.sym_load: sym_load_update
        }
    
    def run_time_series_power_flow(
        self, 
        update_data: Dict,
        symmetric: bool = True,
        calculation_method: CalculationMethod = CalculationMethod.newton_raphson,
        error_tolerance: float = 1e-8,
        max_iterations: int = 20
    ) -> Dict:
        """
        Run time-series (batch) power flow calculation.
        
        Args:
            update_data: Batch update data from create_batch_update
            symmetric: Whether to perform symmetric calculation
            calculation_method: Power flow calculation method
            error_tolerance: Error tolerance for iterative methods
            max_iterations: Maximum iterations for iterative methods
            
        Returns:
            Dictionary containing power flow results for all scenarios.
            
        Raises:
            ValueError: If the batch dataset is invalid.
            RuntimeError: If the power flow calculation fails.
        """
        try:
            return self.model.calculate_power_flow(
                update_data=update_data,
                symmetric=symmetric,
                calculation_method=calculation_method,
                error_tolerance=error_tolerance,
                max_iterations=max_iterations
            )
        except RuntimeError as e:
            # If calculation fails, validate to get detailed errors
            errors = validate_batch_data(
                input_data=self.input_data,
                update_data=update_data,
                calculation_type=CalculationType.power_flow,
                symmetric=True
            )
            if errors:
                error_msg = errors_to_string(errors, "power_flow", details=True)
                raise ValueError(error_msg)
            raise RuntimeError(str(e))
        
    def aggregate_voltage_results(self, results: Dict) -> pd.DataFrame:
        """
        Aggregate voltage results across all timestamps.

        Args:
            results: Power flow results from run_time_series_power_flow

        Returns:
            DataFrame with columns:
            - timestamp (index)
            - max_pu_voltage
            - max_voltage_node_id
            - min_pu_voltage
            - min_voltage_node_id
        """
        node_results = results[ComponentType.node]
        n_scenarios = len(node_results)

        voltage_data = []
        timestamps = []
        for i in range(n_scenarios):
            scenario = node_results[i]
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
            # Try to get timestamp if present in scenario, else use index
            if 'timestamp' in scenario:
                timestamps.append(scenario['timestamp'])
            else:
                timestamps.append(i)

        df = pd.DataFrame(voltage_data)
        df.index = timestamps
        df.index.name = 'timestamp'
        return df
    
    def aggregate_line_results(
        self, 
        results: Dict,
        timestamps: List[datetime]
    ) -> pd.DataFrame:
        """
        Aggregate line results across all timestamps.
        
        Args:
            results: Power flow results from run_time_series_power_flow
            timestamps: List of timestamps corresponding to each scenario
            
        Returns:
            DataFrame with columns:
            - line_id (index)
            - energy_loss_kwh
            - max_loading_pu
            - max_loading_timestamp
            - min_loading_pu
            - min_loading_timestamp
        """
        line_results = results[ComponentType.line]
        n_scenarios = len(line_results)
        n_lines = len(line_results[0]['id'])

        # Initialize storage
        line_ids = line_results[0]['id']
        all_loadings = np.zeros((n_scenarios, n_lines))
        all_losses = np.zeros((n_scenarios, n_lines))

        # Collect data from all scenarios
        for i in range(n_scenarios):
            scenario = line_results[i]
            all_loadings[i] = scenario['loading']
            p_from = scenario['p_from']
            p_to = scenario['p_to']
            all_losses[i] = (p_from + p_to) * 1e-3  # Convert to kW

        # Calculate energy losses using trapezoidal rule
        energy_losses = []
        for line_idx in range(n_lines):
            losses_kw = all_losses[:, line_idx]
            energy_kwh = trapezoidal_integral(losses_kw.tolist(), timestamps)
            energy_losses.append(energy_kwh)

        # Find min/max loading and timestamps
        max_loadings = np.max(all_loadings, axis=0)
        min_loadings = np.min(all_loadings, axis=0)

        max_timestamps = [
            timestamps[np.argmax(all_loadings[:, i])] 
            for i in range(n_lines)
        ]
        min_timestamps = [
            timestamps[np.argmin(all_loadings[:, i])] 
            for i in range(n_lines)
        ]

        # Create DataFrame
        return pd.DataFrame({
            'line_id': line_ids,
            'energy_loss_kwh': energy_losses,
            'max_loading_pu': max_loadings,
            'max_loading_timestamp': max_timestamps,
            'min_loading_pu': min_loadings,
            'min_loading_timestamp': min_timestamps
        }).set_index('line_id')

def trapezoidal_integral(y_values: List[float], x_values: List[datetime]) -> float:
    """
    Calculate discrete numerical integral using the Trapezoidal rule.

    Args:
        y_values: List of function values
        x_values: List of corresponding timestamps

    Returns:
        Integral value
    """
    if len(y_values) < 2 or len(x_values) < 2:
        return 0.0
    integral = 0.0
    for i in range(len(y_values)-1):
        delta_hours = (x_values[i+1] - x_values[i]).total_seconds() / 3600
        integral += 0.5 * (y_values[i] + y_values[i+1]) * delta_hours
    return integral