import json
import pandas as pd
import numpy as np
import pytest 
from pathlib import Path
from pandas.testing import assert_frame_equal
from power_grid_model import CalculationType
from grid_calculator import PowerGridCalculator, trapezoidal_integral
from datetime import datetime

# Set up paths
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "src" / "power_system_simulation"
EXPECTED_DIR = BASE_DIR / "expected_output"

def _row_to_columnar(row_data):
    if isinstance(row_data, list) and row_data and isinstance(row_data[0], dict):
        columns = {k: np.array([d.get(k) for d in row_data]) for k in row_data[0].keys()}
        return columns
    return row_data

def test_power_flow_pipeline_against_expected():
    """Test the complete power flow pipeline against expected output files"""
    # Load input data
    with open("input/input_network_data.json") as f:
        pgm_data = json.load(f)

    # Convert row-based to columnar for each component
    component_data = {k: _row_to_columnar(v) for k, v in pgm_data["data"].items()}

    active_df = pd.read_parquet("input/active_power_profile.parquet")
    reactive_df = pd.read_parquet("input/reactive_power_profile.parquet")

    # Load expected results
    expected_node_df = pd.read_parquet("tests/output_table_row_per_timestamp.parquet")
    expected_line_df = pd.read_parquet("tests/output_table_row_per_line.parquet")

    # Run power flow pipeline
    calc = PowerGridCalculator(component_data)
    update_data = calc.create_batch_update(active_df, reactive_df)
    results = calc.run_time_series_power_flow(update_data)

    # Get timestamps from active power profile
    timestamps = active_df['timestamp'].unique()

    # Aggregate results
    node_df = calc.aggregate_voltage_results(results)
    node_df.index = timestamps
    node_df.index.name = 'timestamp'
    node_df = node_df.sort_index().round(5)
    line_df = calc.aggregate_line_results(results, timestamps).sort_index().round(5)

    # Prepare expected results
    expected_node_df = expected_node_df.sort_index().round(5)
    expected_line_df = expected_line_df.sort_index().round(5)

    # Compare results
    assert_frame_equal(
        node_df, 
        expected_node_df, 
        check_dtype=False, 
        check_exact=False, 
        atol=1e-4,
        check_index_type=False
    )
    
    assert_frame_equal(
        line_df, 
        expected_line_df, 
        check_dtype=False, 
        check_exact=False, 
        atol=1e-4,
        check_index_type=False
    )

def test_trapezoidal_integral():
    """Test the trapezoidal integration helper function"""
    # Test with uniform hourly intervals
    y = [10, 20, 30]  # kW
    x = pd.date_range(start="2023-01-01", periods=3, freq="h")
    assert trapezoidal_integral(y, x) == pytest.approx(40.0)  # (10+20)/2*1 + (20+30)/2*1

    # Test with non-uniform intervals
    y = [10, 20, 30]
    x = [
        datetime(2023, 1, 1, 0, 0, 0),
        datetime(2023, 1, 1, 0, 30, 0), 
        datetime(2023, 1, 1, 2, 0, 0)
    ]
    assert trapezoidal_integral(y, x) == pytest.approx(45.0)  # (10+20)/2*0.5 + (20+30)/2*1.5

    # Edge cases
    assert trapezoidal_integral([], []) == 0.0
    assert trapezoidal_integral([10], [datetime(2023, 1, 1)]) == 0.0