import json
import pandas as pd
import numpy as np
from pathlib import Path
from pandas.testing import assert_frame_equal
import pytest

from power_system_simulation.power_grid_calculation_model import PowerGridCalculator

# Set up paths
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "src" / "power_system_simulation"
EXPECTED_DIR = BASE_DIR / "expected_output"


def test_power_flow_pipeline_against_expected():
    with open(INPUT_DIR / "input_network_data.json") as f:
        pgm_data = json.load(f)

    active_df = pd.read_parquet(INPUT_DIR / "active_power_profile.parquet")
    reactive_df = pd.read_parquet(INPUT_DIR / "reactive_power_profile.parquet")

    expected_node_df = pd.read_parquet(EXPECTED_DIR / "output_table_row_per_timestamp.parquet")
    expected_line_df = pd.read_parquet(EXPECTED_DIR / "output_table_row_per_line.parquet")

    calc = PowerGridCalculator(pgm_data)
    dataset = calc.create_batch_update(active_df, reactive_df)
    results = calc.run_time_series_power_flow(dataset)

    node_df = calc.aggregate_voltage_results(results).sort_index().round(5)
    line_df = calc.aggregate_line_results(results, active_df.index.to_list()).sort_index().round(5)


    expected_node_df = expected_node_df.sort_index().round(5)
    expected_line_df = expected_line_df.sort_index().round(5)

    assert_frame_equal(node_df, expected_node_df, check_dtype=False, check_exact=False, atol=1e-4)
    assert_frame_equal(line_df, expected_line_df, check_dtype=False, check_exact=False, atol=1e-4)


def test_mismatched_timestamps():
    pgm_data = {
    "node": [{"id": 1}],
    "line": [],
    "sym_load": [{"id": 1, "node": 1}]
}
    calc = PowerGridCalculator(pgm_data)
    df1 = pd.DataFrame({1: [1]}, index=[pd.Timestamp("2024-01-01 00:00")])
    df2 = pd.DataFrame({1: [1]}, index=[pd.Timestamp("2024-01-01 01:00")])
    with pytest.raises(ValueError, match="Timestamps in active and reactive profiles do not match"):
        calc.create_batch_update(df1, df2)


def test_mismatched_load_ids():
    pgm_data = {"node": [], "line": []}
    calc = PowerGridCalculator(pgm_data)
    df1 = pd.DataFrame({1: [1]}, index=[pd.Timestamp("2024-01-01 00:00")])
    df2 = pd.DataFrame({2: [1]}, index=[pd.Timestamp("2024-01-01 00:00")])
    with pytest.raises(ValueError, match="Load IDs in active and reactive profiles don't match"):
        calc.create_batch_update(df1, df2)


def test_missing_timestamp_column():
    pgm_data = {"node": [], "line": []}
    calc = PowerGridCalculator(pgm_data)
    df1 = pd.DataFrame({1: [1]})
    df2 = pd.DataFrame({1: [1]})
    with pytest.raises(KeyError):
        calc.create_batch_update(df1, df2)