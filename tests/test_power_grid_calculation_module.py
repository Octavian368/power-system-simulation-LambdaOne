import pytest
import json
import pandas as pd
import numpy as np
from datetime import datetime
from power_system_simulation.power_grid_calculation_model import (
    PowerGridCalculator,
    TimestampMismatchError,
    LoadIdsDoNotMatchError,
)
from power_grid_model.validation import ValidationException


def sample_input_data():
    return {
        "version": "1.0",
        "data": {
            "node": [{"id": 1, "u_rated": 10000}, {"id": 2, "u_rated": 10000}],
            "line": [{"id": 10, "from_node": 1, "to_node": 2, "from_status": 1, "to_status": 1,
                      "r1": 0.01, "x1": 0.01, "c1": 1e-9, "tan1": 0, "i_n": 1000}],
            "sym_load": [{"id": 100, "node": 1, "p_specified": 1.0, "q_specified": 0.5, "status": 1, "type": 1}]
        }
    }


def test_valid_create_batch_update():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    timestamps = pd.date_range("2024-01-01", periods=3, freq="H", name="Timestamp")
    ids = [100]

    active = pd.DataFrame(np.ones((3, 1)), index=timestamps, columns=ids)
    reactive = pd.DataFrame(np.ones((3, 1)) * 0.5, index=timestamps, columns=ids)

    update = calc.create_batch_update(active, reactive)
    assert 100 in update['sym_load']['id']
    assert update['sym_load']['p_specified'].shape == (3, 1)


def test_mismatched_timestamps():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    df1 = pd.DataFrame({100: [1, 2]}, index=pd.date_range("2024-01-01", periods=2, freq="H"))
    df1.index.name = "Timestamp"
    df2 = pd.DataFrame({100: [1, 2]}, index=pd.date_range("2024-01-02", periods=2, freq="H"))
    df2.index.name = "Timestamp"

    with pytest.raises(TimestampMismatchError):
        calc.create_batch_update(df1, df2)


def test_mismatched_columns():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    ts = pd.date_range("2024-01-01", periods=2, freq="H", name="Timestamp")
    df1 = pd.DataFrame({100: [1, 2]}, index=ts)
    df2 = pd.DataFrame({101: [1, 2]}, index=ts)

    with pytest.raises(LoadIdsDoNotMatchError):
        calc.create_batch_update(df1, df2)


def test_invalid_index_type():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    ts = ["2024-01-01 00:00", "2024-01-01 01:00"]
    df1 = pd.DataFrame({100: [1, 2]}, index=pd.DatetimeIndex(ts, name="Timestamp"))
    df2 = pd.DataFrame({100: [1, 2]}, index=[0, 1])

    with pytest.raises(ValueError, match="DatetimeIndex"):
        calc.create_batch_update(df1, df2)


def test_missing_timestamp_column():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    df1 = pd.DataFrame({100: [1]}, index=[pd.Timestamp("2024-01-01 00:00")])
    df1.index.name = "Timestamp"
    df2 = pd.DataFrame({100: [1]})

    with pytest.raises(ValueError, match="DatetimeIndex"):
        calc.create_batch_update(df1, df2)


def test_run_power_flow():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    ts = pd.date_range("2024-01-01", periods=2, freq="H", name="Timestamp")
    df = pd.DataFrame({100: [1, 2]}, index=ts)

    update = calc.create_batch_update(df, df)
    results = calc.run_time_series_power_flow(update)
    assert 'node' in results


def test_voltage_aggregation():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    ts = pd.date_range("2024-01-01", periods=2, freq="H", name="Timestamp")
    df = pd.DataFrame({100: [1, 2]}, index=ts)

    update = calc.create_batch_update(df, df)
    results = calc.run_time_series_power_flow(update)
    voltage = calc.aggregate_voltage_results(results)
    assert "Max_Voltage" in voltage.columns


def test_line_aggregation():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    ts = pd.date_range("2024-01-01", periods=2, freq="H", name="Timestamp")
    df = pd.DataFrame({100: [1, 2]}, index=ts)

    update = calc.create_batch_update(df, df)
    results = calc.run_time_series_power_flow(update)
    df = calc.aggregate_line_results(results, list(ts))
    assert "Total_Loss" in df.columns


def test_from_json_file_valid(tmp_path):
    json_data = sample_input_data()
    json_path = tmp_path / "input.json"
    json_path.write_text(json.dumps(json_data))

    calc = PowerGridCalculator.from_json_file(str(json_path))
    assert isinstance(calc, PowerGridCalculator)


def test_from_json_file_missing():
    with pytest.raises(ValueError, match="File not found"):
        PowerGridCalculator.from_json_file("non_existent.json")


def test_from_json_file_invalid_json(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not a json")

    with pytest.raises(ValueError, match="Invalid JSON"):
        PowerGridCalculator.from_json_file(str(bad_path))
