import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from power_grid_model.validation import ValidationException
from power_system_simulation.power_grid_calculation_model import (
    LoadIdsDoNotMatchError,
    PowerGridCalculator,
    TimestampMismatchError,
)

INPUT_DIR = Path(__file__).parent.parent / "input"


def sample_input_data():
    return {
        "version": "1.0",
        "type": "input",
        "is_batch": False,
        "attributes": {},
        "data": {
            "node": [{"id": 1, "u_rated": 10000}, {"id": 2, "u_rated": 10000}],
            "line": [
                {
                    "id": 10,
                    "from_node": 1,
                    "to_node": 2,
                    "from_status": 1,
                    "to_status": 1,
                    "r1": 0.01,
                    "x1": 0.01,
                    "c1": 1e-9,
                    "tan1": 0,
                    "i_n": 1000,
                }
            ],
            "sym_load": [
                {
                    "id": 100,
                    "node": 1,
                    "p_specified": 1.0,
                    "q_specified": 0.5,
                    "status": 1,
                    "type": 1,
                }
            ],
        },
    }


def test_valid_create_batch_update():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", name="Timestamp")
    ids = [100]

    active = pd.DataFrame(np.ones((3, 1)), index=timestamps, columns=ids)
    reactive = pd.DataFrame(np.ones((3, 1)) * 0.5, index=timestamps, columns=ids)

    update = calc.create_batch_update(active, reactive)
    assert 100 in update["sym_load"]["id"]
    assert update["sym_load"]["p_specified"].shape == (3, 1)


def test_mismatched_timestamps():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    df1 = pd.DataFrame({100: [1, 2]}, index=pd.date_range("2024-01-01", periods=2, freq="h", name="Timestamp"))
    df2 = pd.DataFrame({100: [1, 2]}, index=pd.date_range("2024-01-02", periods=2, freq="h", name="Timestamp"))

    with pytest.raises(TimestampMismatchError):
        calc.create_batch_update(df1, df2)


def test_mismatched_columns():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    ts = pd.date_range("2024-01-01", periods=2, freq="h", name="Timestamp")
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

    ts = pd.date_range("2024-01-01", periods=2, freq="h", name="Timestamp")
    df = pd.DataFrame({100: [1, 2]}, index=ts)

    update = calc.create_batch_update(df, df)
    results = calc.run_time_series_power_flow(update)
    assert "node" in results


def test_voltage_aggregation():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    ts = pd.date_range("2024-01-01", periods=2, freq="h", name="Timestamp")
    df = pd.DataFrame({100: [1, 2]}, index=ts)

    update = calc.create_batch_update(df, df)
    results = calc.run_time_series_power_flow(update)
    voltage = calc.aggregate_voltage_results(results)
    assert "Max_Voltage" in voltage.columns


def test_line_aggregation():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    ts = pd.date_range("2024-01-01", periods=2, freq="h", name="Timestamp")
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


def test_convert_structured_array_direct():
    from power_system_simulation.power_grid_calculation_model import _convert_to_columnar_format

    dtype = [
        ("id", "i4"),
        ("node", "i4"),
        ("p_specified", "f8"),
        ("q_specified", "f8"),
        ("status", "i1"),
        ("type", "i1"),
    ]
    structured_array = np.array([(100, 1, 1.0, 0.5, 1, 1)], dtype=dtype)
    data = {"sym_load": structured_array}

    result = _convert_to_columnar_format(data)
    assert isinstance(result["sym_load"], dict)
    assert "id" in result["sym_load"]
    assert result["sym_load"]["p_specified"][0] == 1.0


def test_from_json_file_invalid_json(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not a json")

    with pytest.raises(ValueError, match="Invalid JSON"):
        PowerGridCalculator.from_json_file(str(bad_path))


def test_run_power_flow_with_invalid_data():
    data = sample_input_data()
    calc = PowerGridCalculator(data)

    bad_update = {
        "sym_load": {
            "id": np.array([999], dtype=np.int32),
            "p_specified": np.array([[1.0]]),
            "q_specified": np.array([[0.5]]),
        }
    }

    with pytest.raises(ValueError, match="Data buffers must be consistent"):
        calc.run_time_series_power_flow(bad_update)


def test_invalid_input_data_triggers_validation_error():
    bad_data = {
        "version": "1.0",
        "type": "input",
        "is_batch": False,
        "attributes": {},
        "data": {
            "line": [
                {
                    "id": 10,
                    "from_node": 1,
                    "to_node": 2,
                    "from_status": 1,
                    "to_status": 1,
                    "r1": 0.01,
                    "x1": 0.01,
                    "c1": 1e-9,
                    "tan1": 0,
                    "i_n": 1000,
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="input_data"):
        PowerGridCalculator(bad_data)
