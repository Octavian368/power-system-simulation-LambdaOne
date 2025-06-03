# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Import PowerGridCalculator and trapezoidal_integral from your module
from power_system_simulation.power_grid import PowerGridCalculator, trapezoidal_integral


# Define dummy/mocked classes for ValidationException and BatchResult
class ValidationException(Exception):
    pass

class BatchResult:
    def __init__(self):
        # Dummy internal structure to mimic actual BatchResult behavior
        self._data = {"node": {"u_pu": {}}, "line": {"p_loss_pu": {}, "loading_pu": {}}}

    def get_data(self, element_type, attribute):
        return self._data.get(element_type, {}).get(attribute, {})

# --- Mock Data for Testing ---

MOCK_PGM_INPUT_DATA = {
    "base_power_mva": 1.0,
    "grid_elements": {
        "node": [
            {"id": "node_1", "u_nom_kv": 0.4},
            {"id": "node_2", "u_nom_kv": 0.4},
            {"id": "node_3", "u_nom_kv": 0.4},
        ],
        "line": [
            {"id": "line_1_2", "from_node": "node_1", "to_node": "node_2", "r_ohm_per_km": 0.1, "x_ohm_per_km": 0.05, "length_km": 1.0},
            {"id": "line_2_3", "from_node": "node_2", "to_node": "node_3", "r_ohm_per_km": 0.1, "x_ohm_per_km": 0.05, "length_km": 1.0},
        ],
        "sym_load": [
            {"id": "load_1", "node": "node_1", "p_mw": 0.1, "q_mvar": 0.05},
            {"id": "load_2", "node": "node_2", "p_mw": 0.2, "q_mvar": 0.1},
        ],
        "external_grid": [
            {"id": "ext_grid_1", "node": "node_1", "u_pu": 1.0},
        ]
    }
}

# The rest of your code can remain the same for generating mock profiles and tests...
# (Copy everything from your original file except the incorrect import line)

# Replace:
# from power_grid_model import ValidationException, BatchResult
# With:
# [dummy classes above]
