"""
Assignment 3: Power System Simulation

This script provides a modular structure for LV grid analytics, including:
- Data validation
- EV penetration assignment
- Optimal tap position analysis
- N-1 contingency analysis
"""

# import json

import numpy as np
import pandas as pd

from power_grid_model import CalculationType, ComponentType  # , DatasetType, PowerGridModel, initialize_array
from power_grid_model.validation import (
    ValidationException,
    assert_valid_input_data,
    errors_to_string,
)
from src.power_system_simulation.graph_processing import GraphProcessor

# 1. Data Validation


def validate_lv_grid(network_data, feeder_ids, active_profile, reactive_profile, ev_profiles):
    """
    Validate the LV grid and all input profiles according to Assignment 3 requirements.
    Raises ValidationException or ValueError with details if validation fails.
    """

    # 1. Validate PGM input data structure
    try:
        assert_valid_input_data(network_data, calculation_type=CalculationType.power_flow, symmetric=True)
    except ValidationException as ex:
        print("Input data validation failed:")
        print(errors_to_string(ex.errors, name="input_data", details=True))
        raise

    # 2. Check for exactly one transformer and one source
    nodes = network_data.get("node", []) if isinstance(network_data, dict) else network_data[ComponentType.node]
    transformers = [n for n in nodes if n.get("type", None) == "transformer"] if isinstance(nodes, list) else []
    sources = network_data.get("source", []) if isinstance(network_data, dict) else network_data[ComponentType.source]
    if len(transformers) != 1:
        raise ValueError(f"Expected exactly one transformer, found {len(transformers)}.")
    if len(sources) != 1:
        raise ValueError(f"Expected exactly one source, found {len(sources)}.")

    # 3. Check LV feeder IDs are valid line IDs
    line_ids = set(l["id"] for l in network_data["line"])
    for fid in feeder_ids:
        if fid not in line_ids:
            raise ValueError(f"Feeder ID {fid} is not a valid line ID.")

    # 4. Check feeder lines connect to transformer output 
    transformer_to_node = transformers[0]["to_node"]
    for fid in feeder_ids:
        feeder_line = next(l for l in network_data["line"] if l["id"] == fid)
        if feeder_line["from_node"] != transformer_to_node:
            raise ValueError(f"Feeder line {fid} does not start at transformer output node {transformer_to_node}.")

    # 5. Check timestamps and IDs match across all profiles
    if not active_profile.index.equals(reactive_profile.index):
        raise ValueError("Timestamps do not match between active and reactive profiles.")
    if set(active_profile.columns) != set(reactive_profile.columns):
        raise ValueError("Load IDs do not match between active and reactive profiles.")
    sym_load_ids = set(l["id"] for l in network_data["sym_load"])
    if not set(active_profile.columns).issubset(sym_load_ids):
        raise ValueError("Some load IDs in profiles are not valid sym_load IDs.")

    # 6. Check enough EV profiles for all sym_loads
    if len(ev_profiles) < len(sym_load_ids):
        raise ValueError("Not enough EV profiles for all sym_loads.")

    print("All validation checks passed.")


# 2. EV Penetration Assignment


def assign_ev_profiles(sym_load_ids, ev_profiles, penetration_level, feeders, random_seed=42):
    """
    Assign EV profiles
    """
    np.random.seed(random_seed)
    n_houses = len(sym_load_ids)
    n_feeders = len(feeders)
    n_evs_per_feeder = int(np.floor(penetration_level * n_houses / n_feeders))
    assignments = {}
    available_profiles = list(ev_profiles)
    for feeder in feeders:
        feeder_houses = sym_load_ids  # Replace with actual feeder-house mapping
        selected = np.random.choice(feeder_houses, n_evs_per_feeder, replace=False)
        for house in selected:
            profile = available_profiles.pop()
            assignments[house] = profile
    return assignments


def assign_ev_penetration(network_data, feeder_ids, ev_profiles, penetration_level, random_seed=42):
    """
    Assign EV profiles to houses according to the assignment 3 requirements.
    Returns a dict {house_id: ev_profile} for all houses with EVs.
    """

    np.random.seed(random_seed)

    # Get all sym_load IDs (houses)
    sym_loads = network_data["sym_load"]
    house_ids = [l["id"] for l in sym_loads]
    total_houses = len(house_ids)
    n_feeders = len(feeder_ids)
    n_evs_per_feeder = int(np.floor(penetration_level * total_houses / n_feeders))

    # Build graph for feeder mapping
    nodes = [n["id"] for n in network_data["node"]]
    lines = network_data["line"]
    edge_ids = [l["id"] for l in lines]
    edge_vertex_id_pairs = [(l["from_node"], l["to_node"]) for l in lines]
    edge_enabled = [(l["from_status"] == 1 and l["to_status"] == 1) for l in lines]
    source_id = network_data["source"][0]["node"]
    gp = GraphProcessor(nodes, edge_ids, edge_vertex_id_pairs, edge_enabled, source_id)

    # Map houses to feeders (find all sym_loads downstream of each feeder line)
    feeder_to_houses = {}
    for feeder_id in feeder_ids:
        downstream_nodes = gp.find_downstream_vertices(feeder_id)
        feeder_houses = [l["id"] for l in sym_loads if l["node"] in downstream_nodes]
        feeder_to_houses[feeder_id] = feeder_houses

    # Assign EVs per feeder
    assignments = {}
    available_profiles = list(ev_profiles)
    np.random.shuffle(available_profiles)
    for feeder_id, houses in feeder_to_houses.items():
        if len(houses) < n_evs_per_feeder:
            selected = houses  # assign to all if not enough
        else:
            selected = list(np.random.choice(houses, n_evs_per_feeder, replace=False))
        for house in selected:
            if not available_profiles:
                raise ValueError("Not enough EV profiles for all assignments.")
            assignments[house] = available_profiles.pop()
    return assignments


# 3. Optimal Tap Position Analysis


def find_optimal_tap(calc, tap_positions, criterion="loss"):
    """
    Finding optimal tap
    """
    best_tap = None
    best_metric = float("inf")
    for tap in tap_positions:
        # TODO: Set tap position in network data and run power flow
        metric = np.random.rand()  # Placeholder
        if metric < best_metric:
            best_metric = metric
            best_tap = tap
    return best_tap


# 4. N-1 Contingency Analysis


def n_minus_one_analysis(calc, outaged_line_id):
    """
    N-1 calculation
    """
    # Disconnect the line, find alternatives, run power flow for each
    # Return a summary DataFrame
    return pd.DataFrame({"Alternative_Line_ID": [], "Max_Loading": [], "Line_ID_of_Max": [], "Timestamp_of_Max": []})
