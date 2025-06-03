import pytest
from power_system_simulation.graph_processing import (
    GraphProcessor,
    IDNotFoundError,
    InputLengthDoesNotMatchError,
    IDNotUniqueError,
    GraphNotFullyConnectedError,
    GraphCycleError,
    EdgeAlreadyDisabledError,
)
import networkx as nx

def test_initialization_and_validation():
    with pytest.raises(ValueError):
        GraphProcessor([], [], [], [], 0)
    with pytest.raises(IDNotUniqueError):
        GraphProcessor([1, 1], [10], [(1, 2)], [True], 1)
    with pytest.raises(IDNotFoundError):
        GraphProcessor([1, 2], [10], [(1, 3)], [True], 1)
    with pytest.raises(InputLengthDoesNotMatchError):
        GraphProcessor([1, 2], [10], [(1, 2), (2, 3)], [True], 1)
    with pytest.raises(GraphNotFullyConnectedError):
        GraphProcessor([1, 2, 3], [10], [(1, 2)], [True], 1)
    with pytest.raises(GraphCycleError):
        GraphProcessor([1, 2, 3], [10, 20, 30], [(1, 2), (2, 3), (3, 1)], [True, True, True], 1)

def test_find_downstream_vertices_and_subgraph():
    gp = GraphProcessor([1, 2, 3], [10, 20], [(1, 2), (2, 3)], [True, True], 1)
    downstream = gp.find_downstream_vertices(20)
    assert set(downstream) == {3}
    subgraph = gp.find_downstream_subgraph(20)
    assert set(subgraph.nodes) == {3}

def test_find_alternative_edges():
    # Add backup edge 1-3 to act as alternative when 20 is disabled
    gp = GraphProcessor([1, 2, 3], [10, 20, 30], [(1, 2), (2, 3), (1, 3)], [True, True, False], 1)
    alternatives = gp.find_alternative_edges(20)
    assert sorted(alternatives) == [30]

def test_edge_already_disabled_error():
    gp = GraphProcessor([1, 2], [10], [(1, 2)], [True], 1)
    gp.disable_edge(10)
    with pytest.raises(EdgeAlreadyDisabledError):
        gp.find_alternative_edges(10)

def test_enable_disable_edge():
    gp = GraphProcessor([1, 2, 3], [10, 20], [(1, 2), (2, 3)], [True, True], 1)
    gp.disable_edge(20)
    assert not gp.edge_enabled[1]
    gp.enable_edge(20)
    assert gp.edge_enabled[1]

def test_find_downstream_vertices_edge_absent():
    gp = GraphProcessor([1, 2], [10], [(1, 2)], [True], 1)
    gp.disable_edge(10)
    result = gp.find_downstream_vertices(10)
    assert result == []

def test_find_alternative_edges_when_connected():
    gp = GraphProcessor([1, 2, 3], [10, 20], [(1, 2), (2, 3)], [True, True], 1)
    result = gp.find_alternative_edges(10)
    assert result == []

def test_dynamic_disable_breaks_connectivity():
    # Correct: let find_downstream_vertices handle disabling
    gp = GraphProcessor([1, 2, 3], [10, 20], [(1, 2), (2, 3)], [True, True], 1)
    downstream = gp.find_downstream_vertices(20)
    assert set(downstream) == {3}

def test_dynamic_enable_reconnects_graph():
    gp = GraphProcessor([1, 2, 3, 4], [10, 20, 30], [(1, 2), (2, 3), (3, 4)], [True, True, True], 1)
    gp.disable_edge(20)
    assert not nx.is_connected(gp.graph)
    gp.enable_edge(20)
    assert nx.is_connected(gp.graph)

def test_invalid_edge_id():
    gp = GraphProcessor([1, 2], [10], [(1, 2)], [True], 1)
    with pytest.raises(IDNotFoundError):
        gp.find_downstream_vertices(999)
    with pytest.raises(IDNotFoundError):
        gp.find_alternative_edges(999)
    with pytest.raises(IDNotFoundError):
        gp.enable_edge(999)
    with pytest.raises(IDNotFoundError):
        gp.disable_edge(999)
