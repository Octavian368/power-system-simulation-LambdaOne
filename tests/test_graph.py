import pytest
import sys
import os


from power_system_simulation.graph_processing import (
    GraphProcessor,
    IDNotFoundError,
    InputLengthDoesNotMatchError,
    IDNotUniqueError,
    GraphNotFullyConnectedError,
    GraphCycleError,
    EdgeAlreadyDisabledError,
)


def test_find_downstream_vertices():
    gp = GraphProcessor(
        vertex_ids=[0, 2, 4],
        edge_ids=[1, 3],
        edge_vertex_id_pairs=[(0, 2), (2, 4)],
        edge_enabled=[True, True],
        source_vertex_id=0,
    )
    assert sorted(gp.find_downstream_vertices(1)) == [2, 4]
    assert sorted(gp.find_downstream_vertices(3)) == [4]


def test_graph_raises_when_disconnected():
    # Edge 3 is disabled, cutting off vertex 4
    with pytest.raises(GraphNotFullyConnectedError):
        GraphProcessor(
            vertex_ids=[0, 2, 4],
            edge_ids=[1, 3],
            edge_vertex_id_pairs=[(0, 2), (2, 4)],
            edge_enabled=[True, False],
            source_vertex_id=0,
        )


def test_invalid_edge_id_raises_error():
    gp = GraphProcessor(
        vertex_ids=[0, 2],
        edge_ids=[1],
        edge_vertex_id_pairs=[(0, 2)],
        edge_enabled=[True],
        source_vertex_id=0,
    )
    with pytest.raises(IDNotFoundError):
        gp.find_downstream_vertices(99)


def test_find_alternative_edges():
    gp = GraphProcessor(
        vertex_ids=[0, 2, 4, 6],
        edge_ids=[1, 3, 5, 7],
        edge_vertex_id_pairs=[(0, 2), (2, 4), (0, 6), (4, 6)],
        edge_enabled=[True, True, True, False],
        source_vertex_id=0,
    )
    assert sorted(gp.find_alternative_edges(3)) == [7]


def test_graph_fails_on_initial_disabled_only_edge():
    # All edges are disabled — should raise error during init
    with pytest.raises(GraphNotFullyConnectedError):
        GraphProcessor(
            vertex_ids=[0, 2],
            edge_ids=[1],
            edge_vertex_id_pairs=[(0, 2)],
            edge_enabled=[False],
            source_vertex_id=0,
        )