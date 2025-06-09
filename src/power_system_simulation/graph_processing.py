"""
graph_processing.py

Provides the GraphProcessor class for manipulating undirected graphs
with support for edge disabling and analysis, including downstream queries
and alternative edge suggestions.
"""

from typing import List, Tuple
import networkx as nx

class IDNotFoundError(Exception):
    """Raised when a requested vertex or edge ID does not exist."""
    pass

class InputLengthDoesNotMatchError(Exception):
    """Raised when input lists have mismatched lengths."""
    pass

class IDNotUniqueError(Exception):
    """Raised when duplicate IDs are detected in the input."""
    pass

class GraphNotFullyConnectedError(Exception):
    """Raised when the initialized graph is not fully connected."""
    pass

class GraphCycleError(Exception):
    """Raised when the initialized graph contains cycles."""
    pass

class EdgeAlreadyDisabledError(Exception):
    """Raised when attempting to disable an already disabled edge."""
    pass

class GraphProcessor:
    """
    Processes undirected graphs with edge disabling and analysis features.

    Allows disabling/enabling of edges, querying downstream components,
    and finding alternative edges to restore connectivity.
    """

    def __init__(
        self,
        vertex_ids: List[int],
        edge_ids: List[int],
        edge_vertex_id_pairs: List[Tuple[int, int]],
        edge_enabled: List[bool],
        source_vertex_id: int,
    ) -> None:
        """
        Initialize the GraphProcessor.

        Args:
            vertex_ids: List of unique vertex IDs.
            edge_ids: List of unique edge IDs.
            edge_vertex_id_pairs: List of (u, v) pairs for each edge.
            edge_enabled: List of booleans for enabled state of each edge.
            source_vertex_id: The source vertex ID.

        Raises:
            ValueError: If any required list is empty.
            IDNotUniqueError: If IDs are not unique.
            InputLengthDoesNotMatchError: If edge lists do not match in length.
            IDNotFoundError: If a referenced vertex is missing.
            GraphNotFullyConnectedError: If the graph is not fully connected.
            GraphCycleError: If the graph contains cycles.
        """
        if not vertex_ids or not edge_ids:
            raise ValueError("Vertex and edge lists cannot be empty.")

        if len(set(vertex_ids)) != len(vertex_ids):
            raise IDNotUniqueError("Duplicate vertex IDs detected.")
        if len(set(edge_ids)) != len(edge_ids):
            raise IDNotUniqueError("Duplicate edge IDs detected.")

        if len(edge_ids) != len(edge_vertex_id_pairs) or len(edge_ids) != len(edge_enabled):
            raise InputLengthDoesNotMatchError("Edge lists have inconsistent lengths.")

        if source_vertex_id not in vertex_ids:
            raise IDNotFoundError(f"Source vertex ID {source_vertex_id} not found in vertices.")

        for u, v in edge_vertex_id_pairs:
            if u not in vertex_ids or v not in vertex_ids:
                raise IDNotFoundError(f"Edge ({u}, {v}) references missing vertex.")

        self.vertex_ids = vertex_ids
        self.edge_ids = edge_ids
        self.edge_vertex_id_pairs = edge_vertex_id_pairs
        self.edge_enabled = edge_enabled
        self.source_vertex_id = source_vertex_id

        self.edge_id_to_index = {eid: idx for idx, eid in enumerate(edge_ids)}
        self.edge_id_to_vertices = dict(zip(edge_ids, edge_vertex_id_pairs))

        self.graph = self._build_graph()

        if not nx.is_connected(self.graph):
            raise GraphNotFullyConnectedError("The initialized graph is not fully connected.")
        if not nx.is_forest(self.graph):
            raise GraphCycleError("The graph contains cycles.")

    def _build_graph(self) -> nx.Graph:
        """
        Build a NetworkX graph using only currently enabled edges.

        Returns:
            nx.Graph: The built graph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.vertex_ids)
        for edge_id, (u, v), enabled in zip(self.edge_ids, self.edge_vertex_id_pairs, self.edge_enabled):
            if enabled:
                graph.add_edge(u, v, edge_id=edge_id)
        return graph

    def _get_edge_vertices(self, edge_id: int) -> Tuple[int, int]:
        """
        Get the (u, v) vertex tuple for a given edge ID.

        Args:
            edge_id: The edge ID.

        Returns:
            Tuple[int, int]: The (u, v) tuple for this edge.

        Raises:
            IDNotFoundError: If edge ID not found.
        """
        if edge_id not in self.edge_id_to_index:
            raise IDNotFoundError(f"Edge ID {edge_id} not found.")
        return self.edge_vertex_id_pairs[self.edge_id_to_index[edge_id]]

    def find_downstream_vertices(self, edge_id: int) -> List[int]:
        """
        Find all vertices that would become disconnected from the source if edge_id is removed.

        Args:
            edge_id: The edge ID.

        Returns:
            List[int]: List of downstream vertex IDs (may be empty).
        """
        u, v = self._get_edge_vertices(edge_id)
        if not self.edge_enabled[self.edge_id_to_index[edge_id]]:
            return []

        temp_graph = self.graph.copy()
        if not temp_graph.has_edge(u, v):
            return []

        temp_graph.remove_edge(u, v)
        components = list(nx.connected_components(temp_graph))
        downstream_component = next(
            (comp for comp in components if self.source_vertex_id not in comp and (u in comp or v in comp)), None
        )

        return list(downstream_component) if downstream_component else []

    def find_downstream_subgraph(self, edge_id: int) -> nx.Graph:
        """
        Return the subgraph induced by downstream vertices if edge_id is removed.

        Args:
            edge_id: The edge ID.

        Returns:
            nx.Graph: The subgraph (may be empty).
        """
        downstream_vertices = self.find_downstream_vertices(edge_id)
        return self.graph.subgraph(downstream_vertices).copy() if downstream_vertices else nx.Graph()

    def find_alternative_edges(self, disabled_edge_id: int) -> List[int]:
        """
        Suggest disabled edges whose re-enabling would reconnect the graph (as a forest).

        Args:
            disabled_edge_id: The ID of the edge being disabled.

        Returns:
            List[int]: Sorted list of alternative edge IDs.

        Raises:
            EdgeAlreadyDisabledError: If the given edge is already disabled.
        """
        u, v = self._get_edge_vertices(disabled_edge_id)
        if not self.edge_enabled[self.edge_id_to_index[disabled_edge_id]]:
            raise EdgeAlreadyDisabledError(f"Edge ID {disabled_edge_id} is already disabled.")

        temp_graph = self.graph.copy()
        temp_graph.remove_edge(u, v)

        if nx.is_connected(temp_graph):
            return []

        alternatives = []
        for idx, (edge_id, (a, b)) in enumerate(zip(self.edge_ids, self.edge_vertex_id_pairs)):
            if not self.edge_enabled[idx]:
                check_graph = temp_graph.copy()
                check_graph.add_edge(a, b, edge_id=edge_id)
                if nx.is_connected(check_graph) and nx.is_forest(check_graph):
                    alternatives.append(edge_id)

        return sorted(alternatives)

    def enable_edge(self, edge_id: int) -> None:
        """
        Enable a given edge by edge ID.

        Args:
            edge_id: The edge ID to enable.

        Raises:
            IDNotFoundError: If the edge ID is not found.
        """
        if edge_id not in self.edge_id_to_index:
            raise IDNotFoundError(f"Edge ID {edge_id} not found.")
        self.edge_enabled[self.edge_id_to_index[edge_id]] = True
        self.graph = self._build_graph()

    def disable_edge(self, edge_id: int) -> None:
        """
        Disable a given edge by edge ID.

        Args:
            edge_id: The edge ID to disable.

        Raises:
            IDNotFoundError: If the edge ID is not found.
        """
        if edge_id not in self.edge_id_to_index:
            raise IDNotFoundError(f"Edge ID {edge_id} not found.")
        self.edge_enabled[self.edge_id_to_index[edge_id]] = False
        self.graph = self._build_graph()
