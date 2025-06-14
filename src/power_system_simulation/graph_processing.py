"""
graph_processing.py

Provides the GraphProcessor class for manipulating undirected graphs
with support for edge disabling and analysis, including downstream queries
and alternative edge suggestions.
"""

from typing import List, Tuple
import networkx as nx

# Custom exceptions to handle various graph-related error scenarios
class IDNotFoundError(Exception):
    """Raised when a requested vertex or edge ID does not exist."""


class InputLengthDoesNotMatchError(Exception):
    """Raised when input lists have mismatched lengths."""


class IDNotUniqueError(Exception):
    """Raised when duplicate IDs are detected in the input."""


class GraphNotFullyConnectedError(Exception):
    """Raised when the initialized graph is not fully connected."""


class GraphCycleError(Exception):
    """Raised when the initialized graph contains cycles."""


class EdgeAlreadyDisabledError(Exception):
    """Raised when attempting to disable an already disabled edge."""


class GraphProcessor:
    """
    Class for handling undirected graphs with functionality to enable/disable edges,
    analyze downstream effects of edge removals, and suggest reconnection edges.
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
        Initializes the graph using node and edge data.
        Performs input validation and builds an internal NetworkX graph.

        Args:
            vertex_ids: List of unique node identifiers.
            edge_ids: List of unique edge identifiers.
            edge_vertex_id_pairs: Corresponding (u, v) tuples per edge.
            edge_enabled: Boolean list tracking which edges are enabled.
            source_vertex_id: Root node for downstream queries.
        """
        # Basic input validation
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

        # Check each edge refers to existing nodes
        for u, v in edge_vertex_id_pairs:
            if u not in vertex_ids or v not in vertex_ids:
                raise IDNotFoundError(f"Edge ({u}, {v}) references missing vertex.")

        # Store inputs
        self.vertex_ids = vertex_ids
        self.edge_ids = edge_ids
        self.edge_vertex_id_pairs = edge_vertex_id_pairs
        self.edge_enabled = edge_enabled
        self.source_vertex_id = source_vertex_id

        # Maps for fast lookup
        self.edge_id_to_index = {eid: idx for idx, eid in enumerate(edge_ids)}
        self.edge_id_to_vertices = dict(zip(edge_ids, edge_vertex_id_pairs))

        # Construct the active graph
        self.graph = self._build_graph()

        # Ensure graph is a valid forest (connected, acyclic)
        if not nx.is_connected(self.graph):
            raise GraphNotFullyConnectedError("The initialized graph is not fully connected.")
        if not nx.is_forest(self.graph):
            raise GraphCycleError("The graph contains cycles.")

    def _build_graph(self) -> nx.Graph:
        """
        Internal method to construct a NetworkX graph from enabled edges only.

        Returns:
            Graph containing all active (enabled) edges.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.vertex_ids)
        for edge_id, (u, v), enabled in zip(self.edge_ids, self.edge_vertex_id_pairs, self.edge_enabled):
            if enabled:
                graph.add_edge(u, v, edge_id=edge_id)
        return graph

    def _get_edge_vertices(self, edge_id: int) -> Tuple[int, int]:
        """
        Retrieve the endpoints of an edge by ID.

        Args:
            edge_id: The edge identifier.

        Returns:
            Tuple containing (u, v) endpoints.

        Raises:
            IDNotFoundError if edge ID is invalid.
        """
        if edge_id not in self.edge_id_to_index:
            raise IDNotFoundError(f"Edge ID {edge_id} not found.")
        return self.edge_vertex_id_pairs[self.edge_id_to_index[edge_id]]

    def find_downstream_vertices(self, edge_id: int) -> List[int]:
        """
        Identify vertices that would be disconnected from the source if this edge is removed.

        Args:
            edge_id: Edge whose removal is being simulated.

        Returns:
            List of downstream vertices (not containing source).
        """
        u, v = self._get_edge_vertices(edge_id)
        if not self.edge_enabled[self.edge_id_to_index[edge_id]]:
            return []

        temp_graph = self.graph.copy()
        if not temp_graph.has_edge(u, v):
            return []

        temp_graph.remove_edge(u, v)
        components = list(nx.connected_components(temp_graph))

        # Look for component disconnected from source, but includes u or v
        downstream_component = next(
            (comp for comp in components if self.source_vertex_id not in comp and (u in comp or v in comp)), None
        )

        return list(downstream_component) if downstream_component else []

    def find_downstream_subgraph(self, edge_id: int) -> nx.Graph:
        """
        Extract the subgraph containing downstream vertices caused by removal of a given edge.

        Args:
            edge_id: Edge to virtually remove.

        Returns:
            Subgraph with disconnected downstream vertices.
        """
        downstream_vertices = self.find_downstream_vertices(edge_id)
        return self.graph.subgraph(downstream_vertices).copy() if downstream_vertices else nx.Graph()

    def find_alternative_edges(self, disabled_edge_id: int) -> List[int]:
        """
        Recommend edges that can be re-enabled to restore connectivity when one is removed.

        Args:
            disabled_edge_id: The edge being hypothetically removed.

        Returns:
            Sorted list of candidate edge IDs that reconnect graph as a forest.

        Raises:
            EdgeAlreadyDisabledError: If the edge is already disabled.
        """
        u, v = self._get_edge_vertices(disabled_edge_id)
        if not self.edge_enabled[self.edge_id_to_index[disabled_edge_id]]:
            raise EdgeAlreadyDisabledError(f"Edge ID {disabled_edge_id} is already disabled.")

        temp_graph = self.graph.copy()
        temp_graph.remove_edge(u, v)

        if nx.is_connected(temp_graph):
            return []

        # Try enabling each disabled edge and check if connectivity is restored
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
        Re-enable a previously disabled edge.

        Args:
            edge_id: Identifier of edge to activate.

        Raises:
            IDNotFoundError if edge is invalid.
        """
        if edge_id not in self.edge_id_to_index:
            raise IDNotFoundError(f"Edge ID {edge_id} not found.")
        self.edge_enabled[self.edge_id_to_index[edge_id]] = True
        self.graph = self._build_graph()  # Rebuild graph with updated state

    def disable_edge(self, edge_id: int) -> None:
        """
        Disable an edge by its ID.

        Args:
            edge_id: Edge identifier to deactivate.

        Raises:
            IDNotFoundError if edge ID is not valid.
        """
        if edge_id not in self.edge_id_to_index:
            raise IDNotFoundError(f"Edge ID {edge_id} not found.")
        self.edge_enabled[self.edge_id_to_index[edge_id]] = False
        self.graph = self._build_graph()  # Rebuild graph with updated state
