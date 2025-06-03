import networkx as nx
from typing import List, Tuple

class IDNotFoundError(Exception): pass
class InputLengthDoesNotMatchError(Exception): pass
class IDNotUniqueError(Exception): pass
class GraphNotFullyConnectedError(Exception): pass
class GraphCycleError(Exception): pass
class EdgeAlreadyDisabledError(Exception): pass

class GraphProcessor:
    """
    A class for processing undirected graphs with support for edge disabling and analysis.
    """

    def __init__(
        self,
        vertex_ids: List[int],
        edge_ids: List[int],
        edge_vertex_id_pairs: List[Tuple[int, int]],
        edge_enabled: List[bool],
        source_vertex_id: int,
    ) -> None:
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
        """Helper to build the graph from current edge states."""
        G = nx.Graph()
        G.add_nodes_from(self.vertex_ids)
        for eid, (u, v), enabled in zip(self.edge_ids, self.edge_vertex_id_pairs, self.edge_enabled):
            if enabled:
                G.add_edge(u, v, edge_id=eid)
        return G

    def _get_edge_vertices(self, edge_id: int) -> Tuple[int, int]:
        if edge_id not in self.edge_id_to_index:
            raise IDNotFoundError(f"Edge ID {edge_id} not found.")
        return self.edge_vertex_id_pairs[self.edge_id_to_index[edge_id]]

    def find_downstream_vertices(self, edge_id: int) -> List[int]:
        u, v = self._get_edge_vertices(edge_id)
        if not self.edge_enabled[self.edge_id_to_index[edge_id]]:
            return []

        G_temp = self.graph.copy()
        if not G_temp.has_edge(u, v):
            return []  # Edge already absent

        G_temp.remove_edge(u, v)
        components = list(nx.connected_components(G_temp))
        downstream_component = next((comp for comp in components if self.source_vertex_id not in comp and (u in comp or v in comp)), None)

        return list(downstream_component) if downstream_component else []

    def find_downstream_subgraph(self, edge_id: int) -> nx.Graph:
        downstream_vertices = self.find_downstream_vertices(edge_id)
        return self.graph.subgraph(downstream_vertices).copy() if downstream_vertices else nx.Graph()

    def find_alternative_edges(self, disabled_edge_id: int) -> List[int]:
        u, v = self._get_edge_vertices(disabled_edge_id)
        if not self.edge_enabled[self.edge_id_to_index[disabled_edge_id]]:
            raise EdgeAlreadyDisabledError(f"Edge ID {disabled_edge_id} is already disabled.")

        G_temp = self.graph.copy()
        G_temp.remove_edge(u, v)

        if nx.is_connected(G_temp):
            return []

        # Look for alternative disabled edges that reconnect the graph and keep it a forest
        alternatives = []
        for idx, (eid, (a, b)) in enumerate(zip(self.edge_ids, self.edge_vertex_id_pairs)):
            if not self.edge_enabled[idx]:
                G_check = G_temp.copy()
                G_check.add_edge(a, b, edge_id=eid)
                if nx.is_connected(G_check) and nx.is_forest(G_check):
                    alternatives.append(eid)

        return sorted(alternatives)

    # Optional: add dynamic enabling/disabling methods for extra flexibility
    def enable_edge(self, edge_id: int):
        if edge_id not in self.edge_id_to_index:
            raise IDNotFoundError(f"Edge ID {edge_id} not found.")
        self.edge_enabled[self.edge_id_to_index[edge_id]] = True
        self.graph = self._build_graph()

    def disable_edge(self, edge_id: int):
        if edge_id not in self.edge_id_to_index:
            raise IDNotFoundError(f"Edge ID {edge_id} not found.")
        self.edge_enabled[self.edge_id_to_index[edge_id]] = False
        self.graph = self._build_graph()
