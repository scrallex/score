"""
Theme detection and graph analytics for the Sep Text Manifold.

The functions in this module build a co‑occurrence graph of strings and
apply community detection algorithms to identify themes.  They also
compute simple centrality scores used in the connector score.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore


def build_theme_graph(string_windows: Dict[str, Iterable[int]], *, cooccurrence_threshold: int = 1) -> "Graph":
    """Build an undirected co‑occurrence graph from string window coverage.

    Parameters
    ----------
    string_windows:
        Mapping from string to the iterable of manifold window
        identifiers that covered at least one occurrence of the string.
    cooccurrence_threshold:
        Minimum shared window count for an edge to be included.

    Returns
    -------
    Graph
        A NetworkX graph if the `networkx` package is installed;
        otherwise a simple adjacency dictionary.
    """
    # Convert window iterables into sets for efficient intersection.
    window_sets: Dict[str, Set[int]] = {
        string: set(windows)
        for string, windows in string_windows.items()
    }
    nodes = list(window_sets.keys())
    if nx is not None:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for u, v in combinations(nodes, 2):
            weight = len(window_sets[u] & window_sets[v])
            if weight >= cooccurrence_threshold:
                G.add_edge(u, v, weight=weight)
        return G
    else:
        # Simple adjacency list representation
        adj: Dict[str, Set[str]] = defaultdict(set)
        for u, v in combinations(nodes, 2):
            weight = len(window_sets[u] & window_sets[v])
            if weight >= cooccurrence_threshold:
                adj[u].add(v)
                adj[v].add(u)
        # Ensure isolated nodes are present
        for node in nodes:
            adj.setdefault(node, set())
        return adj


def detect_themes(graph: "Graph") -> List[Set[str]]:
    """Detect communities/themes in the co‑occurrence graph.

    If NetworkX is available, this function uses the greedy modularity
    community algorithm.  Otherwise, it returns a single community
    containing all nodes.

    Returns
    -------
    List[Set[str]]
        A list of sets, each set containing strings belonging to a
        theme.
    """
    if nx is not None and isinstance(graph, nx.Graph):
        try:
            from networkx.algorithms.community import greedy_modularity_communities
        except Exception:
            # Fallback: single community
            return [set(graph.nodes())]
        communities = list(greedy_modularity_communities(graph))
        return [set(c) for c in communities]
    else:
        # Without networkx, treat each connected component in the
        # adjacency dict as a theme.
        if isinstance(graph, dict):
            visited: Set[str] = set()
            components: List[Set[str]] = []
            for node in graph.keys():
                if node in visited:
                    continue
                stack = [node]
                component: Set[str] = set()
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    component.add(current)
                    stack.extend(graph.get(current, []))
                components.append(component)
            return components if components else [set(graph.keys())]
        return []


def compute_graph_metrics(graph: "Graph") -> Dict[str, Dict[str, float]]:
    """Compute simple centrality measures for each node in the graph.

    The returned dictionary can be passed to the connector score.
    If networkx is not available, this function returns zeros for all
    metrics.
    """
    metrics: Dict[str, Dict[str, float]] = {}
    if nx is not None and isinstance(graph, nx.Graph):
        # Normalised betweenness centrality
        bet = nx.betweenness_centrality(graph, normalised=True)
        # Use bridging centrality approximation: edges bridging clusters have
        # high betweenness; reuse betweenness for bridging for now.
        bridging = bet
        # Compute degree entropy for neighbours of each node
        for node in graph.nodes():
            neighbours = list(graph.neighbors(node))
            # Compute theme entropy of neighbours: uniform if all
            # neighbours belong to different communities.  Since we
            # don’t have community assignments here, approximate with
            # inverse degree
            degs = [graph.degree(v) for v in neighbours]
            total = sum(degs)
            if not neighbours or total == 0:
                entropy = 0.0
            else:
                entropy = 0.0
                for d in degs:
                    p = d / total
                    if p > 0:
                        entropy -= p * math.log2(p)
                # Normalise by log2(len(neighbours))
                if len(neighbours) > 1:
                    entropy /= math.log2(len(neighbours))
                else:
                    entropy = 0.0
            metrics[node] = {
                "betweenness": bet.get(node, 0.0),
                "bridging": bridging.get(node, 0.0),
                "theme_entropy_neighbors": entropy,
                "redundant_degree": graph.degree(node) / max(1, len(graph.nodes()) - 1),
            }
        return metrics
    else:
        # Fallback: return zeros based on adjacency dictionary
        if isinstance(graph, dict):
            for node, neighbours in graph.items():
                degree = len(neighbours)
                metrics[node] = {
                    "betweenness": 0.0,
                    "bridging": 0.0,
                    "theme_entropy_neighbors": 0.0,
                    "redundant_degree": degree / max(1, len(graph) - 1),
                }
        return metrics
