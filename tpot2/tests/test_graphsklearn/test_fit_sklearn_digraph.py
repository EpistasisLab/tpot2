from tpot2.graphsklearn import fit_sklearn_digraph
import pytest
import numpy as np
import networkx as nx
import sklearn

class TestFitSklearnDigraph:
    # Tests that the function correctly handles a graph with no nodes and a graph with cycles. 
    def test_edge_case(self):
        # Edge case test case
        # Test with an empty graph
        graph = nx.DiGraph()

        # Generate some sample data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([1, 2, 3])

        # Call the function
        fit_sklearn_digraph(graph, X, y)

        # Check that the function does not raise any errors

        # Test with a graph with cycles
        graph = nx.DiGraph()
        transformer1 = sklearn.preprocessing.StandardScaler()
        transformer2 = sklearn.preprocessing.MinMaxScaler()
        estimator = sklearn.linear_model.LinearRegression()
        graph.add_node(0, instance=transformer1)
        graph.add_node(1, instance=transformer2)
        graph.add_node(2, instance=estimator)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)

        # Call the function
        with pytest.raises(nx.NetworkXUnfeasible):
            fit_sklearn_digraph(graph, X, y)
