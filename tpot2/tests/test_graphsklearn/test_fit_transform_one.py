from tpot2.graphsklearn import _fit_transform_one
import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

class TestFitTransformOne:

    # Tests that the function can correctly fit and transform data using a model with a fit_transform method. 
    def test_fit_transform_one_with_fit_transform_method(self):
        X = [[0, 15], [1, -10]]
        y = [0, 1]
        model = StandardScaler()
        res, _ = _fit_transform_one(model, X, y)
        assert res.tolist() == [[-1.0, 1.0], [1.0, -1.0]]

    # Tests that the function can handle subset_indexes argument None.
    def test_fit_transform_one_with_subset_indexes_none(self):
        # Edge case test
        X = [[0, 15], [1, -10]]
        y = [0, 1]
        model = StandardScaler()
        res, _ = _fit_transform_one(model, X, y, subset_indexes=None)
        assert res.tolist() == [[-1.0, 1.0], [1.0, -1.0]]

    # Tests that the function can correctly fit and transform data using a model with neither fit_transform nor transform methods.  
    def test_fit_transform_one_with_no_fit_transform_or_transform_methods(self):
        X = [[1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0]
        model = KMeans(n_clusters=2)
        res, fitted_model = _fit_transform_one(model, X, y)
        assert res.shape == (3, 2)
        assert isinstance(fitted_model, KMeans)
