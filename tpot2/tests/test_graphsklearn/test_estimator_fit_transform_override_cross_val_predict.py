from tpot2.graphsklearn import estimator_fit_transform_override_cross_val_predict
import sklearn
import pytest

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# @pytest.mark.skip(reason="test method name first")
class TestEstimatorFitTransformOverrideCrossValPredict:

    # Test when 'cv' greater than 1 and 'subset_indexes' is None.
    def test_estimator_fit_transform_override_cross_val_predict_cv_gt_1_subset_indexes_none(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3

        estimator = LinearRegression()
        preds, estimator = estimator_fit_transform_override_cross_val_predict(estimator, X, y, cv=3, method='predict')

        assert isinstance(preds, np.ndarray), "Expected preds to be an np.ndarray."
        assert preds.shape == (4,), "Expected preds to have shape (4,)."
        assert isinstance(estimator, LinearRegression), "Expected estimator to be an instance of LinearRegression."

    # Test when 'cv' is greater than 1 and 'subset_indexes' is not None.
    def test_estimator_fit_transform_override_cross_val_predict_cv_gt_1_subset_indexes_not_none(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3

        estimator = LinearRegression()
        subset_indexes = [0, 1]
        preds, estimator = estimator_fit_transform_override_cross_val_predict(estimator, X, y, cv=3, method='predict', subset_indexes=subset_indexes)

        assert isinstance(preds, np.ndarray), "Expected preds to be an np.ndarray."
        assert preds.shape == (4,), "Expected preds to have shape (4,)."
        assert isinstance(estimator, LinearRegression), "Expected estimator to be an instance of LinearRegression."

        
    # Test when 'cv' is 1 and 'subset_indexes' is None.
    def test_estimator_fit_transform_override_cross_val_predict_cv_1_subset_indexes_none(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3

        estimator = LinearRegression()
        preds, estimator = estimator_fit_transform_override_cross_val_predict(estimator, X, y, cv=1, method='predict')

        assert isinstance(preds, np.ndarray), "Expected preds to be an np.ndarray."
        assert preds.shape == (4,), "Expected preds to have shape (4,)."
        assert isinstance(estimator, LinearRegression), "Expected estimator to be an instance of LinearRegression."

    # Test when 'cv' is 1 and 'subset_indexes' is not None.
    def test_estimator_fit_transform_override_cross_val_predict_cv_1_subset_indexes_not_none(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3

        estimator = LinearRegression()
        subset_indexes = [0, 1]
        preds, estimator = estimator_fit_transform_override_cross_val_predict(estimator, X, y, cv=1, method='predict', subset_indexes=subset_indexes)

        assert isinstance(preds, np.ndarray), "Expected preds to be an np.ndarray."
        assert preds.shape == (4,), "Expected preds to have shape (4,)."
        assert isinstance(estimator, LinearRegression), "Expected estimator to be an instance of LinearRegression."
        
    # Test with 'cv' greater than 1 and 'subset_indexes' is None and 'method' is 'transform'.
    def test_estimator_fit_transform_override_cross_val_predict_transform(self):
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
        y = np.array([0, 0, 1, 1])

        estimator = StandardScaler()
        preds, estimator = estimator_fit_transform_override_cross_val_predict(estimator, X, y, cv=3, method='transform')

        assert isinstance(preds, np.ndarray), "Expected preds to be an np.ndarray."
        assert preds.shape == (4, 2), "Expected preds to have shape (4, 2)."
        assert isinstance(estimator, StandardScaler), "Expected estimator to be an instance of StandardScaler."
            
    # Test when 'cv' is greater than 1 and 'subset_indexes' is not None and 'method' is 'transform'.
    def test_estimator_fit_transform_override_cross_val_predict_transform_subset(self):
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
        y = np.array([0, 0, 1, 1])

        estimator = StandardScaler()
        subset_indexes = [0, 1]
        preds, estimator = estimator_fit_transform_override_cross_val_predict(estimator, X, y, cv=3, method='transform', subset_indexes=subset_indexes)

        assert isinstance(preds, np.ndarray), "Expected preds to be an np.ndarray."
        assert preds.shape == (4, 2), "Expected preds to have shape (4, 2)."
        assert isinstance(estimator, StandardScaler), "Expected estimator to be an instance of StandardScaler."
        
    # Test when 'cv' is 1, 'subset_indexes' is None, and 'method' is 'transform'.
    def test_estimator_fit_transform_override_cross_val_predict_transform_cv_1(self):
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
        y = np.array([0, 0, 1, 1])

        estimator = StandardScaler()
        preds, estimator = estimator_fit_transform_override_cross_val_predict(estimator, X, y, cv=1, method='transform')

        assert isinstance(preds, np.ndarray), "Expected preds to be an np.ndarray."
        assert preds.shape == (4, 2), "Expected preds to have shape (4, 2)."
        assert isinstance(estimator, StandardScaler), "Expected estimator to be an instance of StandardScaler."
        
    # Test when 'cv' is 1, 'subset_indexes' is not None, and 'method' is 'transform'.
    def test_estimator_fit_transform_override_cross_val_predict_transform_cv_1_subset(self):
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
        y = np.array([0, 0, 1, 1])

        estimator = StandardScaler()
        subset_indexes = [0, 1]
        preds, estimator = estimator_fit_transform_override_cross_val_predict(estimator, X, y, cv=1, method='transform', subset_indexes=subset_indexes)

        assert isinstance(preds, np.ndarray), "Expected preds to be an np.ndarray."
        assert preds.shape == (4, 2), "Expected preds to have shape (4, 2)."
        assert isinstance(estimator, StandardScaler), "Expected estimator to be an instance of StandardScaler."

    # Tests that the function correctly handles multiple roots. 
    def test_estimator_fit_transform_override_cross_val_predict_multiple_roots(self):
        # Other possible issue: multiple roots
        X = [[0, 1], [2, 3], [4, 5], [6, 7]]
        y = [0, 1, 0, 1]
        estimator = sklearn.linear_model.LogisticRegression()
        preds, fitted_estimator = estimator_fit_transform_override_cross_val_predict(estimator, X, y, cv=2, method='auto')
        assert len(preds) == len(X)
        assert isinstance(fitted_estimator, sklearn.linear_model.LogisticRegression)