## What does this PR do?

Add the new feature of allowing users to specify customized initial pipeline population for TPOT2.

## Where should the reviewer start?

- tpot2/tests/test_customized_iniPop.py
- tpot2/config/get_configspace.py
- tpot2/evolvers/base_evolver.py
- tpot2/tpot_estimator/estimator.py

## How should this PR be tested?

The test code is at tpot2/tests/test_customized_iniPop.py:

**pytest test_customized_iniPop.py**

```
import pytest


@pytest.fixture
def test_customized_iniPop():
    import tpot2
    import sklearn
    import sklearn.datasets

    scorer = sklearn.metrics.get_scorer('roc_auc_ovo')

    X, y = sklearn.datasets.load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.75, test_size=0.25)

    from tpot2.config.get_configspace import set_node
    from tpot2.search_spaces.pipelines.union import UnionPipeline
    from tpot2.search_spaces.pipelines.choice import ChoicePipeline
    from tpot2.search_spaces.pipelines.sequential import SequentialPipeline
    from tpot2.config.get_configspace import get_search_space

    scalers = set_node("MinMaxScaler", {})
    selectors = set_node("SelectFwe", {'alpha': 0.0002381268562})
    transformers_layer =UnionPipeline([
                            ChoicePipeline([
                                set_node("SkipTransformer", {})
                            ]),
                            get_search_space("Passthrough",)
                            ]
                        )

    inner_estimators_layer = UnionPipeline([
                                get_search_space("Passthrough",)]
                            )
    estimators = set_node("HistGradientBoostingClassifier", 
                        {'early_stop': 'valid', 
                        'l2_regularization': 0.0011074158219, 
                        'learning_rate': 0.0050792320068, 
                        'max_depth': None, 
                        'max_features': 0.3430178535213, 
                        'max_leaf_nodes': 237, 
                        'min_samples_leaf': 63, 
                        'tol': 0.0001, 
                        'n_iter_no_change': 14, 
                        'validation_fraction': 0.2343285974496})

    pipeline = SequentialPipeline(search_spaces=[
                                        scalers,
                                        selectors, 
                                        transformers_layer,
                                        inner_estimators_layer,
                                        estimators,
                                        ])
    ind = pipeline.generate()

    est = tpot2.TPOTClassifier(search_space="linear", n_jobs=40, verbose=5, generations=1, population_size=5, customized_initial_population=[ind])

    est.fit(X_train, y_train)

    print(str(est.fitted_pipeline_))

    print(scorer(est, X_test, y_test))
```

## Any background context you want to provide?

Under this version, users can specify well-defined initial pipeline population in SequentialPipeline type pipeline. This update has the potential to enhance the algorithm's performance and reduce evolutionary time.

Several Tips:

1. These SequentialPipeline pipelines can be obtained:

Referencing the examples in customized_initial_population.py and modifying them according to TPOT2's config_dict.

2. We consider the relationship between #customized initial pipelines and #population as follows:

```
init_population_size = len(customized_initial_population)
if self.cur_population_size <= init_population_size:
    initial_population = customized_initial_population[:self.cur_population_size]
else:
    initial_population = [next(self.individual_generator) for _ in range(self.cur_population_size - init_population_size)]
    initial_population = customized_initial_population + initial_population
```
3. The current version is only applicable to solve the problem where search_spaces is linear and the initialized pipeline is of type SequentialPipeline. We will continue to refine the scenario where search_spaces is graph and the pipeline is of type GraphPipeline in the near future if you think our approach is appropriate.


## What are the relevant issues?

[issue-61](https://github.com/EpistasisLab/tpot2/issues/61)

## Main Contributors

@peiyanpan @t-harden