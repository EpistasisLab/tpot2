import tpot2
import numpy as np
from ..estimator import TPOTEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tpot2.parent_selectors import survival_select_NSGA2, TournamentSelection_Dominated
#TODO These do not follow sklearn conventions of __init__

class TPOTRegressor(TPOTEstimator):
    def __init__(       self,
                        scorers=['neg_mean_squared_error'], 
                        scorers_weights=[1],
                        classification=False,
                        cv=5,
                        other_objective_functions=[tpot2.estimator_objective_functions.average_path_length_objective], #tpot2.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights=[-1],
                        objective_function_names=None,
                        bigger_is_better=True,
                        max_size=np.inf, 
                        linear_pipeline=False,
                        root_config_dict='Auto',
                        inner_config_dict=["selectors", "transformers"],
                        leaf_config_dict=None,                        
                        cross_val_predict_cv=0,
                        categorical_features = None,
                        subsets=None,
                        memory=None,
                        preprocessing=False,  
                        validation_strategy="none",
                        validation_fraction=.2,
                        population_size=50,
                        initial_population_size=None,
                        population_scaling=.5, 
                        generations_until_end_population=1,  
                        generations=50,
                        early_stop=None,
                        scorers_early_stop_tol=0.001,
                        other_objectives_early_stop_tol=None,
                        max_time_seconds=float('inf'), 
                        max_eval_time_seconds=60*10, 
                        n_jobs=1,
                        memory_limit="4GB",
                        client=None,
                        survival_percentage=1,
                        crossover_probability=.2,
                        mutate_probability=.7,
                        mutate_then_crossover_probability=.05,
                        crossover_then_mutate_probability=.05,
                        survival_selector=survival_select_NSGA2,
                        parent_selector=TournamentSelection_Dominated,
                        budget_range=None,
                        budget_scaling=.5,
                        generations_until_end_budget=1,  
                        stepwise_steps=5,
                        threshold_evaluation_early_stop=None, 
                        threshold_evaluation_scaling=.5,
                        min_history_threshold=20,
                        selection_evaluation_early_stop=None, 
                        selection_evaluation_scaling=.5, 
                        n_initial_optimizations=0,
                        optimization_cv=3,
                        max_optimize_time_seconds=60*20,
                        optimization_steps=10,
                        warm_start=False,
                        subset_column=None,
                        evolver="nsga2",
                        verbose=0,
                        periodic_checkpoint_folder=None, 
                        callback: tpot2.CallBackInterface=None,
                        processes = True,
        ):
        """
        See TPOTEstimator for documentation
        """
        super(TPOTRegressor,self).__init__(
            scorers=scorers, 
            scorers_weights=scorers_weights,
            classification=classification,
            cv=cv,
            other_objective_functions=other_objective_functions, #tpot2.estimator_objective_functions.number_of_nodes_objective],
            other_objective_functions_weights=other_objective_functions_weights,
            objective_function_names=objective_function_names,
            bigger_is_better=bigger_is_better,
            max_size=max_size, 
            linear_pipeline=linear_pipeline,
            root_config_dict=root_config_dict,
            inner_config_dict=inner_config_dict,
            leaf_config_dict=leaf_config_dict,                        
            cross_val_predict_cv=cross_val_predict_cv,
            categorical_features = categorical_features,
            subsets=subsets,
            memory=memory,
            preprocessing=preprocessing,  
            validation_strategy=validation_strategy,
            validation_fraction=validation_fraction,
            population_size=population_size,
            initial_population_size=initial_population_size,
            population_scaling=population_scaling, 
            generations_until_end_population=generations_until_end_population,  
            generations=generations,
            early_stop=early_stop,
            scorers_early_stop_tol=scorers_early_stop_tol,
            other_objectives_early_stop_tol=other_objectives_early_stop_tol,
            max_time_seconds=max_time_seconds, 
            max_eval_time_seconds=max_eval_time_seconds, 
            n_jobs=n_jobs,
            memory_limit=memory_limit,
            client=client,
            survival_percentage=survival_percentage,
            crossover_probability=crossover_probability,
            mutate_probability=mutate_probability,
            mutate_then_crossover_probability=mutate_then_crossover_probability,
            crossover_then_mutate_probability=crossover_then_mutate_probability,
            survival_selector=survival_selector,
            parent_selector=parent_selector,
            budget_range=budget_range,
            budget_scaling=budget_scaling,
            generations_until_end_budget=generations_until_end_budget,  
            stepwise_steps=stepwise_steps,
            threshold_evaluation_early_stop=threshold_evaluation_early_stop, 
            threshold_evaluation_scaling=threshold_evaluation_scaling,
            min_history_threshold=min_history_threshold,
            selection_evaluation_early_stop=selection_evaluation_early_stop, 
            selection_evaluation_scaling=selection_evaluation_scaling, 
            n_initial_optimizations=n_initial_optimizations,
            optimization_cv=optimization_cv,
            max_optimize_time_seconds=max_optimize_time_seconds,
            optimization_steps=optimization_steps,
            warm_start=warm_start,
            subset_column=subset_column,
            evolver=evolver,
            verbose=verbose,
            periodic_checkpoint_folder=periodic_checkpoint_folder, 
            callback=callback,
            processes=processes,
)


class TPOTClassifier(TPOTEstimator):
    def __init__(       self,
                        scorers=['roc_auc_ovr'], 
                        scorers_weights=[1],
                        classification=True,
                        cv=5,
                        other_objective_functions=[tpot2.estimator_objective_functions.average_path_length_objective], #tpot2.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights=[-1],
                        objective_function_names=None,
                        bigger_is_better=True,
                        
                        max_size=np.inf, 
                        linear_pipeline=False,
                        root_config_dict='Auto',
                        inner_config_dict=["selectors", "transformers"],
                        leaf_config_dict=None,                        
                        cross_val_predict_cv=0,
                        categorical_features = None,
                        subsets=None,
                        memory=None,
                        preprocessing=False,  
                        validation_strategy="none",
                        validation_fraction=.2,
                        population_size=50,
                        initial_population_size=None,
                        population_scaling=.5, 
                        generations_until_end_population=1,  
                        generations=50,
                        early_stop=None,
                        scorers_early_stop_tol=0.001,
                        other_objectives_early_stop_tol=None,
                        max_time_seconds=float('inf'), 
                        max_eval_time_seconds=60*10, 
                        n_jobs=1,
                        memory_limit="4GB",
                        client=None,
                        survival_percentage=1,
                        crossover_probability=.2,
                        mutate_probability=.7,
                        mutate_then_crossover_probability=.05,
                        crossover_then_mutate_probability=.05,
                        survival_selector=survival_select_NSGA2,
                        parent_selector=TournamentSelection_Dominated,
                        budget_range=None,
                        budget_scaling=.5,
                        generations_until_end_budget=1,  
                        stepwise_steps=5,
                        threshold_evaluation_early_stop=None, 
                        threshold_evaluation_scaling=.5,
                        min_history_threshold=20,
                        selection_evaluation_early_stop=None, 
                        selection_evaluation_scaling=.5, 
                        n_initial_optimizations=0,
                        optimization_cv=3,
                        max_optimize_time_seconds=60*20,
                        optimization_steps=10,
                        warm_start=False,
                        subset_column=None,
                        evolver="nsga2",
                        verbose=0,
                        periodic_checkpoint_folder=None, 
                        callback: tpot2.CallBackInterface=None,
                        processes = True,
        ):
        """
        See TPOTEstimator for documentation
        """
        super(TPOTClassifier,self).__init__(
                        scorers=scorers, 
            scorers_weights=scorers_weights,
            classification=classification,
            cv=cv,
            other_objective_functions=other_objective_functions, #tpot2.estimator_objective_functions.number_of_nodes_objective],
            other_objective_functions_weights=other_objective_functions_weights,
            objective_function_names=objective_function_names,
            bigger_is_better=bigger_is_better,
           
            max_size=max_size, 
            linear_pipeline=linear_pipeline,
            root_config_dict=root_config_dict,
            inner_config_dict=inner_config_dict,
            leaf_config_dict=leaf_config_dict,                        
            cross_val_predict_cv=cross_val_predict_cv,
            categorical_features = categorical_features,
            subsets=subsets,
            memory=memory,
            preprocessing=preprocessing,  
            validation_strategy=validation_strategy,
            validation_fraction=validation_fraction,
            population_size=population_size,
            initial_population_size=initial_population_size,
            population_scaling=population_scaling, 
            generations_until_end_population=generations_until_end_population,  
            generations=generations,
            early_stop=early_stop,
            scorers_early_stop_tol=scorers_early_stop_tol,
            other_objectives_early_stop_tol=other_objectives_early_stop_tol,
            max_time_seconds=max_time_seconds, 
            max_eval_time_seconds=max_eval_time_seconds, 
            n_jobs=n_jobs,
            memory_limit=memory_limit,
            client=client,
            survival_percentage=survival_percentage,
            crossover_probability=crossover_probability,
            mutate_probability=mutate_probability,
            mutate_then_crossover_probability=mutate_then_crossover_probability,
            crossover_then_mutate_probability=crossover_then_mutate_probability,
            survival_selector=survival_selector,
            parent_selector=parent_selector,
            budget_range=budget_range,
            budget_scaling=budget_scaling,
            generations_until_end_budget=generations_until_end_budget,  
            stepwise_steps=stepwise_steps,
            threshold_evaluation_early_stop=threshold_evaluation_early_stop, 
            threshold_evaluation_scaling=threshold_evaluation_scaling,
            min_history_threshold=min_history_threshold,
            selection_evaluation_early_stop=selection_evaluation_early_stop, 
            selection_evaluation_scaling=selection_evaluation_scaling, 
            n_initial_optimizations=n_initial_optimizations,
            optimization_cv=optimization_cv,
            max_optimize_time_seconds=max_optimize_time_seconds,
            optimization_steps=optimization_steps,
            warm_start=warm_start,
            subset_column=subset_column,
            evolver=evolver,
            verbose=verbose,
            periodic_checkpoint_folder=periodic_checkpoint_folder, 
            callback=callback,
            processes = processes,
        )


    def predict(self, X, **predict_params):
        check_is_fitted(self)
        #X=check_array(X)
        return self.fitted_pipeline_.predict(X,**predict_params)