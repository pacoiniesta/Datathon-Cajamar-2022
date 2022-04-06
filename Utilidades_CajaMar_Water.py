################################################################################
#                        skforecast.model_selection                            #
#                                                                              #
# This work by Joaquin Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8
# Modificado por Felipe Higón Martínez, Josep Año Gosp y Francisco José Iniesta
# Función grid_search_forecaster se comentarizan los prints y se eliminan los tqdm para agilizar el grid


from typing import Union, Tuple, Optional, Any
import numpy as np
import pandas as pd
import warnings
import logging
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid

from datetime import timedelta
from datetime import date

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
import math
from joblib import dump, load
import pickle

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


def time_series_splitter(
    y: Union[np.ndarray, pd.Series],
    initial_train_size: int,
    steps: int,
    allow_incomplete_fold: bool=True,
    verbose: bool=True
) -> Union[np.ndarray, np.ndarray]:
    '''
    
    Split indices of a time series into multiple train-test pairs. The order of
    is maintained and the training set increases in each iteration.
    
    Parameters
    ----------        
    y : 1d numpy ndarray, pandas Series
        Training time series values. 
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    allow_incomplete_fold : bool, default `True`
        The last test set is allowed to be incomplete if it does not reach `steps`
        observations. Otherwise, the latest observations are discarded.
        
    verbose : bool, default `True`
        Print number of splits created.

    Yields
    ------
    train : 1d numpy ndarray
        Training indices.
        
    test : 1d numpy ndarray
        Test indices.
        
    '''
    
    if not isinstance(y, (np.ndarray, pd.Series)):

        raise Exception('`y` must be `1D np.ndarray` o `pd.Series`.')

    elif isinstance(y, np.ndarray) and y.ndim != 1:

        raise Exception(
            f"`y` must be `1D np.ndarray` o `pd.Series`, "
            f"got `np.ndarray` with {y.ndim} dimensions."
        )
        
    if initial_train_size > len(y):
        raise Exception(
            '`initial_train_size` must be smaller than length of `y`.'
            ' Try to reduce `initial_train_size` or `steps`.'
        )

    if isinstance(y, pd.Series):
        y = y.to_numpy().copy()
    
  
    folds = (len(y) - initial_train_size) // steps  + 1
    # +1 fold is needed to allow including the remainder in the last iteration.
    remainder = (len(y) - initial_train_size) % steps   
    
    if verbose:
        if folds == 1:
            print(f"Number of folds: {folds - 1}")
            print("Not enough observations in `y` to create even a complete fold."
                  " Try to reduce `initial_train_size` or `steps`."
            )

        elif remainder == 0:
            print(f"Number of folds: {folds - 1}")

        elif remainder != 0 and allow_incomplete_fold:
            print(f"Number of folds: {folds}")
            print(
                f"Since `allow_incomplete_fold=True`, "
                f"last fold only includes {remainder} observations instead of {steps}."
            )
            print(
                'Incomplete folds with few observations could overestimate or ',
                'underestimate validation metrics.'
            )
        elif remainder != 0 and not allow_incomplete_fold:
            print(f"Number of folds: {folds - 1}")
            print(
                f"Since `allow_incomplete_fold=False`, "
                f"last {remainder} observations are descarted."
            )

    if folds == 1:
        # There are no observations to create even a complete fold
        return []
    
    for i in range(folds):
          
        if i < folds - 1:
            train_end     = initial_train_size + i * steps    
            train_indices = range(train_end)
            test_indices  = range(train_end, train_end + steps)
            
        else:
            if remainder != 0 and allow_incomplete_fold:
                train_end     = initial_train_size + i * steps  
                train_indices = range(train_end)
                test_indices  = range(train_end, len(y))
            else:
                break
        
        yield train_indices, test_indices
        

def _get_metric(metric:str) -> callable:
    '''
    Get the corresponding scikitlearn function to calculate the metric.
    
    Parameters
    ----------
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
    
    Returns 
    -------
    metric : callable
        scikitlearn function to calculate the desired metric.
    '''
    
    if metric not in ['mean_squared_error', 'mean_absolute_error',
                      'mean_absolute_percentage_error']:
        
        raise Exception(
            f"Allowed metrics are: 'mean_squared_error', 'mean_absolute_error' and "
            f"'mean_absolute_percentage_error'. Got {metric}."
        )
    
    metrics = {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error
    }

    metric = metrics[metric]
    
    return metric
    

def cv_forecaster(
    forecaster,
    y: pd.Series,
    initial_train_size: int,
    steps: int,
    metric: Union[str, callable],
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    allow_incomplete_fold: bool=True,
    verbose: bool=True
) -> Tuple[np.array, pd.DataFrame]:
    '''
    Cross-validation of forecaster. The order of data is maintained and the
    training set increases in each iteration.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forecaster model.
        
    y : pandas Series
        Training time series values. 
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        It callable:
            Function with arguments y_true, y_pred that returns a float.
        
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
            
    allow_incomplete_fold : bool, default `True`
        The last test partition is allowed to be incomplete if it does not reach `steps`
        observations. Otherwise, the latest observations are discarded.
            
    verbose : bool, default `True`
        Print number of folds used for cross validation.

    Returns 
    -------
    cv_metrics: 1d numpy ndarray
        Value of the metric for each fold.

    cv_predictions: pandas DataFrame
        Predictions.

    '''

    if initial_train_size > len(y):
        raise Exception(
            '`initial_train_size` must be smaller than length of `y`.'
        )
        
    if initial_train_size is not None and initial_train_size < forecaster.window_size:
        raise Exception(
            f"`initial_train_size` must be greater than "
            f"forecaster's window_size ({forecaster.window_size})."
        )
        
    forecaster = deepcopy(forecaster)
    if isinstance(metric, str):
        metric = _get_metric(metric=metric)
    
    splits = time_series_splitter(
                y                     = y,
                initial_train_size    = initial_train_size,
                steps                 = steps,
                allow_incomplete_fold = allow_incomplete_fold,
                verbose               = verbose
             )

    cv_predictions = []
    cv_metrics = []
    
    for train_index, test_index in splits:
        
        if exog is None:
            forecaster.fit(y=y.iloc[train_index])      
            pred = forecaster.predict(steps=len(test_index))
            
        else:
            forecaster.fit(y=y.iloc[train_index], exog=exog.iloc[train_index,])      
            pred = forecaster.predict(steps=len(test_index), exog=exog.iloc[test_index])
               
        metric_value = metric(
                            y_true = y.iloc[test_index],
                            y_pred = pred
                       )
        
        cv_predictions.append(pred)
        cv_metrics.append(metric_value)
            
    cv_predictions = pd.concat(cv_predictions)
    cv_predictions = pd.DataFrame(cv_predictions)
    cv_metrics = np.array(cv_metrics)
    
    return cv_metrics, cv_predictions


def _backtesting_forecaster_refit(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=False,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False,
    set_out_sample_residuals: Any='deprecated'
) -> Tuple[float, pd.DataFrame]:
    '''
    Backtesting of forecaster model with a re-fitting strategy. A copy of the  
    original forecaster is created so it is not modified during the process.
    
    In each iteration:
        - Fit forecaster with the training set.
        - A number of `steps` ahead are predicted.
        - The training set increases with `steps` observations.
        - The model is re-fitted using the new training set.

    In order to apply backtesting with re-fit, an initial training set must be
    available, otherwise it would not be possible to increase the training set 
    after each iteration. `initial_train_size` must be provided.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forecaster model.
        
    y : pandas Series
        Training time series values. 
    
    initial_train_size: int
        Number of samples in the initial train split. The backtest forecaster is
        trained using the first `initial_train_size` observations.
        
    fixed_train_size: bool, default `False`
        If True, train size doesn't increases but moves by `steps` in each iteration.
        
    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        It callable:
            Function with arguments y_true, y_pred that returns a float.
        
    exog :panda Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    interval: list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot: int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    random_state: int, default 123
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.

    in_sample_residuals: bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals. If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.

    set_out_sample_residuals: 'deprecated'
        Deprecated since version 0.4.2, will be removed on version 0.5.0.
            
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metric_value: float
        Value of the metric.

    backtest_predictions: pandas Dataframe
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.

    '''

    forecaster = deepcopy(forecaster)
    if isinstance(metric, str):
        metric = _get_metric(metric=metric)
    backtest_predictions = []
    
    folds = int(np.ceil((len(y) - initial_train_size) / steps))
    remainder = (len(y) - initial_train_size) % steps
    
    if verbose:
        print(f"Information of backtesting process")
        print(f"----------------------------------")
        print(f"Number of observations used for initial training: {initial_train_size}")
        print(f"Number of observations used for backtesting: {len(y) - initial_train_size}")
        print(f"    Number of folds: {folds}")
        print(f"    Number of steps per fold: {steps}")
        if remainder != 0:
            print(f"    Last fold only includes {remainder} observations.")
        print("")
        for i in range(folds):
            if fixed_train_size:
                # The train size doesn't increase but moves by `steps` in each iteration.
                train_idx_start = i * steps
                train_idx_end = initial_train_size + i * steps
            else:
                # The train size increases by `steps` in each iteration.
                train_idx_start = 0
                train_idx_end = initial_train_size + i * steps
            print(f"Data partition in fold: {i}")
            if i < folds - 1:
                print(f"    Training:   {y.index[train_idx_start]} -- {y.index[train_idx_end - 1]}")
                print(f"    Validation: {y.index[train_idx_end]} -- {y.index[train_idx_end + steps - 1]}")
            else:
                print(f"    Training:   {y.index[train_idx_start]} -- {y.index[train_idx_end - 1]}")
                print(f"    Validation: {y.index[train_idx_end]} -- {y.index[-1]}")
        print("")
        
    if folds > 50:
        print(
            f"Forecaster will be fit {folds} times. This can take substantial amounts of time. "
            f"If not feasible, try with `refit = False`. \n"
        )

    for i in range(folds):
        # In each iteration (except the last one) the model is fitted before making predictions.
        if fixed_train_size:
            # The train size doesn't increases but moves by `steps` in each iteration.
            train_idx_start = i * steps
            train_idx_end = initial_train_size + i * steps
        else:
            # The train size increases by `steps` in each iteration.
            train_idx_start = 0
            train_idx_end = initial_train_size + i * steps
            
        if exog is not None:
            next_window_exog = exog.iloc[train_idx_end:train_idx_end + steps, ]

        if interval is None:

            if i < folds - 1:
                if exog is None:
                    forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                    pred = forecaster.predict(steps=steps)
                else:
                    forecaster.fit(
                        y = y.iloc[train_idx_start:train_idx_end], 
                        exog = exog.iloc[train_idx_start:train_idx_end, ]
                    )
                    pred = forecaster.predict(steps=steps, exog=next_window_exog)
            else:    
                if remainder == 0:
                    if exog is None:
                        forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict(steps=steps)
                    else:
                        forecaster.fit(
                            y = y.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict(steps=steps, exog=next_window_exog)
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict(steps=steps)
                    else:
                        forecaster.fit(
                            y = y.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict(steps=steps, exog=next_window_exog)
        else:

            if i < folds - 1:
                if exog is None:
                    forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                else:
                    forecaster.fit(
                        y = y.iloc[train_idx_start:train_idx_end], 
                        exog = exog.iloc[train_idx_start:train_idx_end, ]
                    )
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                exog         = next_window_exog,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )
            else:    
                if remainder == 0:
                    if exog is None:
                        forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                    else:
                        forecaster.fit(
                            y = y.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                exog         = next_window_exog,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        forecaster.fit(y=y.iloc[train_idx_start:train_idx_end])
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                    else:
                        forecaster.fit(
                            y = y.iloc[train_idx_start:train_idx_end], 
                            exog = exog.iloc[train_idx_start:train_idx_end, ]
                        )
                        pred = forecaster.predict_interval(
                                steps        = steps,
                                exog         = next_window_exog,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                           )

        backtest_predictions.append(pred)
    
    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)

    metric_value = metric(
                    y_true = y.iloc[initial_train_size: initial_train_size + len(backtest_predictions)],
                    y_pred = backtest_predictions['pred']
                   )

    return metric_value, backtest_predictions


def _backtesting_forecaster_no_refit(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: Optional[int]=None,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False,
    set_out_sample_residuals: Any='deprecated'
) -> Tuple[float, pd.DataFrame]:
    '''
    Backtesting of forecaster without iterative re-fitting. In each iteration,
    a number of `steps` are predicted. A copy of the  original forecaster is
    created so it is not modified during the process.

    If `forecaster` is already trained and `initial_train_size` is `None`,
    no initial train is done and all data is used to evaluate the model.
    However, the first `len(forecaster.last_window)` observations are needed
    to create the initial predictors, so no predictions are calculated for them.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forecaster model.
        
    y : pandas Series
        Training time series values. 
    
    initial_train_size: int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is already
        trained, no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.
        
    steps : int, None
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        It callable:
            Function with arguments y_true, y_pred that returns a float.
        
    exog :panda Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    interval: list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot: int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    random_state: int, default 123
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.

    in_sample_residuals: bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals.  If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.

    set_out_sample_residuals: 'deprecated'
        Deprecated since version 0.4.2, will be removed on version 0.5.0.
            
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metric_value: float
        Value of the metric.

    backtest_predictions: pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.

    '''

    forecaster = deepcopy(forecaster)
    if isinstance(metric, str):
        metric = _get_metric(metric=metric)
    backtest_predictions = []

    if initial_train_size is not None:
        if exog is None:
            forecaster.fit(y=y.iloc[:initial_train_size])      
        else:
            forecaster.fit(
                y = y.iloc[:initial_train_size],
                exog = exog.iloc[:initial_train_size, ]
            )
        window_size = forecaster.window_size
    else:
        # Although not used for training, first observations are needed to create
        # the initial predictors
        window_size = forecaster.window_size
        initial_train_size = window_size
    
    folds     = int(np.ceil((len(y) - initial_train_size) / steps))
    remainder = (len(y) - initial_train_size) % steps
    
    if verbose:
        print(f"Information of backtesting process")
        print(f"----------------------------------")
        print(f"Number of observations used for initial training or as initial window: {initial_train_size}")
        print(f"Number of observations used for backtesting: {len(y) - initial_train_size}")
        print(f"    Number of folds: {folds}")
        print(f"    Number of steps per fold: {steps}")
        if remainder != 0:
            print(f"    Last fold only includes {remainder} observations")
        print("")
        for i in range(folds):
            last_window_end = initial_train_size + i * steps
            print(f"Data partition in fold: {i}")
            if i < folds - 1:
                print(f"    Training:   {y.index[0]} -- {y.index[initial_train_size - 1]}")
                print(f"    Validation: {y.index[last_window_end]} -- {y.index[last_window_end + steps -1]}")
            else:
                print(f"    Training:   {y.index[0]} -- {y.index[initial_train_size - 1]}")
                print(f"    Validation: {y.index[last_window_end]} -- {y.index[-1]}")
        print("")

    for i in range(folds):
        # Since the model is only fitted with the initial_train_size, last_window
        # and next_window_exog must be updated to include the data needed to make
        # predictions.
        last_window_end   = initial_train_size + i * steps
        last_window_start = last_window_end - window_size 
        last_window_y     = y.iloc[last_window_start:last_window_end]
        if exog is not None:
            next_window_exog = exog.iloc[last_window_end:last_window_end + steps, ]
    
        if interval is None:  

            if i < folds - 1: 
                if exog is None:
                    pred = forecaster.predict(
                                steps       = steps,
                                last_window = last_window_y
                            )
                else:
                    pred = forecaster.predict(
                                steps       = steps,
                                last_window = last_window_y,
                                exog        = next_window_exog
                            )            
            else:    
                if remainder == 0:
                    if exog is None:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    last_window = last_window_y
                                )
                    else:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    last_window = last_window_y,
                                    exog        = next_window_exog
                                )
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    last_window = last_window_y
                                )
                    else:
                        pred = forecaster.predict(
                                    steps       = steps,
                                    last_window = last_window_y,
                                    exog        = next_window_exog
                                )
            
            backtest_predictions.append(pred)

        else:
            if i < folds - 1:
                if exog is None:
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                last_window  = last_window_y,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )
                else:
                    pred = forecaster.predict_interval(
                                steps        = steps,
                                last_window  = last_window_y,
                                exog         = next_window_exog,
                                interval     = interval,
                                n_boot       = n_boot,
                                random_state = random_state,
                                in_sample_residuals = in_sample_residuals
                            )            
            else:    
                if remainder == 0:
                    if exog is None:
                        pred = forecaster.predict_interval(
                                    steps        = steps,
                                    last_window  = last_window_y,
                                    interval     = interval,
                                    n_boot       = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                                )
                    else:
                        pred = forecaster.predict_interval(
                                    steps        = steps,
                                    last_window  = last_window_y,
                                    exog         = next_window_exog,
                                    interval     = interval,
                                    n_boot       = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                                )
                else:
                    # Only the remaining steps need to be predicted
                    steps = remainder
                    if exog is None:
                        pred = forecaster.predict_interval(
                                    steps        = steps,
                                    last_window  = last_window_y,
                                    interval     = interval,
                                    n_boot       = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                                )
                    else:
                        pred = forecaster.predict_interval(
                                    steps        = steps,
                                    last_window  = last_window_y,
                                    exog         = next_window_exog,
                                    interval     = interval,
                                    n_boot       = n_boot,
                                    random_state = random_state,
                                    in_sample_residuals = in_sample_residuals
                                )
            
            backtest_predictions.append(pred)

    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)

    metric_value = metric(
                    y_true = y.iloc[initial_train_size : initial_train_size + len(backtest_predictions)],
                    y_pred = backtest_predictions['pred']
                   )

    return metric_value, backtest_predictions


def backtesting_forecaster(
    forecaster,
    y: pd.Series,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: Optional[int],
    fixed_train_size: bool=False,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    refit: bool=False,
    interval: Optional[list]=None,
    n_boot: int=500,
    random_state: int=123,
    in_sample_residuals: bool=True,
    verbose: bool=False,
    set_out_sample_residuals: Any='deprecated'
) -> Tuple[float, pd.DataFrame]:
    '''
    Backtesting of forecaster model.

    If `refit` is False, the model is trained only once using the `initial_train_size`
    first observations. If `refit` is True, the model is trained in each iteration
    increasing the training set. A copy of the original forecaster is created so 
    it is not modified during the process.

    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forecaster model.
        
    y : pandas Series
        Training time series values. 
    
    initial_train_size: int, default `None`
        Number of samples in the initial train split. If `None` and `forecaster` is already 
        trained, no initial train is done and all data is used to evaluate the model. However, 
        the first `len(forecaster.last_window)` observations are needed to create the 
        initial predictors, so no predictions are calculated for them.

        `None` is only allowed when `refit` is False.
    
    fixed_train_size: bool, default `False`
        If True, train size doesn't increases but moves by `steps` in each iteration.
        
    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        It callable:
            Function with arguments y_true, y_pred that returns a float.
        
    exog :panda Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].

    refit: bool, default False
        Whether to re-fit the forecaster in each iteration.

    interval: list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. If `None`, no
        intervals are estimated. Only available for forecaster of type ForecasterAutoreg
        and ForecasterAutoregCustom.
            
    n_boot: int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.

    random_state: int, default 123
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.

    in_sample_residuals: bool, default `True`
        If `True`, residuals from the training data are used as proxy of
        prediction error to create prediction intervals.  If `False`, out_sample_residuals
        are used if they are already stored inside the forecaster.

    set_out_sample_residuals: 'deprecated'
        Deprecated since version 0.4.2, will be removed on version 0.5.0.
                  
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used for backtesting.

    Returns 
    -------
    metric_value: float
        Value of the metric.

    backtest_predictions: pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.
            column pred = predictions.
            column lower_bound = lower bound of the interval.
            column upper_bound = upper bound interval of the interval.

    '''

    if initial_train_size is not None and initial_train_size > len(y):
        raise Exception(
            'If used, `initial_train_size` must be smaller than length of `y`.'
        )
        
    if initial_train_size is not None and initial_train_size < forecaster.window_size:
        raise Exception(
            f"`initial_train_size` must be greater than "
            f"forecaster's window_size ({forecaster.window_size})."
        )

    if initial_train_size is None and not forecaster.fitted:
        raise Exception(
            '`forecaster` must be already trained if no `initial_train_size` is provided.'
        )

    if not isinstance(refit, bool):
        raise Exception(
            f'`refit` must be boolean: True, False.'
        )

    if initial_train_size is None and refit:
        raise Exception(
            f'`refit` is only allowed when there is a initial_train_size.'
        )

    if interval is not None and isinstance(forecaster, ForecasterAutoregMultiOutput):
        raise Exception(
            ('Interval prediction is only available when forecaster is of type '
            'ForecasterAutoreg or ForecasterAutoregCustom.')
        )

    if set_out_sample_residuals != 'deprecated':
        warnings.warn(
            ('`set_out_sample_residuals` is deprecated since version 0.4.2, '
            'will be removed on version 0.5.0.')
        )    
    
    if refit:
        metric_value, backtest_predictions = _backtesting_forecaster_refit(
            forecaster          = forecaster,
            y                   = y,
            steps               = steps,
            metric              = metric,
            initial_train_size  = initial_train_size,
            fixed_train_size    = fixed_train_size,
            exog                = exog,
            interval            = interval,
            n_boot              = n_boot,
            random_state        = random_state,
            in_sample_residuals = in_sample_residuals,
            verbose             = verbose
        )
    else:
        metric_value, backtest_predictions = _backtesting_forecaster_no_refit(
            forecaster          = forecaster,
            y                   = y,
            steps               = steps,
            metric              = metric,
            initial_train_size  = initial_train_size,
            exog                = exog,
            interval            = interval,
            n_boot              = n_boot,
            random_state        = random_state,
            in_sample_residuals = in_sample_residuals,
            verbose             = verbose
        )

    return metric_value, backtest_predictions


def grid_search_forecaster(
    forecaster,
    y: pd.Series,
    param_grid: dict,
    steps: int,
    metric: Union[str, callable],
    initial_train_size: int,
    fixed_train_size: bool=False,
    exog: Optional[Union[pd.Series, pd.DataFrame]]=None,
    lags_grid: Optional[list]=None,
    refit: bool=False,
    return_best: bool=True,
    verbose: bool=True
) -> pd.DataFrame:
    '''
    Exhaustive search over specified parameter values for a Forecaster object.
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregMultiOutput
        Forcaster model.
        
    y : pandas Series
        Training time series values. 
        
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.

    steps : int
        Number of steps to predict.
        
    metric : str, callable
        Metric used to quantify the goodness of fit of the model.
        
        If string:
            {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}

        It callable:
            Function with arguments y_true, y_pred that returns a float.

    initial_train_size: int 
        Number of samples in the initial train split.
 
    fixed_train_size: bool, default `False`
        If True, train size doesn't increases but moves by `steps` in each iteration.

    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
           
    lags_grid : list of int, lists, np.narray or range. 
        Lists of `lags` to try. Only used if forecaster is an instance of 
        `ForecasterAutoreg`.
        
    refit: bool, default False
        Whether to re-fit the forecaster in each iteration of backtesting.
        
    return_best : bool
        Refit the `forecaster` using the best found parameters on the whole data.
        
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.

    Returns 
    -------
    results: pandas DataFrame
        Metric value estimated for each combination of parameters.

    '''

    if isinstance(forecaster, ForecasterAutoregCustom):
        if lags_grid is not None:
            warnings.warn(
                '`lags_grid` ignored if forecaster is an instance of `ForecasterAutoregCustom`.'
            )
        lags_grid = ['custom predictors']
        
    elif lags_grid is None:
        lags_grid = [forecaster.lags]
   
    lags_list = []
    params_list = []
    metric_list = []
    
    param_grid =  list(ParameterGrid(param_grid))

    #print(
    #    f"Number of models compared: {len(param_grid)*len(lags_grid)}"
    #)

    #for lags in tqdm(lags_grid, desc='loop lags_grid', position=0, ncols=90):
    for lags in lags_grid:
        
        if isinstance(forecaster, (ForecasterAutoreg, ForecasterAutoregMultiOutput)):
            forecaster.set_lags(lags)
            lags = forecaster.lags.copy()
        
        #for params in tqdm(param_grid, desc='loop param_grid', position=1, leave=False, ncols=90):
        for params in param_grid:

            forecaster.set_params(**params)
            metrics = backtesting_forecaster(
                            forecaster         = forecaster,
                            y                  = y,
                            exog               = exog,
                            steps              = steps,
                            metric             = metric,
                            initial_train_size = initial_train_size,
                            fixed_train_size   = fixed_train_size,
                            refit              = refit,
                            interval           = None,
                            verbose            = verbose
                            )[0]

            lags_list.append(lags)
            params_list.append(params)
            metric_list.append(metrics)
            
    results = pd.DataFrame({
                'lags'  : lags_list,
                'params': params_list,
                'metric': metric_list})
    
    results = results.sort_values(by='metric', ascending=True)
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_lags = results['lags'].iloc[0]
        best_params = results['params'].iloc[0]
        best_metric = results['metric'].iloc[0]
        
        if isinstance(forecaster, (ForecasterAutoreg, ForecasterAutoregMultiOutput)):
            forecaster.set_lags(best_lags)
        forecaster.set_params(**best_params)
        forecaster.fit(y=y, exog=exog)
        
        #print(
        #    f"`Forecaster` refitted using the best-found lags and parameters, and the whole data set: \n"
        #    f"  Lags: {best_lags} \n"
        #    f"  Parameters: {best_params}\n"
        #    f"  Backtesting metric: {best_metric}\n"
        #)
            
    return results


    ################################################################################
#                Clases para Datathon Cajamar: Predicción consumo agua             #
#                                                                                  #
#          Felipe Higón Martínez, Josep Año Gosp y Francisco José Iniesta          #  
    ################################################################################

class Contador(object):
    '''
    Clase contador que realiza las siguientes funciones:
    
        - Pre-procesa los datos del contador.
        - Resample de la variable consumo en dias y semanas.
        - Aplica las medias moviles a la variable consumo.
        - Genera el modelo según los datos estadísticos.
    
    Parameters
    ----------
        
    datos_contador : Pandas DataFrame 
                     Datos con el formato del datathon del contador
                     ordenado por fecha de forma ascendente. 
    
    ID: int
        ID del contador.
    
    umbral: int
            Valor númerico que indica el número de dias de datos a partir
            del cual se utiliza modelo. Por debajo del umbral se utiliza
            la media como predicción.

    Modelo_Semanal: bool
                    True -> Se utlizan dos modelos uno para predecir dias
                    y otro para predecir semanas.
                    False -> Solo un modelo para predecir dias.

    Media_Movil: int
                 Valor númerico que indica el número de dias que se usa 
                 para el pre-procesado de la media móvil.

    Fecha_Fin: date
               Dia teórico hasta donde hay datos, en este caso (31-01-2020).

    Cargar_Modelo: bool
                   True -> Carga el modelo en la ruta "./modelos/Modelo_ID_"
                           + ID del contador.

    Outlier: int
             Valor númerico que indica las desviaciones tipicas a partir de las
             cuales se consideran outliers y se elimina el dato de consumo y consumo
             calculado a partir de la lectura.
             Default = 8
    
    dias_test: int
                   Valor númerico que indica los dias que se guardan para test
                   default: 14

    Returns 
    -------
    
    Las variables se almacen en la propia clase, por tanto, no devuelve ningún
    parámetro.

    '''
    def __init__(self, datos_contador, ID, umbral, Modelo_Semanal, Media_Movil, Fecha_Fin, Cargar_Modelo, Outlier=8, dias_test=14):
        self.datos_contador = datos_contador
        self.ID = ID
        self.umbral = umbral
        self.Modelo_Semanal = Modelo_Semanal
        self.Media_Movil = Media_Movil
        self.Fecha_Fin = Fecha_Fin
        self.Cargar_Modelo = Cargar_Modelo
        self.Outlier = Outlier
        self.dias_test = dias_test
        self.Preprocesado()
        self.SinDatos = False  # NO TIENE DATOS DATASET VACIO
        self.dias = 0
        self.RMSE_Train = 0
        self.RMSE_Test = 0
        self.datos_test = pd.DataFrame()
        self.datos_test_semanal = pd.DataFrame()
        self.Mejor_Parametro = {}
        self.Mejor_Lag = 14 
        self.DiffConsumo = 0
        self.LecturaErronea = False
        
        self.Predicciones = [self.ID,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        if not datos_contador.empty:
     
            self.__Comprobar_Consumo()

            # CONSUMO POR DIAS
            # RESAMPLE
            self.datos_consumo = pd.DataFrame(self.datos_contador['CONSUMO'].resample(rule='1D', closed='left', label ='left').sum())

            # MEDIAS MÓVILES
            if self.Media_Movil > 0:
                self.datos_consumo['CONSUMO_MED'] = self.datos_consumo['CONSUMO'].rolling(self.Media_Movil).mean()
                self.datos_consumo['CONSUMO_MED'].fillna(0,inplace=True)
            else:
                self.datos_consumo['CONSUMO_MED'] = self.datos_consumo['CONSUMO']
            
            self.datos_consumo_semanal = pd.DataFrame(self.__SemanaCompleta()['CONSUMO'].resample(rule='1W', closed='left', label ='left').sum())
            
            # COGER SOLO SEMANAS COMPLETAS PENDIENTE DE IMPLEMENTACIÓN
            #self.__SemanaCompleta()
            
            # ESTADISTICAS
            self.dias = len(self.datos_consumo.index)
            self.semanas = len(self.datos_consumo_semanal.index)
            
            self.fecha_inicio_contador= self.datos_consumo.index[0]
            self.fecha_fin_contador= self.datos_consumo.index[-1]
            
        else:
            self.datos_consumo = pd.DataFrame()
            self.SinDatos = True
            self.fecha_fin_contador = None

        # GENERACION DEL MODELO
        self.modelo()
    
    def __SemanaCompleta(self):
        # PARTIENDO DE LOS 7 DIAS INICIALES Y 7 DIAS FINALES DE datos_consumo
        # DEVUELVE datos_consumo ELIMINANDO LAS SEMANAS INCOMPLETAS

        index_inicio = 0
        index_fin = 0
        semana = 0

        # DATOS CONSUMO TAIL VER DIA SEMANA

        for fecha in self.datos_consumo.head(7).index:
            if semana == 0:
                semana = fecha.isocalendar()[1]
            elif semana != fecha.isocalendar()[1]:
                break
            index_inicio += 1
        semana = 0
        for fecha in self.datos_consumo.tail(7).index:
            if semana == 0:
                semana = fecha.isocalendar()[1]
            elif semana != fecha.isocalendar()[1]:
                index_fin = 7 - index_fin
                break
            index_fin += 1

        if index_fin == 0:
            return(self.datos_consumo[index_inicio:])
        else:
            return(self.datos_consumo[index_inicio:-index_fin])


    def __Comprobar_Consumo(self):
        # COMPRUEBA SI HAY MAS DE 1 LITRO DE DIFERENCIA ENTRE EL CONSUMO EL CALCULO DEL CONSUMO CON LA LECTURA DEL CONTADOR
        # EL VALOR ACUMULADO SE GUARDA EN DiffConsumo

        if not(self.datos_contador['CONSUMO_CAL'].equals(self.datos_contador['CONSUMO'])):
            for index, row in self.datos_contador.iterrows():
                if np.absolute(row['CONSUMO_CAL']) != row['CONSUMO'] and (np.absolute(row['CONSUMO_CAL'] - row['CONSUMO']) > 1):
                    #print(row['CONSUMO_CAL'] - row['CONSUMO'])
                    #print(index,row)
                    self.DiffConsumo += np.absolute(row['CONSUMO_CAL'] - row['CONSUMO'])
        
    def __Convertir_fecha(self, serie):
        fechas = []
        for año, mes, dia in serie.index:
            fechas.append(date(año,mes,dia))
        return fechas
    
    def Preprocesado(self,):
        # SAMPLETIME READINGINTEGER	READINGTHOUSANDTH	DELTAINTEGER	DELTATHOUSANDTH 

        # VALORES ENTEROS N/A INTERPOLAMOS
        self.datos_contador['READINGINTEGER'].interpolate(inplace=True)
        self.datos_contador['DELTAINTEGER'].interpolate(inplace=True)
        
        # VALORES DECIMALES N/A SUSTITUIMOS POR CERO
        self.datos_contador['READINGTHOUSANDTH'].fillna(0,inplace=True)
        self.datos_contador['DELTATHOUSANDTH'].fillna(0,inplace=True)
        
        # ELIMINAR VALORES NEGATIVOS
        indices = self.datos_contador[self.datos_contador['READINGINTEGER'] < 0].index
        self.datos_contador.drop(indices,inplace=True)
        indices = self.datos_contador[self.datos_contador['DELTAINTEGER'] < 0].index
        self.datos_contador.drop(indices,inplace=True)

        # CALCULO LECTURA CONTADOR Y CONSUMO JUNTANDO DECIMALES
        self.datos_contador = self.datos_contador.set_index('SAMPLETIME')
        self.datos_contador['LECTURA'] = self.datos_contador['READINGINTEGER'] + (self.datos_contador['READINGTHOUSANDTH'] / 100)
        self.datos_contador['CONSUMO'] = self.datos_contador['DELTAINTEGER'] + (self.datos_contador['DELTATHOUSANDTH'] / 100)

        # CALCULO DEL CONSUMO CON LA LECTURA
        self.datos_contador['CONSUMO_CAL'] = self.datos_contador['LECTURA'].diff(1).fillna(0)
        indices = self.datos_contador[self.datos_contador['CONSUMO_CAL'] < 0].index
        self.datos_contador.drop(indices,inplace=True)

        # NORMALIZACIÓN A DISTRIBUCIÓN NORMAL(0,1)
        self.datos_contador['CONSUMO_NORM'] = (self.datos_contador['CONSUMO'] - self.datos_contador['CONSUMO'].mean()) / self.datos_contador['CONSUMO'].std()
        self.datos_contador['CONSUMO_CAL_NORM'] = (self.datos_contador['CONSUMO_CAL'] - self.datos_contador['CONSUMO_CAL'].mean()) / self.datos_contador['CONSUMO_CAL'].std()

        # ELIMINACIÓN DE OUTLIERS USANDO EL CONSUMO NORMALIZADO Y LIMITE DE DESVIACIÓN STANDARD PROGRAMABLE
        if self.dias_test > 0:
            self.datos_contador.drop(self.datos_contador.index[np.where(np.abs(self.datos_contador['CONSUMO_NORM'].iloc[:-self.dias_test])>self.Outlier)[0]], inplace=True)
            self.datos_contador.drop(self.datos_contador.index[np.where(np.abs(self.datos_contador['CONSUMO_CAL_NORM'].iloc[:-self.dias_test])>self.Outlier)[0]], inplace=True)
        elif self.dias_test == 0:
            self.datos_contador.drop(self.datos_contador.index[np.where(np.abs(self.datos_contador['CONSUMO_NORM'])>self.Outlier)[0]], inplace=True)
            self.datos_contador.drop(self.datos_contador.index[np.where(np.abs(self.datos_contador['CONSUMO_CAL_NORM'])>self.Outlier)[0]], inplace=True)

    def modelo(self):
        # DECISION DEL MODELO A TOMAR EN FUNCION DE LA ESTADISTICAS
        # CARGA DEL MODELO
        # LA MEDIA CORRECTA SE CALCULA POSTERIORMENTE CON LOS DATOS YA PARTIDOS EXCEPTO SI ES CERO

        self.model = None
        if self.dias > self.umbral: 
            if self.fecha_fin_contador == self.Fecha_Fin:
                if self.Cargar_Modelo:  # CARGAR MODELO DESDE FICHERO
                    self.model = load('./modelos/Modelo_ID_' + str(self.ID))
                    if isinstance(contador.model,ForecasterAutoreg):
                        self.Mejor_Lag = self.model.max_lag
                        self.Mejor_Parametro = {'max_depth':self.model.regressor.max_depth,'n_estimators':self.model.regressor.n_estimators}
                else:   # USAR MODELO GENERICO
                    # RandomForestRegressor Diario
                    self.model = ForecasterAutoreg(regressor=RandomForestRegressor(random_state=123), lags=14)
                    # RandomForestRegressor Semanal
                    self.model_Semanal = ForecasterAutoreg(regressor=RandomForestRegressor(random_state=123), lags=2)
            else:   # USAR COMO PREDICCIÓN LA MEDIA
                self.model = 1

        else:
            if self.dias > 0: 
                self.model = 1
            else:
                self.model = 0

    def Juntar_Datos(self):
        # JUNTA TODO EN datos_consumo
        # PRE-PROCESA DATOS DE TEST

        self.datos_consumo = pd.concat([self.datos_consumo,self.datos_test])
        self.datos_test =pd.DataFrame()
        
        

class CajaMar_Water(object):
    '''
    Clase principal Cajamar_Water que realiza las siguientes funciones:
    
        - Carga los datos.
        - Genera los contadores a partir de la clase contador.
        - Entrena y hace grid de los modelos.
        - Cálculo del RMSE.
        - Tiene las funciones auxiliares de visualización y guardado de modelos/configuraciones.
        - Genera el fichero de respuesta.
    
    Parameters
    ----------
        
    Ruta_Datos : str 
                Ruta del fichero con los datos de los contadores en el formato
                del Datathon. 
    
    Ruta_Mejor: str
                Ruta del fichero que guarda la configuración del mejor pre-procesado
                para seleccionar el valor de la media movil. Default='resultados_1_30.xlsx'.
    
    Fecha_Inicio: date
                  Fecha de inicio teórica de los datos para saber si la serie está completa.
                  Default=date(2019,2,1).

    Fecha_Fin: date
               Fecha de finalización teórica de los datos para saber si la serie está completa.
               Default=date(2020,1,31).

    umbral_modelo: int
                   Valor númerico que indica el número de dias de datos a partir
                   del cual se utiliza modelo. Por debajo del umbral se utiliza
                   la media como predicción. Default=100.

    Modelo_Semanal: bool
                    True -> Se utlizan dos modelos uno para predecir dias
                    y otro para predecir semanas.
                    False -> Solo un modelo para predecir dias.
                    Default: False

    Media_Movil: int
                 Valor númerico que indica el número de dias que se usa 
                 para el pre-procesado de la media movil.
                 Default: 0 (No aplica media movil).

    Cargar_Modelos: bool
                    True -> Carga el modelo en la ruta "./modelos/Modelo_ID_"
                    + ID del contador.
                    Default: False.

    Cargar_Conf_modelo: bool
                        True -> Carga la configuración del modelo desde el 
                        archivo: 'Conf_modelos.xlsx'.
                        Default: False.
    Outlier: int
             Valor númerico que indica las desviaciones tipicas a partir de las
             cuales se consideran outliers y se elimina el dato de consumo y consumo
             calculado a partir de la lectura.
             Default: 8

    Cargar_Preprocesado: bool
                    True -> Carga la configuración de medias moviles del fichero
                            indicado en Ruta_Mejor.
                    False -> Utiliza la misma media movil para todos los contadores
                            el parámetro Media_Movil
                    Default: False

    dias_test: int
                   Valor númerico que indica los dias que se guardan para test
                   default: 14

    Returns 
    -------
    
    Las variables se almacen en la propia clase, por tanto, no devuelve ningún
    parámetro.

    '''   
    def __init__(self,Ruta_Datos,Ruta_Mejor='resultados_1_30.xlsx',Fecha_Inicio=date(2019,2,1),Fecha_Fin=date(2020,1,31),
                umbral_modelo=100, Modelo_Semanal=False, Media_Movil=0,Cargar_Modelos=False,Cargar_Conf_modelo=False,
                Outlier=8,Cargar_Preprocesado=False,dias_test=14):
        self.Ruta_Datos = Ruta_Datos
        self.Ruta_Mejor = Ruta_Mejor
        self.Fecha_Inicio = Fecha_Inicio
        self.Fecha_Fin = Fecha_Fin
        self.umbral_modelo = umbral_modelo
        self.Modelo_Semanal = Modelo_Semanal    # True usamos dos modelos uno para dias y otro para semanas
        self.Media_Movil = Media_Movil
        self.Cargar_Modelos = Cargar_Modelos
        self.Cargar_Conf_modelo = Cargar_Conf_modelo
        self.Outlier = Outlier
        self.Cargar_Preprocesado = Cargar_Preprocesado
        self.datos = pd.DataFrame()
        self.contadores = []
        self.dias_test = dias_test      # DIAS QUE SE RESERVAN PARA TEST EMPEZANDO POR LA FECHA MAS RECIENTE
        self.semanas_test = 2           # SEMANAS QUE SE RESERVAN PARA TEST
        self.Cargar_Datos()
        self.RMSE_Test_Promedio=0
        self.RMSE_Test_Suma=0
        self.dataframe_RMSE = pd.DataFrame()
        
    

    def Cargar_Datos(self):
        # CARGA EL FICHERO DE DATOS Y GENERA LOS CONTADORES
        # CARGA CONFIGURACIONES DE MODELOS Y PREPROCESADO SI PROCEDE

        print('CARGANDO DATOS.....')
        self.datos = pd.read_csv(self.Ruta_Datos,sep='|',parse_dates=['SAMPLETIME'],infer_datetime_format=True,encoding='utf-8')
        self.datos.sort_values(by=['ID','SAMPLETIME'],inplace=True)
        if self.Cargar_Preprocesado:
            self.Mejor_Preprocesado = pd.read_excel(self.Ruta_Mejor,index_col=0)
        
 
        # GENERANDO CONTADORES
        print('GENERANDO CONTADORES.....')
        for ID in tqdm(pd.unique(self.datos['ID'])):
            # ALMACENAMOS ID DE CONTADOR Y BORRAMOS LA COLUMNA
            if self.Cargar_Preprocesado:
                self.contadores.append(Contador(self.datos[self.datos['ID']==ID].drop('ID',axis=1),ID,self.umbral_modelo,self.Modelo_Semanal,
                                            int(self.Mejor_Preprocesado.loc[ID]['Mejor']),self.Fecha_Fin,self.Cargar_Modelos,self.Outlier,self.dias_test))
            else:
                self.contadores.append(Contador(self.datos[self.datos['ID']==ID].drop('ID',axis=1),ID,self.umbral_modelo,self.Modelo_Semanal,
                                            self.Media_Movil,self.Fecha_Fin,self.Cargar_Modelos,self.Outlier,self.dias_test))

        print('PROCESO DE CARGA FINALIZADO')
        if self.Cargar_Conf_modelo:
            print('ACTUALIZANDO CONFIGURACIÓN MODELOS')
            Res_load = pd.read_excel('Conf_modelos.xlsx',index_col=0)
            for contador in tqdm(self.contadores):
                try:
                    contador.Mejor_Parametro = {'max_depth':Res_load.loc[contador.ID]['max_depth'],
                                                'n_estimators':Res_load.loc[contador.ID]['n_estimators']}
                    contador.Mejor_Lag = Res_load.loc[contador.ID]['Lag']

                    contador.model.set_params(**contador.Mejor_Parametro)
                    contador.model.lags = np.arange(contador.Mejor_Lag) + 1
                    contador.model.max_lag = contador.Mejor_Lag
                    contador.model.window_size = contador.Mejor_Lag
                    
                except:
                    None
    
    
    def PartirDatos(self):
        # Separar datos de entrenamiento y test / validacion (dias=0 no hace validacion)
        if self.dias_test > 0:     
            for contador in tqdm(self.contadores):
                if contador.SinDatos == False:      # COMPROBAMOS QUE HAY ALGUN REGISTRO
                    #if (contador.fecha_fin_contador==self.Fecha_Fin):
                    if contador.dias > self.dias_test + 1:    # COMPROBAMOS QUE HAY SUFICIENTES DATOS PARA HACER PARTICION
                        contador.datos_test = contador.datos_consumo.tail(self.dias_test)
                        contador.datos_consumo = contador.datos_consumo[0:-self.dias_test]
                    if contador.semanas > self.semanas_test + 1:
                        contador.datos_test_semanal = contador.datos_consumo_semanal.tail(self.semanas_test)
                        contador.datos_consumo_semanal = contador.datos_consumo_semanal[0:-self.semanas_test]


    
    def Grid_Search(self,ID,steps):
        # Grid Search de los parametros del modelo
        # Hiperparámetros del regresor
        param_grid = {'n_estimators': [100,200],
              'max_depth': [3, 5, 10]}

        # Lags utilizados como predictores
        lags_grid = [14, 30]

        resultados_grid = model_selection_no_print.grid_search_forecaster(
                                forecaster         = self.contadores[ID].model,
                                y                  = self.contadores[ID].datos_consumo['CONSUMO'],
                                param_grid         = param_grid,
                                lags_grid          = lags_grid,
                                steps              = steps,
                                refit              = True,
                                metric             = 'mean_squared_error',
                                initial_train_size = len(self.contadores[ID].datos_consumo) - self.dias_test,
                                return_best        = True,
                                verbose            = False)
        self.contadores[ID].RMSE_Train = math.sqrt(resultados_grid['metric'].head(1))
        self.contadores[ID].Mejor_Parametro = list(resultados_grid['params'])[0]
        self.contadores[ID].Mejor_Lag = max(list(resultados_grid['lags'].head(1))[0])    

    def Entrenamiento(self,Grid=False,Print_disable=False, dias_pred=14):
        # Entrenamiento de los modelos en self.contadores(lista) 
        # Rellena la variable self.contadores.Predicciones
        self.dias_pred = dias_pred
        if not Print_disable:
            print('ENTRENANDO MODELOS.....')
        indexC = 0
        for contador in tqdm(self.contadores,disable=Print_disable):
            if isinstance(contador.model,ForecasterAutoreg):    # COMPROBAMOS QUE ES UN MODELO
                if Grid:
                    self.Grid_Search(indexC,self.dias_test)
                    #contador.Juntar_Datos() #  JUNTA LOS DATOS PARA LUEGO HACER PREDICT
                
                #contador.model.fit(y=contador.datos_consumo['CONSUMO_NORM'])
                
                contador.model.fit(y=contador.datos_consumo['CONSUMO_MED'])

                predict = contador.model.predict(steps=self.dias_pred)

                if self.Modelo_Semanal:
                    contador.model_Semanal.fit(y=contador.datos_consumo_semanal['CONSUMO'])
                    predict_semanal = contador.model_Semanal.predict(steps=self.semanas_test)
                
                index = 1
                for predict_dia in predict[0:14]:    # GUARDA 14 DIAS PREDICCION
                    contador.Predicciones[index] = predict_dia
                    index += 1
                if self.Modelo_Semanal:
                    contador.Predicciones[15]=predict_semanal[0]              # GUARDA SEMANA1 PREDICCION
                    contador.Predicciones[16]=predict_semanal[1]              # GUARDA SEMANA2 PREDICCION
                else:
                    contador.Predicciones[15]=predict[0:7].sum()     # GUARDA SEMANA1 PREDICCION
                    contador.Predicciones[16]=predict[7:14].sum()    # GUARDA SEMANA2 PREDICCION
            else:
                if contador.model == 1:     # RECALCULAMOS LA MEDIA
                    media = contador.datos_consumo['CONSUMO_MED'].mean()
                elif contador.model == 0:
                    media = 0
                for index in range(1,15):
                    contador.Predicciones[index] = media  # GUARDAMOS MEDIA EN LOS 14 DIAS
                contador.Predicciones[15]=media*7     # GUARDAMOS MEDIA SEMANA1
                contador.Predicciones[16]=media*7     # GUARDAMOS MEDIA SEMANA2
            
            indexC += 1
                  
    def Calculo_RMSE(self,Print_disable=False):
        # Calculo rmse utlizando datos de test
        if not Print_disable:
            print('CALCULANDO RMSE.....')
        RMSE_Todos = []
        ID_Todos = []
        for contador in tqdm(self.contadores,disable=Print_disable):
            #if contador.dias > 180 and len(contador.datos_test) == self.dias_test:  # VER LIMITE NO DEPENDE DE LA CLASE CONTADOR
            if len(contador.datos_test) == self.dias_test and len(contador.datos_test_semanal) == self.semanas_test:
                if self.Modelo_Semanal:
                    media_RMSE_semanal=math.sqrt(mean_squared_error(contador.datos_test_semanal['CONSUMO'] , [contador.Predicciones[15], contador.Predicciones[16]]))
                else:
                    media_RMSE_semanal=math.sqrt(mean_squared_error([contador.datos_test['CONSUMO'][0:7].sum(), contador.datos_test['CONSUMO'][7:14].sum()] , [contador.Predicciones[15], contador.Predicciones[16]]))
                
                contador.RMSE_Test = 0.5*math.sqrt(mean_squared_error(contador.datos_test['CONSUMO'][0:7],contador.Predicciones[1:8]))+0.5*media_RMSE_semanal
                
                RMSE_Todos.append(contador.RMSE_Test)
                ID_Todos.append(contador.ID)

        self.RMSE_Test_Suma = sum(RMSE_Todos)
        if self.RMSE_Test_Suma != 0:
            self.RMSE_Test_Promedio = self.RMSE_Test_Suma / len(RMSE_Todos)
            self.dataframe_RMSE = pd.DataFrame(data=RMSE_Todos,index=ID_Todos,columns=['RMSE'])
            self.dataframe_RMSE.sort_values(by=['RMSE'],inplace=True)
            #print(dataframe_RMSE)
        else:
            print('ERROR NO HAY CALCULO DE RMSE')
        
    def Grid_Medias_Moviles(self,Grid=[1,2,3,4,5,6,10],Print_disable=True):
        # REALIZA BARRIDO CON DISTINTOS VALORES DE MEDIAS PARA GENERAR
        # FICHERO resultados.xlsx CON LA MEJOR COMBINACIÓN

        self.PartirDatos()
        Resultados = pd.DataFrame()
        RMSE = []
        print('CALCULANDO GRID CON MEDIAS MOVILES.....')
        for x in tqdm(Grid):
            for contador in self.contadores:
                if not contador.datos_consumo.empty:
                    contador.datos_consumo['CONSUMO_MED'] = contador.datos_consumo['CONSUMO'].rolling(x).mean()
                    contador.datos_consumo['CONSUMO_MED'].fillna(0,inplace=True)
            self.Entrenamiento(Print_disable=Print_disable)
            self.Calculo_RMSE(Print_disable=Print_disable)
            for contadores in self.contadores:
                RMSE.append(contadores.RMSE_Test)
            Resultados['RMSE_MOV_'+str(x)] = RMSE
            RMSE = []

        ID = []
        for contador in self.contadores:
            ID.append(contador.ID)

        Resultados.index = ID
        Mejor = []
        RMSE_Mejor = []
        for index, data in Resultados.iterrows():
            Mejor.append(data.argmin())
            RMSE_Mejor.append(data[data.argmin()])
        Resultados['Mejor_RMSE'] = RMSE_Mejor
        Resultados['Mejor'] = Mejor
        Resultados.to_excel('Resultados_Prueba.xlsx')

    def GenerarResultados(self,equipo='Team2021IA3',numeral=1):
        # Mediante los datos predicciones generamos txt con el formato del datathon
        # Separando campos con “|”, el valor de la predicción en litros, y los decimales con “.” 2 decimales
        # Fichero: "Cajamar_Universitat de València (UV)_Team2021IA3_numeral.txt"
        # Fichero final: Team2021IA3.txt
        
        columnas = ['ID','Dia_1','Dia_2','Dia_3','Dia_4','Dia_5','Dia_6','Dia_7','Semana_1','Semana_2']
        datos = []

        for contador in tqdm(self.contadores):
            aux = contador.Predicciones[0:8] + contador.Predicciones[15:17] # ELIMINAMOS DE LAS PREDICCIONES LA SEGUNDA SEMANA EN DIAS
            datos.append(aux)
        
        print('TAMAÑO DE LOS DATOS: ', len(datos), 'x' ,len(datos[0]))
        
        dataset = pd.DataFrame(datos,columns=columnas)
        dataset.to_csv(equipo + '.txt',header=False ,sep='|' ,index=False ,decimal='.' ,float_format='%.2f')
         
    def GraficasEstadisticas(self,contador=0,Test=True):
        # GRÁFICAS DE CONTADORES INDIVIDUALES CON TEST Y PREDICCIONES
        if Test:
            fig, ax = plt.subplots(figsize=(10, 4))
            self.contadores[contador].datos_test['CONSUMO'].plot(ax=ax, label='Test', linewidth=1)
            pd.DataFrame(self.contadores[contador].Predicciones[1:15],index=self.contadores[contador].datos_test['CONSUMO'].index[0:14],columns=['Predicción']).plot(ax=ax, label='Prediccion', linewidth=1)
            ax.set_title('Consumo Agua. RMSE: ' + str(self.contadores[contador].RMSE_Test) + ' Contador: ' + str(contador))
            ax.legend();
        else:
            base = date(2020,2,1)
            index = [base + timedelta(days=x) for x in range(7)]
            index = pd.DataFrame(index)
            index.insert(1, 'SAMPLETIME', [base + timedelta(days=x) for x in range(7)], True)
            index['SAMPLETIME']=pd.to_datetime(index['SAMPLETIME'], format='%Y-%m-%d')
            index=index.set_index('SAMPLETIME')
            index_puro=index.drop([0],axis=1)
            index_puro['Predicción']=Proyecto.contadores[contador].Predicciones[1:8]
            #df_final=pd.DataFrame(index_puro.insert(1, 'Predicción', pd.Series(self.contadores[contador].Predicciones[1:8]), True))

            fig, ax = plt.subplots(figsize=(10, 4))
            self.contadores[contador].datos_consumo['CONSUMO'].plot(ax=ax, label='Consumo', linewidth=1)
            index_puro.plot(ax=ax, label='Predicción', linewidth=1)
            #pd.DataFrame(self.contadores[contador].Predicciones[1:8],index=index_puro,columns=['Predicción']).plot(ax=ax, label='Prediccion', linewidth=1)
            ax.set_title('Consumo Agua. RMSE: ' + str(self.contadores[contador].RMSE_Train) + ' Contador: ' + str(contador))
            ax.legend();
    
    def Mejor_Peor_Resultado(self,cantidad=1):
        # MOSTRAMOS LOS MEJORES Y PEORES RESULTADOS SEGUN EL RMSE
        print('RMSE PROMEDIO: ',self.RMSE_Test_Promedio,'   RMSE SUMA: ',self.RMSE_Test_Suma)
        RMSE_SIN_0 = self.dataframe_RMSE[self.dataframe_RMSE['RMSE'] > 1]  # NO MOSTRAMOS LOS VALORES MENOR O IGUAL QUE 1 DE RMSE
        for contador in RMSE_SIN_0.head(cantidad).index:
            self.GraficasEstadisticas(contador)
        for contador in self.dataframe_RMSE.tail(cantidad).index:
            self.GraficasEstadisticas(contador)


    def Guardar_Modelos(self,Ruta='./modelos/Modelo_ID_',Excel=False):
        # GUARDA TODOS LOS MODELOS A FICHERO O EXCEL SEGUN LA OPCIÓN

        print('GUARDANDO MODELOS.....')
        if Excel:   # GUARDAR LA CONFIGURACIÓN DEL MODELO EN EXCEL
            Conf_model_Lag = []
            Conf_model_max_depth = []
            Conf_model_n_estimators = []
            Conf_model_ID = []

            for contador in tqdm(self.contadores):
                if isinstance(contador.model,ForecasterAutoreg): 
                    Conf_model_Lag.append(contador.Mejor_Lag)
                    Conf_model_max_depth.append(contador.Mejor_Parametro['max_depth'])
                    Conf_model_n_estimators.append(contador.Mejor_Parametro['n_estimators'])
                    Conf_model_ID.append(contador.ID)

            Res = pd.DataFrame()
            Res['Lag'] = Conf_model_Lag
            Res['max_depth'] = Conf_model_max_depth
            Res['n_estimators'] = Conf_model_n_estimators
            Res.index = Conf_model_ID

            Res.to_excel('Conf_modelos.xlsx')
        else:   # GUARDAR EL MODELO ENTERO
            for contador in tqdm(self.contadores):
                dump(contador.model,Ruta + str(contador.ID))

        