#
#  data_analysis.py
#
#  meant to be imported as:     import data_analysis as dana
#
#  Created by Florian Hoppe on 07.11.2013.
#

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import pandas as pd
import statsmodels.api as sm

PVALUE_SIGNIFICANCE_LEVEL = 0.05

def build_OLS_model(all_independent_dims, data, dependent_dim = 'y'):
    """Convenient function to fit a OLS model to some data. A constant component is automatically added to the data.

    Args:
        all_features: array of strings with names of all candidates for input dimensions
        data: DataFrame that holds the data. it must contain columns for every given feature and one column with the
         dependent variable
        dependent_dim: a string with the name of the column of the dependent variable i.e. the output
    Returns:
        the OLS model
    Raises:
        assert exceptions: for invalid inputs
    """

    assert set(all_independent_dims).issubset(set(data.columns)), 'Not all data columns (' + ", ".join(all_independent_dims) + \
                                                                  ') are present in input dataframe.'
    assert dependent_dim in data.columns, 'Column ' + dependent_dim + ' must be present in input dataframe.'

    independent_data = data[all_independent_dims]
    independent_data['const'] = 1.0
    depend_data = data[dependent_dim]

    mod = sm.OLS(depend_data,independent_data)
    return mod.fit()

def find_important_OLS_features(all_independent_dims, data, dependent_dim = 'y'):
    """Tries to identify which input dimension are important to solve a linear regression (ordinary least square) problem.

    Args:
        all_features: array of strings with names of all candidates for input dimensions
        data: DataFrame that holds the data. it must contain columns for every given feature and one column with the
         dependent variable
        dependent_dim: a string with the name of the column of the dependent variable i.e. the output
    Returns:
        an array of strings of these input dimension that give a significant contribution to the OLS model
        (this wont contain the automatically added column 'const')
    Raises:
        assert exceptions: for invalid inputs
    """

    model = build_OLS_model(all_independent_dims, data, dependent_dim)
    better_input_dims = model.pvalues[model.pvalues<=PVALUE_SIGNIFICANCE_LEVEL].index.values

    return (better_input_dims[better_input_dims!='const'],model)

def build_simplest_OLS_model(all_independent_dims, data, dependent_dim = 'y'):
    """Builds a OLS model with the smallest degrees of freedom for the given data. Therefore it automatically removes
     those input dimensions that show no significant contribution to compute the output.
     Warning: the returned model does not necessarily be the best model w.r.t. to achieved performaced. Check R-squared
     and residual error value.

    Args:
        all_features: array of strings with names of all candidates for input dimensions
        data: DataFrame that holds the data. it must contain columns for every given feature and one column with the
         dependent variable
        dependent_dim: a string with the name of the column of the dependent variable i.e. the output
    Returns:
        the OLS model
    Raises:
        assert exceptions: for invalid inputs
    """

    (best_input_dims, model) = find_important_OLS_features(all_independent_dims, data, dependent_dim)
    if len(best_input_dims)==len(all_independent_dims):
        print("Found simplest model to have %d degrees of freedom." % len(best_input_dims))
        return model
    else:
        print("Reducing degrees of freedom from %d to %d." % (len(all_independent_dims), len(best_input_dims)))
        return build_simplest_OLS_model(best_input_dims, data, dependent_dim)