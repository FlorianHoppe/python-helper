#
#  misc.py
#
#  meant to be imported as:     import misc as hlp
#
#  Created by Florian Hoppe on 06.11.2013.
#

from HTMLParser import HTMLParser
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import pandas as pd

class _MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    """Transforms something like "<a href="example.com">My Link</a>" to "My Link"

    Args:
        html: input string
    Returns:
        string
    Raises:

    """
    s = _MLStripper()
    s.feed(html)
    return s.get_data()

def remove_non_ascii(str):
    return "".join(i for i in str if ord(i)<128)

def transform_series_to_percent(ser, precision="%.2f"):
    """prints a Series of float values nicely as percent

    Args:
        ser: Series of float values
    Returns:
        Series of strings of e.g. "12 %"
    Raises:
        none
    """
    return ser.map(lambda x: (precision+" %%") % x)


#def get_indices_of_filter(data, func):
#    """returns a list of all indices of elements of data that pass the given filter function:
#
#    Args:
#
#    Returns:
#
#    Raises:
#
#    """
#    index_data = zip(func(data),range(len(data)))
#    return [i for x,i in index_data if x]
#
#def get_indices_of_elements_that_pass_filter_func(data, func):
#    return [i for i,x in enumerate(data) if func(x)]


def plot_categorical_data(alldata,datacolumns,catcolumn='cat',markers='ox.hs*8+',colors='rbkym'):
    """Creates a scatter plot of 2D data where each data point belongs to a certain category. Good for visualizing data
    of different classes of a supervised learning problem.

    Args:
        alldata: a DataFrame with at least three different columns
        datacolumns: list of exactly two strings defining the data columns e.g. ['dim1','dim2']
        catcolumn: string of the name of the column of the DataFrame that holds the class labels
        markers: string of markers used for the plotting of the different classes
        colors: string of color codes used for the plotting of the different classes
    Returns:
        none
    Raises:
        assert exceptions for invalid input parameters
    """
    assert len(datacolumns)>=2,'Parameter datacolumns must contain at least two values.'
    assert catcolumn in alldata.columns, 'Column ' + str(catcolumn) + ' must be present in dataframe.'
    assert set(datacolumns).issubset(set(alldata.columns)), 'All data columns (' + ", ".join([str(x) for x in datacolumns]) + ') must be present in dataframe.'
    fig, ax = plt.subplots()
    i = 0
    for cat,data in alldata.groupby(catcolumn):
        ax.scatter(data[datacolumns[0]],data[datacolumns[1]],marker=markers[i%len(markers)],facecolor=colors[i%len(colors)],alpha=.2)
        i += 1
    plt.show()
    return fig, ax

def plot_histograms_to_compare(data1,data2,bins=50):
    """Plots two histograms with same y axis so that the distribution of two data sets can easily be compared

    Args:
        data1: DataFrame for left histogram
        data2: DataFrame for right histogram
        bins: number of bins
        labels: list of two strings defining the headlines of both histograms
    Returns:
        none
    Raises:
        none
    """
    histfig, histaxes = plt.subplots(1,2,sharex=True,sharey=True)
    data1.hist(bins=bins,ax=histaxes[0])
    data2.hist(bins=bins,ax=histaxes[1])
    return histfig, histaxes

def _mean_of_cut_level(level_str):
    comma_idx = level_str.find(',')
    low = float(level_str[1:comma_idx-1])
    high = float(level_str[comma_idx+2:-1])
    return .5*(low+high)

def plot_combined_histograms(data1,data2,bins=50,labels=['A','B']):
    """Plot two histograms in one plot in order to make them easily comparable

    Args:
        data1: DataFrame for left histogram
        data2: DataFrame for right histogram
        bins: number of bins
        labels: list of two strings defining labels in the legend of the plot
    Returns:
        none
    Raises:
        none
    """
    cuts = pd.cut(data1,bins)
    fig, ax = plt.subplots(1,1)
    ax.plot([_mean_of_cut_level(x) for x in cuts.levels],pd.value_counts(cuts)[cuts.levels],'.r', label=labels[0])
    cuts = pd.cut(data2,bins)
    ax.plot([_mean_of_cut_level(x) for x in cuts.levels],pd.value_counts(cuts)[cuts.levels],'.b', label=labels[1])
    return fig, ax