#!/bin/usr/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.dpi'] = 300
import pickle

#ACCURACY FUNCTIONS
def get_stats(accuracy, show_stats=True):
    maximum = accuracy.max()
    minimum = accuracy.min()
    mean = accuracy.mean()
    std = accuracy.std()
    if show_stats:
        print("/nGeneral Stadistics\n")
        print(('max accuracy:%s\nmin accuracy:%s\naverage accuracy:%s\nstandard desviation:%s\n')%(maximum,minimum,mean,std))
    return (maximum,minimum,mean,std)

def plot_accuracy_bars(features_names,accuracy,store_plot=False):
    minimum=accuracy.min()
    plt.xlabel('Accuracy')
    plt.ylabel('Features')
    plt.xlim(minimum-0.1, 1)
    plt.tick_params(labelsize=6)
    plt.barh(features_names,accuracy)
    plt.show()

def show_accuracy(accuracy_filename):
    res=np.load(accuracy_filename)
    maximum,minimum,mean,std = get_stats(res)
    plot_accuracy_bars(features_names=state_names,accuracy=res)


#PLOT TRACES
def plot_results(estimate_trace,real_trace,title="Non title provided"):
    plt.plot(estimate_trace,label='Predictive trace')
    plt.plot(real_trace,label='Real trace',linewidth=0.3)
    #plt.plot(real_trace,'*',label='Real trace')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_several_traces(estimate_trace,real_trace):
    for column in range(30):
        state=state_names[column]
        plot_title=('Estimation vs Real for state: %s'%state)
        plot_results(estimate_trace=diff_state_estimation[:,column],real_trace=diff_out[:,column],title=plot_title)

def plot_all_traces(predictor_trace, real_trace, title):
    for column in range(30):
        subtitle=title+' column:'+str(column)
        plot_results(estimate_trace=predictor_trace[:,column],real_trace=real_trace[:,column],title=subtitle)
