from __future__ import division, print_function
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

#================================================
# Forms and colours of the plots
#================================================
forms = {
    0: '-', 1: '--', 2: '-.', 3: ':'
}

symbols_forms = {
    0: '.', 1: 'v', 2: 's', 3: '^'
}

colours = {
    0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'yellow'
}

#================================================
# Size of the figure
#================================================

xsize, xcolumn = 9, 7
size_fig = (xsize,xcolumn)

#================================================
# Showing the figure
#================================================

def show():

    generic_fig = {
            0: plt.rc('text', usetex=True),
            1: plt.rc('font', family='serif'),
            2: plt.tight_layout(h_pad = 0.1, rect = (-0.05,-0.02,1,1)),
            3: plt.show(block=False)
    }

    return 0, 'show'

def spec_fig(nlines,mcolumns,size_fig,labels):

    fig, axs = plt.subplots(nlines, mcolumns, sharex=False, sharey = False, figsize = size_fig)

    a = fig.add_subplot(111, frameon=False)
    a.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    a.grid(False)
    a.set_xlabel(labels[0], labelpad = 25, fontsize = 23)
    a.set_ylabel(labels[1], labelpad = 35, fontsize = 23)

    if nlines != 1 and mcolumns != 1:
        for i in range(nlines):
            for j in range(mcolumns):
                ax = axs[i,j]
                label_ticks(ax)
    elif nlines != 1:
        for i in range(nlines):
            ax = axs[i]
            label_ticks(ax)
    elif mcolumns != 1:
        for i in range(mcolumns):
            ax = axs[i]
            label_ticks(ax)
    else:
        ax = axs
        label_ticks(ax)

    return axs


def spec_fig_glued(nlines,size_fig,labels):

    fig, axs = plt.subplots(nlines, 1, sharex=True, sharey = False, figsize = size_fig, gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0)

    a = fig.add_subplot(111, frameon=False)
    a.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    a.grid(False)
    a.set_xlabel(labels[0], labelpad = 20, fontsize = 23)

    for i in range(nlines):
        ax = axs[i]
        ax.set_ylabel(labels[i+1], labelpad = 5, fontsize = 23)
        label_ticks(ax)

    return axs

def label_ticks(ax):

    ax.tick_params(direction="in")
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(23)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(23)

    return 0, 'label_ticks'

#================================================
