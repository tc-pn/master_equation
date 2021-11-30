import math as ma
import numpy as np
import scipy as sc
import itertools as it
import os
import errno
from IPython.display import *

from module_fig_gen import *

# the total number of occupied states is conserved

# Number of 1p states
A = 6

# Number of occupied states
N = 2

# time step and number of time step
Npas = 10000
dt = 0.001
TIME = np.linspace(0, Npas * dt, Npas)
Nsave = 10

# we are interested in MpMh excitations:
M = 2

PATH = "/projet/pth/czuba/2021/tdhfbproj/" # path of the project, can be changed directly

def markov(A,N,Npas,dt):
# the markov equation solver

    basis = gen_many_body_basis(A,N) # generation of the basis
    size_basis = len(basis)

    mb_weights = ini_state(size_basis) # choose the initial weights

# initialization

    newweights = np.copy(mb_weights)
    oldw = np.copy(mb_weights)
    save_weights = np.copy(newweights)

# mask has 2 uses: 
# state alpha must not be counted in the sums of the RHS (implemented), 
# and force the interactions to be of 2p2h nature (implemented)
# there is one mask per state, computed once before the time evolution
    mask = []
    for i in range(size_basis):
        mask.append(mask_gen(i,basis))

# Gain and Loss terms
# i need the mask for this 
# the gain and loss matrices are constructed, then the mask is applied when normalizing, 
# so that the sum of the RELEVANT terms (i.e. mask allow them) is equal to 1

    Gain, Loss = GL_terms(size_basis,mask)

# RK2 propagation

    for it in range(Npas):

# t + dt/2

        for i in range(size_basis):
            newweights[i] = oldw[i] + dt / 2. * np.sum(Gain[i,:] * oldw - Loss[:,i] * oldw[i], where = mask[i])

# t + dt

        for i in range(size_basis):
            newweights[i] = newweights[i] + dt * np.sum(Gain[i,:] * oldw - Loss[:,i] * newweights[i], where = mask[i])

        oldw = newweights

        save_weights = np.vstack((save_weights,newweights))

        if it % 100 == 0:
            print(it)

# saving the figures
    save_data(save_weights)

    pass

def gen_many_body_basis(A,N):
# the basis

    state = np.zeros(A)
    for i in range(N):
        state[i] = 1

    return list(set(it.permutations((state)))) # construct all UNIQUE permutations of N ones in a array of length A

def ini_state(size_basis):
# the many-body initial weights

    mb_weights = np.zeros(size_basis) 
    mb_weights[0] = 1. 
#    mb_weights[4] = 1. 
    mb_weights = mb_weights / np.sum(mb_weights) # renormalized 

    return mb_weights

def mask_gen(i,basis):

    size_basis = len(basis)

    mask = np.ones(size_basis, dtype = bool)
    mask[i] = False 

    for j in range(len(basis)):
        test = np.array(basis[i]) - np.array(basis[j])
# an excitation is MpMh is recognized if the difference between the 2 arrays yields M minus ones
        nbr_minus = list(test).count(-1.)
#        print(nbr_minus)
        if nbr_minus != M:
            mask[j] = False

    return mask

def GL_terms(size_basis,mask):
# the gain and loss matrices involved in the master equation
# CAREFUL ! The gain and loss matrices are normalized using the mask, 
# but irrelevant elements (i.e. not MpMh escitations) are not set to 0 !

    Gain = np.ones((size_basis,size_basis))
    for i in range(size_basis):
        Gain[i,:] = Gain[i,:] / np.sum(Gain[i,:], where = mask[i])
#        print(mask[i])
#       print(np.sum(Gain[i,:], where = mask[i]))
#        print(np.sum(mask[i]))

    Loss = Gain

    return Gain, Loss

def sp_weights():
# construction of the sp weights
    weights = np.genfromtxt("weights.dat")

    basis = np.array(gen_many_body_basis(A,N))
    sp_weights = np.zeros((Npas+1,A))

 #   print(np.shape(basis))

    for i in range(A):
        for j in range(len(basis)):
#            print(i,j)
            if basis[j,i] == 1:
                sp_weights[:,i] = sp_weights[:,i] + weights[:,j]

    filename = PATH+"sp_weights.dat"

# some lines of code to insure no problem arise if the new file do not already exists

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    f = open(filename, 'wb')
    np.savetxt(f, sp_weights)
    f.close()

    pass


# saving files

def save_data(save_weights):

    filename = PATH+"weights.dat"

# some lines of code to insure no problem arise if the new file do not already exists

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    f = open(filename, 'wb')
    np.savetxt(f, save_weights)
    f.close()
    pass

# showing results

def weights_fig():
# shows the many-body weights
    labels = [r'$t$ (fm/c)', r'$P_{\alpha}(t)$']

    ax = spec_fig(1, 1, size_fig, labels)

    weights = np.genfromtxt("weights.dat")

    for i in range(np.shape(weights)[1]):
        ax.plot(TIME,weights[:-1,i],'-', linewidth = 1)

#    ax.plot(TIME,np.sum(weights[:-1,:], axis = 1),'--', linewidth = 1) # is the total proba conserved ?
#    print(np.sum(weights[-1,:]))

    show()

    pass

def sp_weights_fig():
# shows the sp weights
    labels = [r'$t$ (fm/c)', r'$n_{k}(t)$']

    ax = spec_fig(1, 1, size_fig, labels)

    spweights = np.genfromtxt("sp_weights.dat")

    print(np.sum(spweights[-1,:]))

    for i in range(np.shape(spweights)[1]):
        ax.plot(TIME,spweights[:-1,i],'-', linewidth = 1)

#    ax.plot(TIME,np.sum(weights[:-1,:], axis = 1),'--', linewidth = 1) # is the total proba conserved ?
#    print(np.sum(weights[-1,:]))

    show()

    pass
