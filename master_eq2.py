# math import
import math as ma
import numpy as np
import scipy as sc
import itertools as it
# path management import
import os
import errno
from IPython.display import *

from module_fig_gen import *
from gl_terms import *
# here the energy of the many-body states is taken into account by considering that
# |11100...> has total energy of 1 * 0 * epsilon + 1 * 1 * epsilon + 1 * 2 * epsilon + 0 * 3 * espilon + ... 


# the total number of occupied states is conserved

# Number of 1p states
A = 5

# Number of occupied states
N = 2

# time step and number of time step
Npas = 15000
dt = 0.001
TIME = np.linspace(0, Npas * dt, Npas)
Nsave = 10 # save date every Nsave time steps. Not implemented (flemme)

# we are interested in MpMh excitations:
M = 2

# energy of the lowest single particle level => all sp levels are separated by epsilon
epsilon = 1.
# fermi energy
ef = 0.
for i in range(N):
    ef = ef + i * epsilon
print('ef = ', ef)

PATH = "/projet/pth/czuba/2021/tdhfbproj/master_eq2/" # path of the project, can be changed directly

def markov_solver(A,N,Npas,dt):
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
        mask.append(mask_gen(i,basis,M))

# Gain and Loss terms
# i need the mask for this 
# the gain and loss matrices are constructed, then the mask is applied when normalizing, 
# so that the sum of the RELEVANT terms (i.e. mask allows them) is equal to 1
# i don't know how this will play with an energy dependency of the gain and loss matrices

# first i calculate the lorentzian weight

    mbenergies = np.zeros(size_basis)
    for i in range(size_basis):
#        print(np.array(basis[i]))
#        print(np.array(np.where(np.array(basis[i]) == 1.)).flatten())
        mbenergies[i] = np.sum(epsilon * np.array(np.where(np.array(basis[i]) == 1.)).flatten()) # calculation of the energy of each many body state

    lorentz_weights = lorentz(basis,epsilon) # each gain and loss terms are given by lorentzian matrices, assuming V = 1

# then i put it into the gain and loss matrices

    Gain, Loss = GL_terms(basis,mask,lorentz_weights,epsilon)

# RK2 propagation

    for it in range(Npas):

# t + dt/2

        for i in range(size_basis):
            newweights[i] = oldw[i] + dt / 2. * np.sum(Gain[i,:] * oldw - Loss[:,i] * oldw[i], where = mask[i])

# t + dt

        for i in range(size_basis):
            newweights[i] = newweights[i] + dt * np.sum(Gain[i,:] * newweights - Loss[:,i] * newweights[i], where = mask[i])

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

    return list(set(it.permutations((state)))) # constructs all UNIQUE permutations of N ones in a array of length A

def ini_state(size_basis):
# the many-body initial weights

    mb_weights = np.zeros(size_basis) 
    mb_weights[0] = 1. 
#    mb_weights[4] = 1. 
    mb_weights = mb_weights / np.sum(mb_weights) # renormalized 

    return mb_weights

def sp_weights():
# construction of the occupation numbers of the single particle states
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
    print(np.sum(weights[-1,:]))
    print(weights[-1,:])
    show()

    pass

def sp_weights_fig():
# shows the occupation numbers of the single particle states
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

def entropy():
# shows the sp entropy and mb entropy as functions of time 

    labels = [r'$t$ (fm/c)', r'$\mathcal{S}(t)$ ($k_{\rm B}$ units)']

    ax = spec_fig(1, 1, size_fig, labels)

    weights = np.genfromtxt("weights.dat")
    entropy = - np.sum(weights * np.log(weights) + (1. - weights) * np.log(1. - weights), axis = 1)

    weights = np.genfromtxt("sp_weights.dat")
    spentropy = - np.sum(weights * np.log(weights) + (1. - weights) * np.log(1. - weights), axis = 1)

    ax.plot(TIME,entropy[:-1],'-',color = 'black', linewidth = 1)
    ax.plot(TIME,spentropy[:-1],'--',color = 'black', linewidth = 1)

#    ax.plot(TIME,np.sum(weights[:-1,:], axis = 1),'--', linewidth = 1) # is the total proba conserved ?
#    print(np.sum(weights[-1,:]))

    show()

    pass
