# math import
import math as ma
import numpy as np
import scipy as sc
import itertools as it
# path management import
import os
import errno
from IPython.display import *

def mask_gen(i,basis,M):

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

def GL_terms(basis,mask,lorentz_weights,epsilon):
# the gain and loss matrices involved in the master equation
# CAREFUL ! The gain and loss matrices are normalized using the mask, 
# but irrelevant elements (i.e. not MpMh escitations) are not set to 0 !
# the matrix elements are chosen to be lorentzian functions of th energy with a constant gamma and a V equal to 1

    size_basis = len(basis)

    Gain = np.ones((size_basis,size_basis)) # matrix elements of the residual interaction

    Gain = Gain * lorentz_weights # adding the lorentz weight

#    for i in range(size_basis): # normalization to prevent loss of proba
#        Gain[i,:] = Gain[i,:] / np.sum(Gain[i,:], where = mask[i])
#        print(mask[i])
#       print(np.sum(Gain[i,:], where = mask[i]))
#        print(np.sum(mask[i]))

    Loss = Gain

    return Gain, Loss

def lorentz(basis,epsilon):
# calculate the lorentzian weight associated to a jump

    size_basis = len(basis)

    mbenergies = np.zeros(size_basis)
    for i in range(size_basis):
#        print(np.array(basis[i]))
#        print(np.array(np.where(np.array(basis[i]) == 1.)).flatten())
        mbenergies[i] = np.sum(epsilon * np.array(np.where(np.array(basis[i]) == 1.)).flatten()) # calculation of the energy of each many body state

#    gamma = np.zeros((size_basis,size_basis))
    gamma = width_lorentz(epsilon) # calculation of gamma

    lorentz = gamma / ((mbenergies[:,np.newaxis] - mbenergies[np.newaxis,:])**2. + (gamma/2.)**2.) # each gain and loss terms are given by lorentzian matrices, assuming V = 1
    lorentz = lorentz / (2. * np.pi)
    return lorentz

def width_lorentz(epsilon):
# calculation of width of lorentzian weight
# here i assume the Bohr Mottelson Vol. 1 hypotheses are valid (see pages 302-304)

#    fermi_energy = N * (N + 1.) / 2. 

    gamma = 2. * np.pi  / epsilon
#    gamma = 1.
    return gamma
