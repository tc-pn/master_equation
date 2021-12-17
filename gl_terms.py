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

def GL_terms(basis,mask,lorentz_weights,epsilon,PATH):
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
#        print(np.sum(Gain[i,:], where = mask[i]))
#        print(np.sum(mask[i]))

    Loss = Gain

# i save the gain and loss terms to compute the associated characteristic times

    filename = PATH+"gl_terms.dat"

# some lines of code to2 insure no problem arise if the new file do not already exists

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    f = open(filename, 'wb')
    np.savetxt(f, Gain)
    f.close()

    return Gain, Loss

def lorentz(basis,epsilon,N,A):
# calculate the lorentzian weight associated to a jump

    size_basis = len(basis)

    mbenergies = np.zeros(size_basis)
    for i in range(size_basis):
#        print(np.array(basis[i]))
#        print(np.array(np.where(np.array(basis[i]) == 1.)).flatten())
        mbenergies[i] = np.sum(epsilon * np.arange(0,A) * basis[i,:]) # calculation of the energy of each many body state

#    print('ener', mbenergies)

    gamma = width_lorentz(basis,epsilon,N,A) # calculation of gamma
    lorentz = np.zeros((len(basis),len(basis)))

    DE = mbenergies[np.newaxis,:] - mbenergies[:,np.newaxis]

    cutoff = 10.**(-10.)
    for i in range(len(basis)):
#        for j in range(len(basis)):
#            if gamma[i] <= cutoff and np.abs(DE[i,j]) <= cutoff: #if gamma is near zeero then we have a dirac delta function as per the fermi golden rule
#                lorentz[i,j] = 1.
#            elif gamma[i] <= cutoff and np.abs(DE[i,j]) >= cutoff:
#                lorentz[i,j] = 0.
#            else:
#                lorentz[i,j] = 1. / (2. * np.pi) * gamma[j] / (DE[i,j]**2. + (gamma[j]/2.)**2.)

    # here i try to use a gaussian instead of a lorentzian distribution so that i hopefully may have no problems with big basis anymore
    #            width = np.sqrt(7.) * gamma[j]
    #            lorentz[i,j] = np.sqrt(2. * np.pi * width**2.)**(-1.) * np.exp(- DE[i,j]**2. / (2. * width**2.))

    # here i try to be un gros bourrin with heaviside
        lorentz[:,i] = np.heaviside(DE[:,i] + gamma[:] / 2. ,1.) - np.heaviside(DE[:,i] - gamma[:] / 2.,1.)

    return lorentz

def width_lorentz(basis,epsilon,N,A):
# calculation of width of the final states
# they are supposed to be exact, but since we don't know them we have to use a prescription to compute them.
# Parameters will be adjusted one day :)

    gs_energy = N * (N + 1.) / 2. * epsilon # ground state energy
    gamma = np.zeros(len(basis))
    print('gs=', gs_energy)
    a = 1.
    b = 0.5
#    b = 1.

    diff = epsilon * np.sum(np.arange(0,A)[np.newaxis,:] * basis, axis = 1) - gs_energy
    diff = diff**2.
    gamma = a * diff * np.exp(- b * diff)
#    gamma = np.zeros(len(basis))
#    print(np.shape(gamma),'size gamma')

    return gamma
