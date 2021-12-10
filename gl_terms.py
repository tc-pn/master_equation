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

# some lines of code to insure no problem arise if the new file do not already exists

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

def lorentz(basis,epsilon,N):
# calculate the lorentzian weight associated to a jump

    size_basis = len(basis)

    mbenergies = np.zeros(size_basis)
    for i in range(size_basis):
#        print(np.array(basis[i]))
#        print(np.array(np.where(np.array(basis[i]) == 1.)).flatten())
        mbenergies[i] = np.sum(epsilon * np.array(np.where(np.array(basis[i]) == 1.)).flatten()) # calculation of the energy of each many body state


# ok so the idea is to construct a tensor gamma but this will yield something of the size A**4 which will be gigantic
# using sparce matrices i think it will be easier but the way it will work is not clear in comparison to the newaxis method of numpy
#    diff = np.array(basis)[np.newaxis,:] - np.array(basis)[:,np.newaxis]
    gamma = width_lorentz(basis,epsilon,N) # calculation of gamma

    lorentz = gamma / ((mbenergies[:,np.newaxis] - mbenergies[np.newaxis,:])**2. + (gamma/2.)**2.) # each gain and loss terms are given by lorentzian matrices, assuming V = 1
    np.fill_diagonal(lorentz,0.) # the state does not interact with itself
#    print(lorentz) 
    lorentz = lorentz / (2. * np.pi)

    return lorentz

def width_lorentz(basis,epsilon,N): 
# calculation of width of lorentzian weight
# here i assume the Bohr Mottelson Vol. 1 hypotheses are valid (see pages 302-304)
# i have to change it, bertsch is supposed to be more realistic => we'll do with the Pines Nozieres 
# in the end i just assume as in P and N that the total width is the sum of the particle or hole widths which are proportional to (energy - fermi energy)**2 * a cutoff according to Bertsch 

    fermi_energy = N * (N + 1.) / 2. * epsilon
    
#    diff = np.array(basis)[:,np.newaxis] - np.array(basis)[np.newaxis,:]
#    diff = 
    gamma = np.zeros((len(basis),len(basis)))
    cutoff = 4. *  epsilon
    for i in range(len(basis)): # pas mega efficace le calcul là
        for j in range(i):
#        for j in range(len(basis)):
            diff = np.array(basis[i]) - np.array(basis[j])
#            print(i,j)
            posindex = np.array(np.where(np.array(diff) == 1.)).flatten()
            negindex = np.array(np.where(np.array(diff) == -1.)).flatten()
            for k in range(len(posindex)):
                x = (posindex[k] * epsilon - fermi_energy)**2.
                #print(x * np.exp(-x),i,j,k)
                gamma[i,j] = gamma[i,j] + x * np.exp(-x/(2. * cutoff**2.))
            for k in range(len(negindex)):
                x = (negindex[k] * epsilon - fermi_energy)**2.
                gamma[i,j] = gamma[i,j] + x * np.exp(-x/(2. * cutoff**2.))
            gamma[j,i] = gamma[i,j]
#        print(gamma[i,i])
 #   gamma = 0.
 #   for i in range(len(posindex)):
            
 #   gamma = 2. * np.pi  / epsilon
#    gamma = 1.
    return gamma
