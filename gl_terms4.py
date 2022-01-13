# math import
import math as ma
import numpy as np
import scipy as sc
import itertools as it
# path management import
import os
import errno
from IPython.display import *

hbar = 197.32705

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

def GL_terms(basis,mask,epsilon,PATH,A,N,M,ini_energy):
# the gain and loss matrices involved in the master equation
# CAREFUL ! The gain and loss matrices are normalized using the mask,
# but irrelevant elements (i.e. not MpMh escitations) are not set to 0 !
# the matrix elements are chosen to be lorentzian functions of the energy difference with a constant gamma and a constant V

# intendance cad many body energies
    size_basis = len(basis)

    mbenergies = np.zeros(size_basis)
    for i in range(size_basis):
        mbenergies[i] = np.sum(epsilon * basis[i,:]) # calculation of the energy of each many body state
    
    gamma_conserv = 2.
    
    print('Potential')
    pot = potential()
    pot = np.abs(pot)**2.
    pot = 2. * np.pi * pot / hbar**2.
    
    print('Lorentz weights calculation')
    lorentz_weights = lorentz(basis,mbenergies,N,A,ini_energy) # each gain and loss terms are given by lorentzian matrices

    Gain = lorentz_weights 
    Loss = np.transpose(Gain)
    print('Transition conserving energy ?')
    
# difference of many-body energies with energy of initial state

    diff_ini = mbenergies - ini_energy 
    diff_ini = np.abs(diff_ini)
    
    TRUTH = (diff_ini <= gamma_conserv)
#    print('TRUTH',TRUTH)

    for i in range(size_basis):
        Gain[:,i] = Gain[:,i] * TRUTH
        Loss[:,i] = Loss[:,i] * TRUTH 

    Gain, Loss = Gain * pot, Loss * pot

# i save the gain and loss terms to compute the associated characteristic times

    filename = PATH+"gl_terms"+str(A)+"_"+str(N)+"_"+str(M)+".dat"

# some lines of code to2 insure no problem arise if the new file do not already exists

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

#    f = open(filename, 'wb')
#    np.savetxt(f, Gain)
#    f.close()

    return Gain, Loss

def potential():
    pot = hbar
    return pot

def lorentz(basis,mbenergies,N,A,ini_energy):
# calculate the lorentzian weight associated to a jump

    size_basis = len(basis)
    gs_energy = mbenergies[0] # energy of the ground state
#    ini_energy = # energy of the initial state

    print('width of lorentz')
    gamma = width_lorentz(gs_energy,ini_energy)
    
    print('Lorentzian function')
    lorentz = np.zeros((len(basis),len(basis)))
    DE = mbenergies[np.newaxis,:] - mbenergies[:,np.newaxis]

    lorentz = 1. / (2. * np.pi) * gamma
    lorentz = lorentz / (DE**2. + gamma**2. / 4.) 
    
    return lorentz

def width_lorentz(gs_energy,ini_energy):
# calculation of width of the final states
# they are supposed to be exact, but since we don't know them we have to use a prescription to compute them.
# Parameters will be adjusted one day :)

    a = 1.
    b = 0.1
    DELTA = 1.

    diff = ini_energy - gs_energy
    diff = diff**2.
    gamma = a * (diff + DELTA) * np.exp(- b * diff)

    return gamma
