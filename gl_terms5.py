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

def GL_terms(BASIS_DIC,PATH,A,N,M,ini_energy,gs_energy):
# the gain and loss matrices involved in the master equation
# the matrix elements are chosen to be lorentzian functions of the energy difference with a constant gamma and a constant V

    print('Gain and loss matrices')

# intendance cad many body energies

    print('Potential')
    pot = potential()
    pot = 2. * np.pi * np.abs(pot)**2. / hbar**2.

    print('Lorentz weights calculation')
    lorentz_weights = lorentz(BASIS_DIC,N,A,ini_energy,gs_energy) # each gain and loss terms are given by lorentzian matrices

    Gain = lorentz_weights
    Loss = np.transpose(Gain)

    Gain, Loss = Gain * pot, Loss * pot

# i save the gain and loss terms to compute the associated characteristic times

#    filename = PATH+"gl_terms"+str(A)+"_"+str(N)+"_"+str(M)+".dat"

# some lines of code to2 insure no problem arise if the new file do not already exists

#    if not os.path.exists(os.path.dirname(filename)):
#        try:
#            os.makedirs(os.path.dirname(filename))
#        except OSError as exc: # Guard against race condition
#            if exc.errno != errno.EEXIST:
#                raise

#    f = open(filename, 'wb')
#    np.savetxt(f, Gain)
#    f.close()

    return Gain, Loss

def potential():
    pot = hbar
    return pot

def lorentz(BASIS_DIC,N,A,ini_energy,gs_energy):
# calculate the lorentzian weight associated to a jump

    size_basis = len(BASIS_DIC)    
    size_possible_trans = len(BASIS_DIC[0][3])
    # energy of the ground state

    print('width of lorentz')
    gamma = width_lorentz(gs_energy,ini_energy)

    print('Lorentzian function')
    lorentz = np.zeros((size_basis,size_possible_trans))
    # difference between the energy of each state and the energy of the states they may interact with through a MpMh
    for i in range(size_basis):
        for j in range(size_possible_trans):
            index = BASIS_DIC[i][3][j]
            DE = BASIS_DIC[i][1] - BASIS_DIC[index][1]

            lorentz[i,j] = 1. / (2. * np.pi) * gamma / (DE**2. + gamma**2. / 4.)

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
