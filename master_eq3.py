# math import
import math as ma
import numpy as np
import scipy as sc
import itertools as it
from scipy.optimize import curve_fit
# path management import
import os
import errno
from IPython.display import *
import time

from module_fig_gen import *
from gl_terms import *
# here the energy of the many-body states is taken into account by considering that
# |11100...> has total energy of 1 * 0 * epsilon + 1 * 1 * epsilon + 1 * 2 * epsilon + 0 * 3 * espilon + ...

hbar = 197.32705

# the total number of occupied states is conserved

# Number of 1p states
A = 12

# Number of occupied states
N = 2

# time step  and number of time step
Npas = 2000
dt = 0.001
Nsave = 1 # save date every Nsave time steps. Not implemented (flemme)

# we are interested in MpMh excitations:
M = 2

# energy of the lowest single particle level => all sp levels are separated by epsilon
epsilon = 1.

PATH = "/projet/pth/czuba/2021/tdhfbproj/master_eq3/" # path of the project, can be changed directly

#################################################
# Solver (= main function)
#################################################

def markov_solver(A,N,Npas,dt):
# the markov equation solver

    basis = gen_many_body_basis(A,N) # generation of the basis
    basis = np.array(basis)
    size_basis = len(basis)
    print(np.shape(basis),len(basis))
    mb_weights = ini_state(basis) # choose the initial weights

    mb_energies = np.zeros(size_basis)
    for i in range(size_basis): # energy of each state of the basis
        mb_energies[i] = np.sum(np.arange(0,A) * basis[i,:]) * epsilon
#        print(np.arange(0,A) * basis[i,:],np.sum(np.arange(0,A) * basis[i,:]))
    ini_energy = np.sum(mb_weights * mb_energies) # initial total energy, shold be constant
    print('total energy =', ini_energy)

# initialization

    newweights = np.copy(mb_weights)
    oldw = np.copy(mb_weights)
    save_weights = np.copy(newweights)

    save_energy = newweights * mb_energies

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

    test = time.time()
    lorentz_weights = lorentz(basis,epsilon,N,A) # each gain and loss terms are given by lorentzian matrices, assuming V = 1

# then i put it into the gain and loss matrices
# note here that energies do not change with time, therefore the system is not driven by an external time dependent field in any way !
# the interaction is considered to be a random gaussian number
    Gain, Loss = GL_terms(basis,mask,lorentz_weights,epsilon,PATH)
    test = time.time() - test
    print('Gain and loss terms calculations =', test, 's')
# RK2 propagation : optimzed by using matrix multiplication
    test = time.time()

    TIME, save_tot_ener, ener_deriv = [], [], np.zeros(len(basis))

    for it in range(Npas):

#        potential = np.random.normal(loc=0.0, scale=4.0 * epsilon,size=np.shape(Gain))
        potential = 2. * np.pi #/ hbar**2.# * 10**6.# np.abs(potential)**2.
        Gain2, Loss2 = potential * Gain, potential * Loss

# t + dt/2
        RHSa = (np.matmul(Gain2 * mask, np.transpose(oldw)) - np.sum(Loss2 * oldw * mask, axis=0))

        newweights = oldw + dt / 2. * RHSa
# t + dt
        RHSb = (np.matmul(Gain2 * mask, np.transpose(newweights)) - np.sum(Loss2 * newweights * mask, axis=0))
        newweights = newweights + dt * RHSb

        oldw = newweights

        if it % (Nsave * 10) == 0:# testing the energy conservation
#            print(it)
            tot_energy = np.sum(newweights * mb_energies) # initial total energy, shold be constant
            ener_diff = np.abs(tot_energy - ini_energy)
#            if ener_diff >= 10.**(-3.):
#                print('ENERGY NOT CONSERVED', ener_diff, ener_diff / ini_energy)

        if it % Nsave == 0: # saving the files and testing different aspects of energy conservation
            maskb = np.ones(len(basis), dtype = 'bool' )
#            maskb[7] = False
            save_energy = np.vstack((save_energy,mb_energies*newweights*maskb))
#            print(mb_energies*newweights*maskb)
            save_weights = np.vstack((save_weights,newweights))
            TIME.append(it * dt)
            save_tot_ener.append(np.sum(mb_energies*newweights))
            ener_deriv = np.vstack((ener_deriv,mb_energies*RHSa))

    test = time.time() - test
    print('Duration of propagation = ', test, 's')
# saving the figures
    save_data(save_weights,'weights.dat')
    save_data(save_energy,'mb_energy.dat')
    save_data(TIME,'time.dat')
    save_data(save_tot_ener,'ener_tot.dat')
    save_data(ener_deriv,'deriv_ener.dat')
    sp_weights(basis) # computation of the occupation numbers
    print(size_basis)


# showing some figures, and saving them
    n_epsilon()
#    P_E(basis)
    state_dens(basis)
    pass

#################################################
# basis generation and choosing of the initial state
#################################################

def gen_many_body_basis(A,N):
# the basis
    print('basis construction')
    state = np.zeros(A)
    for i in range(N):
        state[i] = 1

    basis = list(set(it.permutations((state)))) # constructs all UNIQUE permutations of N ones in a array of length A
    basis = np.array(basis)
# the states must now be rearranged by ascending energy order
# this corresponds to ascending base 10 representation of the binary number that encodes a state
# i have to make the conversion, and then simply reorder it
# i use that instead of the energy because there can be degenerated states, and the sort will not work properly
    size_basis = len(basis)
    ten_repr = np.zeros(size_basis)
    print('basis done')

    def tobin(x,s): # converts a real number to a binary one with each bin stored in a array
        return np.array([(x>>k)&1 for k in range(0,s)])

    for i in range(size_basis):
        ten_repr[i] = np.sum(2. ** np.arange(0,A) * basis[i,:])  # conversion binary numbers into base 10 number: unique label for all states

#    print(ten_repr[0],'lol')
    ten_repr = np.sort(ten_repr.astype(int)) # states are rearranged bey ascending base 10 label, equivalent to ascending energy but without degenerescence
#    print(ten_repr[0],'lal')

    for i in range(size_basis):
        basis[i,:] = tobin(ten_repr[i], A) # the basis in its binary representation is reconstructed, sorted as wished

    print('basis rearranged')
    return basis

def ini_state(basis):
# the many-body initial weights
# i decided to start the diffusion from the lowest 4QP excitation
# first i have to find it in the ordered basis then select it as an initial state
# differences between states and the ground state are computed
# the first one to have M -1 in this difference is selected, it will be the lowest MpMh excitation
# as wanted since the basis is ordered by ascending base 10 label

    i, nbr_minus = 0, 0
    while nbr_minus != M:
        i = i + 1
        test = basis[i] - basis[0]
        nbr_minus = list(test).count(-1.)

# an excitation is MpMh is recognized if the difference between the 2 arrays yields M minus ones

    mb_weights = np.zeros(len(basis))
    mb_weights[i] = 1.
    mb_weights = mb_weights / np.sum(mb_weights) # renormalized

    print(i,basis[i])
#    print(mb_weights)
    return mb_weights

def sp_weights(basis):
# construction of the occupation numbers of the single particle states
    weights = np.genfromtxt("weights.dat")

    basis = np.array(basis)
    sp_weights = np.zeros((np.int(Npas/Nsave)+1,A))

#    print(np.shape(basis),np.shape(weights),np.shape(sp_weights),len(basis))

    for i in range(A):
        for j in range(len(basis)):
#            print(i,j)
            if basis[j,i] == 1:
                sp_weights[:,i] = sp_weights[:,i] + weights[:,j]

    save_data(sp_weights,'sp_weights.dat')
    print('sp occupation numbers done')
    pass

#################################################
# saving files
#################################################

def save_data(data,name):

    filename = PATH+str(name)

# some lines of code to insure no problem arise if the new file do not already exists

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    f = open(filename, 'wb')
    np.savetxt(f, data)
    f.close()
    pass

#################################################
# showing results
#################################################

def weights_fig():
# shows the many-body weights
    labels = [r'$t$ (fm/c)', r'$P_{\alpha}(t)$']

    ax = spec_fig(1, 1, size_fig, labels)

    weights = np.genfromtxt("weights.dat")
    TIME = np.genfromtxt("time.dat")

    for i in range(1,np.shape(weights)[1]):
        ax.plot(TIME,weights[:-1,i],'-', linewidth = 1)

#    ax.plot(TIME,np.sum(weights[:-1,:], axis = 1),'--', linewidth = 1) # is the total proba conserved ?
    print(np.sum(weights[-1,:]))
    print(weights[-1,:])
    ax.set_ylim([-0.1,1.05])

    fig_name = 'weights'+str(A)+'_'+str(N)+'_'+str(M)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')
    show()

    pass

def sp_weights_fig():
# shows the occupation numbers of the single particle states
    labels = [r'$t$ (fm/c)', r'$n_{k}(t)$']

    ax = spec_fig(1, 1, size_fig, labels)

    spweights = np.genfromtxt("sp_weights.dat")
    TIME = np.genfromtxt("time.dat")

    print(np.sum(spweights[-1,:]))

    for i in range(1,np.shape(spweights)[1]):
        ax.plot(TIME,spweights[:-1,i],'-', linewidth = 1)

#    ax.plot(TIME,np.sum(weights[:-1,:], axis = 1),'--', linewidth = 1) # is the total proba conserved ?
    print(spweights[-1,:])

    fig_name = 'sp_weights'+str(A)+'_'+str(N)+'_'+str(M)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')
    show()

    pass

def energy_fig(a):
# shows the occupation numbers of the single particle states
# a = 0 : many body energies
# a = 1 : one body energies
    if a == 0:
        labels = [r'$t$ (fm/c)', r'$E_{\alpha}(t)$ (MeV)']
    else:
        labels = [r'$t$ (fm/c)', r'$\varepsion_{k}(t)$ (MeV)']

    ax = spec_fig(1, 1, size_fig, labels)

    energies = np.genfromtxt("mb_energy.dat")
    TIME = np.genfromtxt("time.dat")

    weights = np.genfromtxt("weights.dat")
    ener_tot = np.genfromtxt("ener_tot.dat")
    deriv_ener = np.genfromtxt('deriv_ener.dat')

#    for i in range(np.shape(energies)[1]):
#        ax.plot(TIME,energies[:-1,i]/weights[:-1,i],'-', linewidth = 1)
#        ax.plot(TIME,deriv_ener[1:,i],'-', linewidth = 1)
#        print(deriv_ener[1,i])
#        print(np.shape(energies)[1],energies[1,i]/weights[1,i],energies[-1,i]/weights[-1,i])
    basis_size = np.shape(weights)[1]

    ax.plot(TIME[1:],np.sum(energies[1:-1],axis=1),'-', linewidth = 1)

    fig_name = 'mb_energies'+str(A)+'_'+str(N)+'_'+str(M)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')
    show()

    pass

def entropy():
# shows the sp entropy and mb entropy as functions of time

    TIME = np.genfromtxt("time.dat")

    labels = [r'$t$ (fm/c)', r'$\mathcal{S}(t)$ ($k_{\rm B}$ units)']

    ax = spec_fig(1, 1, size_fig, labels)

    weights = np.genfromtxt(PATH+"weights.dat")
    entropy = - np.sum(weights * np.log(np.where(weights != 0., weights, 1.)), axis = 1)
#    entropy = - np.sum(weights, axis = 1)

    weights = np.genfromtxt(PATH+"sp_weights.dat")
#    spentropy = - np.sum(weights, axis = 1)
    spentropy = - np.sum(weights * np.log(np.where(weights != 0., weights, 1.)) + (1. - weights) * np.log(1. - weights), axis = 1)

    ax.plot(TIME,entropy[:-1],'-',color = 'black', linewidth = 1)
    ax.plot(TIME,spentropy[:-1],'--',color = 'black', linewidth = 1)

#    ax.plot(TIME,np.sum(weights[:-1,:], axis = 1),'--', linewidth = 1) # is the total proba conserved ?
#    print(np.sum(weights[-1,:]))

    show()

    pass

# density of many body states as a function of the energy
def state_dens(basis):
# construct the density of many body states and displays it using an histogram with bars of width e
#    labels = [r'$E_\alpha$ (MeV)', r'$\rho (E)$ (no units)']
    labels = ['', '']

    ax = spec_fig(1, 1, size_fig, labels)

    basis = np.array(basis)
    size_basis = len(basis)

    mbenergies = np.zeros(size_basis)
    for i in range(size_basis):
        mbenergies[i] = np.sum(epsilon * np.array(np.where(np.array(basis[i]) == 1.)).flatten()) # calculation of the energy of each many body state

    weights = np.genfromtxt(PATH+"weights.dat")
    ini_weights = weights[0,:]
    final_weights = weights[-1,:]

    existing_energies = np.array(list(set(mbenergies)))
    occurrences = np.zeros(len(existing_energies))
    wi = np.zeros(len(existing_energies))
    wf = np.zeros(len(existing_energies))
    for i in range(len(occurrences)):
        indexes = np.where(mbenergies == existing_energies[i])[0]
        for j in indexes:
            wi[i] = wi[i] + ini_weights[j]
            wf[i] = wf[i] + final_weights[j]
        occurrences[i] = len(indexes)

    de = (max(existing_energies)-min(existing_energies)) / np.sum(occurrences) # gaffe, je considere que les etats sont regulierement espaces
    occurrences = occurrences / np.sum(occurrences * de)
    print(wi,'lol')
    print(wf,'lal')
    ax.bar(existing_energies, occurrences, color = 'black', alpha = 0.5, label = r'$\rho(E)$')
    ax.bar(existing_energies, wi, alpha = 0.5, color = 'blue', label = r'$a$')
    ax.bar(existing_energies, wf, alpha = 0.5, color = 'red', label = r'$b$')

    fig_name = 'dens_state'+str(A)+'_'+str(N)+'_'+str(M)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')

    show()

    pass

def n_epsilon():
# histogram of the sp occupation numbers as a function of the sp energies
# initial config vs final, to display the thermalization of the system on a 1 body level

    labels = [r'$\varepsilon_k$ (MeV)', r'$n_k (\varepsilon_k)$ (no units)']
    ax = spec_fig(1, 1, size_fig, labels)

    spweights = np.genfromtxt("sp_weights.dat")
    sp_energies = np.zeros(A)
    for i in range(A):
        sp_energies[i] = i * epsilon

    ini_sp = spweights[0,:]
    final_sp = spweights[-1,:]

    print(sp_energies)
    print(spweights[0,:])
    print(spweights[-1,:])

    ax.bar(sp_energies, ini_sp, alpha = 0.5, color = 'blue')
    ax.bar(sp_energies, final_sp, alpha = 0.5, color = 'red')

    fig_name = 'n_epsilon'+str(A)+'_'+str(N)+'_'+str(M)+'_'+str(Npas)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')

    show()
    pass

def P_E(basis):
# histogram of the many body occupation numbers as a function of the many body energies
# initial config vs final, to display the thermalization of the system on a many body level

    labels = [r'$E_\alpha$ (MeV)', r'$P_\alpha$ (no units)']
    ax = spec_fig(1, 1, size_fig, labels)

    weights = np.genfromtxt(PATH+"weights.dat")

    ini_weights = weights[0,:]
    final_weights = weights[-1,:]

    size_basis = np.shape(weights)[1]
    mb_energies = np.zeros(size_basis)
    for i in range(size_basis): # energy of each state of the basis
        mb_energies[i] = np.sum(np.arange(0,A) * basis[i,:]) * epsilon

    ax.bar(mb_energies, ini_weights, alpha = 0.5, color = 'blue')
    ax.bar(mb_energies, final_weights, alpha = 0.5, color = 'red')

    fig_name = 'P_E'+str(A)+'_'+str(N)+'_'+str(M)+'_'+str(Npas)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')

    show()
    pass
