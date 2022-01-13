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
import random
from scipy import sparse

from module_fig_gen import *
from gl_terms import *
# here the energy of the many-body states is taken into account by considering that
# each site has a random energy between 0 and 1.
#Sites are in ascending order so that energy site 0 < ... < energy site A

hbar = 197.32705

# the total number of occupied states is conserved

# Number of 1p states
A = 15

# Number of occupied states
N = 6

# time step  and number of time step
Npas = 1500
dt = 0.001
Nsave = 1 # save date every Nsave time steps. Not implemented (flemme)

# we are interested in MpMh excitations:
M = 2

S = 0


# Melange of ground state and 1st 2p2h excited state as initial condition
# nu = 1 => first 2p2h state
# nu = 0 => groud state
nu = 1.

Time = np.linspace(0.,Npas * dt,Npas)

PATH = '/projet/pth/czuba/2021/tdhfbproj/master_eq_gamma_2/subspace' # path of the project, can be changed directly

#################################################
# Solver (= main function)
#################################################

def markov_solver(A,N,Npas,dt,nu):
# the markov equation solver

    epsilon = sp_ener(S,A)

    basis, shape_basis = gen_many_body_basis(A,N) # generation of the basis, shape of the basis is ((A choose N),A)

    size_basis = shape_basis[0]
    print(size_basis)
    basis = basis.toarray()

    mb_weights = ini_state(basis,nu) # choose the initial weights
    
    mb_energies = np.zeros(size_basis)
    for i in range(size_basis): # energy of each state of the basis
        mb_energies[i] = np.sum(epsilon * basis[i,:])

    ini_energy = np.sum(mb_weights * mb_energies) # initial total energy, shold be constant
    print('initial energy =', ini_energy)

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

    print('Gain and loss matrices')
    test = time.time()
    Gain, Loss = GL_terms(basis,mask,epsilon,PATH,A,N,M,ini_energy)
    test = time.time() - test
    print('Gain and loss matrices calculations =', test, 's')
    
# RK2 propagation : optimzed by using matrix multiplication
    print('RK2 ppgation')
    test = time.time()

    TIME, save_tot_ener, ener_deriv = [], [], np.zeros(len(basis))

    for it in range(Npas):

# t + dt/2
        RHSa = (np.matmul(Gain * mask, np.transpose(oldw)) - np.sum(Loss * oldw * mask, axis=0))

        newweights = oldw + dt / 2. * RHSa
# t + dt
        RHSb = (np.matmul(Gain * mask, np.transpose(newweights)) - np.sum(Loss * newweights * mask, axis=0))
        newweights = newweights + dt * RHSb

        oldw = newweights

        if it % (Nsave * 10) == 0:# testing the energy conservation
#            print(it)
            tot_energy = np.sum(newweights * mb_energies) # initial total energy, shold be constant
            ener_diff = np.abs(tot_energy - ini_energy)
#            if ener_diff >= 10.**(-3.):
#                print('ENERGY NOT CONSERVED', ener_diff, ener_diff / ini_energy)

        if it % Nsave == 0: # saving the files and testing different aspects of energy conservation
            maskb = np.ones(len(basis), dtype = 'bool')
            save_energy = np.vstack((save_energy,mb_energies*newweights*maskb))
            save_weights = np.vstack((save_weights,newweights))
            TIME.append(it * dt)
            save_tot_ener.append(np.sum(mb_energies*newweights))
            ener_deriv = np.vstack((ener_deriv,mb_energies*RHSa))

        
    test = time.time() - test
    print('Duration of propagation = ', test, 's')
# saving the figures
    print('saving data')
    save_data(save_weights,'weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    save_data(save_energy,'mb_energy'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    save_data(TIME,'time.dat')
    save_data(save_tot_ener,'ener_tot'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    save_data(ener_deriv,'deriv_ener'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    sp_weights(basis) # computation of the occupation numbers
    print(size_basis)

# showing some figures, and saving them
    print('construction of figs')
    n_epsilon()
    state_dens(basis)
    entropy()
    pass

#################################################
# sp energies
#################################################

def sp_ener(S,A):
    if S == 0. :
   #     epsilon = np.sort(np.random.rand(A)) * 10
        epsilon = np.arange(0,A) * 1.
        save_data(epsilon,'sp_energies'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    else:
        epsilon = np.genfromtxt(PATH+'sp_energies'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    print(epsilon)
    return epsilon

#################################################
# basis generation, this one is pretty advanced, took the algorithm from 
# https://stackoverflow.com/questions/6284396/permutations-with-unique-values
# probably the fastest way to generate a unique set of permutations in python
# up to A = 25 and N = 10 is reasonable
#################################################

class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1

def gen_many_body_basis(A,N):
# the basis
    print('basis construction')
    state = np.zeros(A)
    for i in range(N):
        state[i] = 1

    test = time.time()
    basis = list(perm_unique(list(state))) # constructs all UNIQUE permutations of N ones in a array of length A
#    basis = list(set(it.permutations((state)))) antiquated, not efficient enough
    basis = np.array(basis)
    size_basis = len(basis)
    test = time.time() - test
    print('duration =', test)
    print('basis done')

# the states must now be rearranged by ascending energy order
# this corresponds to ascending base 10 representation of the binary number that encodes a state
# i have to make the conversion, and then simply reorder it
# i use that instead of the energy because there can be degenerated states, and the sort will not work properly
    print('ordering of basis')

    test = time.time()
    ten_repr = np.zeros(size_basis)

    def tobin(x,s): # converts a real number to a binary one with each bin stored in a array
        return np.array([(x>>k)&1 for k in range(0,s)])

    for i in range(size_basis):
        ten_repr[i] = np.sum(2. ** np.arange(0,A) * basis[i,:])  # conversion binary numbers into base 10 number: unique label for all states

    ten_repr = np.sort(ten_repr.astype(int)) # states are rearranged bey ascending base 10 label, equivalent to ascending energy but without degenerescence

    for i in range(size_basis):
        basis[i,:] = tobin(ten_repr[i], A) # the basis in its binary representation is reconstructed, sorted as wished
    test = time.time() - test
    print('duration =', test)
    
    shape_basis = np.shape(basis)
    basis = sparse.csr_matrix(basis)
    print('basis ordered')
    return basis, shape_basis

#################################################
# choice of the initial state
#################################################


def ini_state(basis,nu):
# the many-body initial weights
# i decided to start the diffusion from the lowest 4QP excitation
# first i have to find it in the ordered basis then select it as an initial state
# differences between states and the ground state are computed
# the first one to have M -1 in this difference is selected, it will be the lowest MpMh excitation
# as wanted since the basis is ordered by ascending base 10 label


# nu is used to create a mix of the ground state and the 1st 2p2h excited state
    i, nbr_minus = 0, 0
    while nbr_minus != M:
        i = i + 1
        test = basis[i] - basis[0]
        nbr_minus = list(test).count(-1.)

# an excitation is MpMh is recognized if the difference between the 2 arrays yields M minus ones

    mb_weights = np.zeros(len(basis))
    mb_weights[i] = nu
    mb_weights[0] = 1. - nu

#    mb_weights[100] = 1.
#    print('ini_state = ', basis[100,:])
    mb_weights = mb_weights / np.sum(mb_weights) # renormalized

    print(i,basis[i])
#    print(mb_weights)
    return mb_weights

#################################################
# computation of the single particle (sp) occupation numbers 
# and many body entropy and sp entropy 
#################################################

def sp_weights(basis):
# construction of the occupation numbers of the single particle states
    weights = np.genfromtxt(PATH+'weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')

    basis = np.array(basis)
    sp_weights = np.zeros((np.int(Npas/Nsave)+1,A))

#    print(np.shape(basis),np.shape(weights),np.shape(sp_weights),len(basis))

    for i in range(A):
        for j in range(len(basis)):
#            print(i,j)
            if basis[j,i] == 1:
                sp_weights[:,i] = sp_weights[:,i] + weights[:,j]

    save_data(sp_weights,'sp_weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    print('sp occupation numbers done')
    pass

def entropy():
    weights = np.genfromtxt(PATH+'weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    entropy = - np.sum(weights * np.log(np.where(weights != 0., weights, 1.)), axis = 1)

    save_data(entropy,'mb_entropy'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')

    weights = np.genfromtxt(PATH+'sp_weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    spentropy = - np.sum(weights * np.log(np.where(weights != 0., weights, 1.)) + (1. - weights) * np.log(1. - weights), axis = 1)

    save_data(spentropy,'sp_entropy'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')

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

    weights = np.genfromtxt(PATH+'weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    TIME = Time#np.genfromtxt(PATH+'time.dat')

    for i in range(1,np.shape(weights)[1]):
        ax.plot(TIME,weights[:-1,i],'-', linewidth = 1)

#    ax.plot(TIME,np.sum(weights[:-1,:], axis = 1),'--', linewidth = 1) # is the total proba conserved ?
    print(np.sum(weights[-1,:]))
    print(weights[-1,:])
    ax.set_ylim([-0.01,0.1])
#    ax.set_xlim(0,0.1)
    fig_name = PATH+'weights'+str(A)+'_'+str(N)+'_'+str(M)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')
    show()

    pass

def sp_weights_fig():
# shows the occupation numbers of the single particle states
    labels = [r'$t$ (fm/c)', r'$n_{k}(t)$']

    ax = spec_fig(1, 1, size_fig, labels)

    spweights = np.genfromtxt(PATH+'sp_weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    TIME = Time

    print(np.sum(spweights[-1,:]),'lalalal')

    for i in range(1,np.shape(spweights)[1]):
        ax.plot(TIME,spweights[:-1,i],'-', linewidth = 1)

#    ax.plot(TIME,np.sum(weights[:-1,:], axis = 1),'--', linewidth = 1) # is the total proba conserved ?

#    ax.set_xlim(0,0.1)
    fig_name = PATH+'sp_weights'+str(A)+'_'+str(N)+'_'+str(M)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')
    show()

    pass

def energy_fig(a):
# shows the occupation numbers of the single particle states
# a = 0 : many body energies
# a = 1 : one body energies
    if a == 0:
        labels = [r'$t$ (fm/c)', r'$E_{\rm tot}(t)$ (MeV)']
    else:
        labels = [r'$t$ (fm/c)', r'$\varepsion_{k}(t)$ (MeV)']

    ax = spec_fig(1, 1, size_fig, labels)

    energies = np.genfromtxt(PATH+'mb_energy'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    TIME = Time

    weights = np.genfromtxt(PATH+'weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    ener_tot = np.genfromtxt(PATH+'ener_tot'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    deriv_ener = np.genfromtxt(PATH+'deriv_ener'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')

#    for i in range(np.shape(energies)[1]):
#        ax.plot(TIME,energies[:-1,i]/weights[:-1,i],'-', linewidth = 1)
#        ax.plot(TIME,deriv_ener[1:,i],'-', linewidth = 1)
#        print(deriv_ener[1,i])
#        print(np.shape(energies)[1],energies[1,i]/weights[1,i],energies[-1,i]/weights[-1,i])
    basis_size = np.shape(weights)[1]

    ax.plot(TIME[1:],np.sum(energies[1:-1],axis=1),'-', linewidth = 1)


    fig_name = PATH+'mb_energies'+str(A)+'_'+str(N)+'_'+str(M)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')
    show()

    pass

def entropy_fig():
# shows the sp entropy and mb entropy as functions of time

    TIME = Time

    labels = [r'$t$ (fm/c)', r'']

    ax = spec_fig(1, 1, size_fig, labels)

    entropy = np.genfromtxt(PATH+'mb_entropy'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    spentropy = np.genfromtxt(PATH+'sp_entropy'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')

    ax.plot(TIME,entropy[:-1],'-',color = 'black', linewidth = 1)
    ax.plot(TIME,spentropy[:-1],'--',color = 'black', linewidth = 1)
    ax.set_xlim(0,0.1)
    fig_name = PATH+'entropy'+str(A)+'_'+str(N)+'_'+str(M)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')

    show()

    pass

# density of many body states as a function of the energy
def state_dens(basis):
# construct the density of many body states and displays it using an histogram with bars of width e
#    labels = [r'$E_\alpha$ (MeV)', r'$\rho (E)$ (no units)']

    epsilon = np.genfromtxt(PATH+'sp_energies'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    labels = ['', '']

    ax = spec_fig(1, 1, size_fig, labels)

    basis = np.array(basis)
    size_basis = len(basis)

    mbenergies = np.zeros(size_basis)
    for i in range(size_basis):
        mbenergies[i] = np.sum(epsilon * basis[i,:]) # calculation of the energy of each many body state

    weights = np.genfromtxt(PATH+'weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
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
    print(wf)
    de = (max(existing_energies)-min(existing_energies)) / np.sum(occurrences) # gaffe, je considere que les etats sont regulierement espaces
    occurrences = occurrences / np.sum(occurrences * de)
#    print(wi,'lol')
#    print(wf,'lal')
#    print(weights)
    ax.bar(existing_energies, occurrences, color = 'black', alpha = 0.5, label = r'$\rho(E)$')
    ax.bar(existing_energies, wi, alpha = 0.5, color = 'blue', label = r'$a$')
    ax.bar(existing_energies, wf, alpha = 0.5, color = 'red', label = r'$b$')

    fig_name = PATH+'dens_state'+str(A)+'_'+str(N)+'_'+str(M)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')

    show()

    pass

def n_epsilon():
# histogram of the sp occupation numbers as a function of the sp energies
# initial config vs final, to display the thermalization of the system on a 1 body level

    labels = [r'$\varepsilon_k$ (MeV)', r'$n_k (\varepsilon_k)$ (no units)']
    ax = spec_fig(1, 1, size_fig, labels)

    spweights = np.genfromtxt(PATH+'sp_weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    sp_energies = np.genfromtxt(PATH+'sp_energies'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')

    ini_sp = spweights[0,:]
    final_sp = spweights[-1,:]

    print(sp_energies)
    print(spweights[0,:])
    print(spweights[-1,:])

    ax.bar(sp_energies, ini_sp, alpha = 0.5, color = 'blue')
    ax.bar(sp_energies, final_sp, alpha = 0.5, color = 'red')

    fig_name = PATH+'n_epsilon'+str(A)+'_'+str(N)+'_'+str(M)+'_'+str(Npas)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')

    show()
    pass

def fig_fermi_dirac():

    labels = [r'$\varepsilon_k$ (MeV)', r'$n_k (\varepsilon_k)$ (no units)']
    ax = spec_fig(1, 1, size_fig, labels)

    spweights = np.genfromtxt(PATH+'sp_weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    sp_energies = np.genfromtxt(PATH+'sp_energies'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    
    final_sp = spweights[-1,:]
    
    def fermi_dirac(x,beta,mu):
        return 1. / (1. + np.exp(beta * (x - mu)))
        
    popt, pcov = curve_fit(fermi_dirac, sp_energies, final_sp)
    print('fitted parameters',popt,np.sqrt(np.diag(pcov)))
    
    ax.bar(sp_energies, final_sp, alpha = 0.5, color = 'red')
    ax.plot(sp_energies, fermi_dirac(sp_energies,popt[0],popt[1]))
    text = [r'$\beta = $',r'$\mu = $']
    for i in range(len(popt)):
        ax.text(0.5,0.75-0.05*i,text[i]+str(round(popt[i], 3))+'$\pm$'+str(round(np.sqrt(np.diag(pcov)[i]), 3))+' MeV', transform=ax.transAxes, fontsize = 20)
    
    
    show()
    
    fig_name = PATH+'fitFD'+str(A)+'_'+str(N)+'_'+str(M)+'_'+str(Npas)+'_'+str(int(10.*nu))+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')
    
    fitted_params = np.zeros((len(popt),len(popt)))
    for i in range(len(popt)):
        for j in range(len(popt)):
            if j ==0:
                fitted_params[i,j] = popt[i]
            else:
                print(np.sqrt(np.diag(pcov)[i]))
                fitted_params[i,j] = np.sqrt(np.diag(pcov)[i])


    save_data(fitted_params,'fitFD_params'+str(A)+'_'+str(N)+'_'+str(M)+'_'+str(int(10.*nu))+'.dat')
    
    pass
    
def fig_fermi_dirac_params(a):
# if a = 0 then temperature
# if a = 1 then chemical potential
 
    if a == 0:
        y_axislabel = r'$k_{\rm B} T$ (MeV)'
        name = 'T'
    else:
        y_axislabel = r'$\mu$ (MeV)'
        name = 'mu'
    labels = [r'$N$ (no units)', y_axislabel]
    ax = spec_fig(1, 1, size_fig, labels)

    particle_number = [2,3,4,6,8,10]

    T, mu = [], []
    T_err, mu_err = [], []
    for value in particle_number:
        fitted_params = np.genfromtxt(PATH+'fitFD_params'+str(A)+'_'+str(value)+'_'+str(M)+'_'+str(int(10.*nu))+'.dat')
        T.append(1./fitted_params[0,0])
        T_err.append(fitted_params[0,1]/T[-1]**2.)
        mu.append(fitted_params[1,0])
        mu_err.append(fitted_params[1,1])
    
    if a == 0:
        ax.scatter(particle_number, T, color = 'black')
    else:
#        ax.scatter(particle_number, mu, color = 'black')
        ax.errorbar(particle_number, mu, yerr=mu_err, label='both limits (default)', color = 'black', fmt = 'o')

    show()
    
    fig_name = PATH+'fit'+name+str(A)+'_'+str(M)+'_'+str(Npas)+'_'+str(int(10.*nu))+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')
    
    pass
    
def fig_width_lorentz():
# fig of width attributed to each transition

    a = 1.
    b = 0.05
    DELTA = 1.

    epsilon = np.arange(0,A)

    gs_energy, max_energy = 0., 0.
    for i in range(N):
        gs_energy = gs_energy + epsilon[i] # ground state energy

    for i in range(A):
        max_energy = max_energy + epsilon[i]

    xener = np.linspace(gs_energy,max_energy,1000) 

    diff = xener - gs_energy
    diff = diff**2.
    gamma = a * (diff + DELTA) * np.exp(- b * diff)

    #########

    epsilon = np.genfromtxt(PATH+'sp_energies'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')
    labels = ['', '']

    ax = spec_fig(1, 1, size_fig, labels)

    basis, shape_basis = gen_many_body_basis(A,N)
    basis = basis.toarray()
    size_basis = shape_basis[0]

    mbenergies = np.zeros(size_basis)
    for i in range(size_basis):
        mbenergies[i] = np.sum(epsilon * basis[i,:]) # calculation of the energy of each many body state

    weights = np.genfromtxt(PATH+'weights'+str(A)+'_'+str(N)+'_'+str(M)+'.dat')

    existing_energies = np.array(list(set(mbenergies)))
    occurrences = np.zeros(len(existing_energies))
    for i in range(len(occurrences)):
        indexes = np.where(mbenergies == existing_energies[i])[0]
        occurrences[i] = len(indexes)

    de = (max(existing_energies)-min(existing_energies)) / np.sum(occurrences) # gaffe, je considere que les etats sont regulierement espaces
    occurrences = occurrences / np.sum(occurrences * de)

    ax.bar(existing_energies, occurrences, color = 'black', alpha = 0.5, label = r'$\rho(E)$')
    ax.plot(xener,gamma,color = 'black')
    ax.set_xlim(gs_energy,35)

    fig_name = 'gamma_vs_dens'+str(A)+'_'+str(N)+'_'+str(M)+'.pdf'
    plt.savefig(fig_name, dpi = 'figure')

    show()

    pass
    
    
    
    
