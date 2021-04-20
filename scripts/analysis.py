import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os.path
from os import path
import decimal

params = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 10,
    'legend.fontsize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    'lines.linewidth':1,
    "patch.edgecolor": "black"
}

plt.rcParams.update(params)
plt.style.use('seaborn-deep')

global_seed = 9999

def colors_from_Ls(Ls, c="Blues"):
    cmap = plt.get_cmap(c)
    colors = [cmap(i) for i in np.linspace(0.3,1,len(Ls))]
    return colors

def observables_json(L, beta, seed=global_seed):
    # template: L=12_beta=0.3_seed=1234_observables.json
    if int(beta) == beta:
        beta = int(beta)

    path = '../examples/data/L={}_beta={}_seed={}_observables.json'.format(L, beta, seed)
    return path

def observables_mcsteps(L, beta):
    # template: energies_J=1.0_B=0.0_beta=3.0_dims=(16, 16).txt
    energies = np.loadtxt('../examples/data/energies_J=1.0_B=0.0_beta={}_dims=({}, {}).txt'.format(beta, L, L))
    energies_sqr = np.loadtxt('../examples/data/energies_sqr_J=1.0_B=0.0_beta={}_dims=({}, {}).txt'.format(beta, L, L))
    mags = np.loadtxt('../examples/data/magnetization_J=1.0_B=0.0_beta={}_dims=({}, {}).txt'.format(beta, L, L))
    mags_sqr = np.loadtxt('../examples/data/sqr_magnetization_J=1.0_B=0.0_beta={}_dims=({}, {}).txt'.format(beta, L, L))
    return energies, energies_sqr, mags, mags_sqr

def partA_plots():
    fig = plt.figure(figsize=(5,2.5))
    for beta in [1.0, 3.0]:
        energies, _ , mags, __ = observables_mcsteps(16, beta)
        plt.plot(np.arange(len(energies)), energies, label=r'$\beta = {}$'.format(beta))
    
    plt.xlabel("MC Step")
    plt.ylabel(r'$\frac{\langle E \rangle}{N}$', rotation=0)
    plt.xlim(0,500)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../examples/figures/partA.pdf', dpi=500)

def partB_plots():
    betas = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.5, 1.0, 2.0, 5.0])
    mags = np.zeros(len(betas))
    abs_mags = np.zeros(len(betas))
    energies = np.zeros(len(betas))

    fig = plt.figure(figsize=(5,2.5))
    for (i,beta) in enumerate(betas):
        path = observables_json(16, beta)
        with open(path) as f:
            contents = json.load(f)
        
        mags[i] = contents["magnetization"]["mean"]
        abs_mags[i] = contents["abs_magnetization"]["mean"]
        energies[i] = contents["energy"]["mean"]

    T = betas**(-1)
    plt.plot(T, mags, label=r'$\frac{\langle M \rangle}{N}$') 
    plt.plot(T, abs_mags, label=r'$\frac{\langle |M| \rangle}{N}$') 
    plt.plot(T, energies, label=r'$\frac{\langle E \rangle}{N}$')
    
    plt.xlabel(r'$T$')
    plt.xlim(0,20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../examples/figures/partB.pdf', dpi=500)

def partC_plots():
    num_sites = 16*16
    betas_coarse = np.array([0.2])
    betas_fine = np.arange(0.3, 0.601, 0.001)
    betas = np.sort(np.concatenate([betas_coarse, betas_fine]))
    betas *= 1000
    betas = np.round(betas)/1000

    specific_heats = np.zeros(len(betas))

    fig = plt.figure(figsize=(5,2.5))
    for (i,beta) in enumerate(betas):
        path = observables_json(16, beta)
        with open(path) as f:
            contents = json.load(f)
        
        # energy is by default per site, mult by num sites to get the desired value
        energy = contents["energy"]["mean"]*num_sites
        energy_sqr = contents["sqr_energy"]["mean"]*(num_sites**2)
        specific_heats[i] = (energy_sqr - energy**2.)*(beta**2.)

    T = betas**(-1)
    Tc_numerical = T[np.where(specific_heats == np.amax(specific_heats))]
    Tc_exact = 2./(np.log(1+np.sqrt(2.)))
    print("Numerical Tc: ", Tc_numerical)
    print("Exact Tc: ", Tc_exact)

    plt.plot(T, specific_heats)
    plt.axvline(x=Tc_numerical, ls='--', color='purple', label=r'$T_{c,\mathrm{MC}}$')
    plt.axvline(x=Tc_exact, ls='--', color='grey', label=r'$T_{c,\mathrm{exact}}$')
    plt.xlabel(r'$T$')
    plt.ylabel(r'$C_v$', rotation=0)
    plt.xlim(1.7, 3.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../examples/figures/partC.pdf', dpi=500)

def partD_plots_nocollapse():
    betas_coarse = np.array([0.2])
    betas_fine = np.arange(0.3, 0.601, 0.001)
    betas = np.sort(np.concatenate([betas_coarse, betas_fine]))
    betas *= 1000
    betas = np.round(betas)/1000
    T = betas**(-1)

    Ls = [12,16,20,24]
    #Ls = [12,16]

    chis = np.zeros((len(Ls), len(betas)))
    chis_wrong = np.zeros((len(Ls), len(betas)))

    colours = colors_from_Ls(Ls, c="Greens")

    # no data collapse
    fig = plt.figure(figsize=(5,2.5))
    for (j,L) in enumerate(Ls):
        num_sites = L**2.
        for (i,beta) in enumerate(betas):
            path = observables_json(L, beta)
            with open(path) as f:
                contents = json.load(f)
            
            # magnetization is by default per site, mult by num sites to get the desired value
            abs_mag = contents["abs_magnetization"]["mean"]*num_sites
            mag_sqr = contents["sqr_magnetization"]["mean"]*(num_sites**2.)
            chis[j,i] = (mag_sqr - abs_mag**2.)*beta
            chis_wrong[j,i] = contents["magnetization"]["variance"]*beta*(num_sites**2.)

        plt.plot(T, chis[j,:], label=r'$L = {}$'.format(L), c=colours[j])

    Tc_exact = 2./(np.log(1+np.sqrt(2.)))
    plt.axvline(x=Tc_exact, ls='--', color='grey', label=r'$T_{c,\mathrm{exact}}$')

    plt.xlabel(r'$T$')
    plt.ylabel(r'$\chi$', rotation=0)

    plt.legend()
    plt.tight_layout()
    plt.savefig('../examples/figures/partD_nocollapse.pdf', dpi=500)

    # the wrong plot that uses variance of M instead
    fig = plt.figure(figsize=(5,2.5))
    for (j,L) in enumerate(Ls):
        plt.plot(T, chis_wrong[j,:], label=r'$L = {}$'.format(L), c=colours[j])

    plt.axvline(x=Tc_exact, ls='--', color='grey', label=r'$T_{c,\mathrm{exact}}$')

    plt.xlabel(r'$T$')
    plt.ylabel(r'$\tilde{\chi}$', rotation=0)

    plt.legend()
    plt.savefig('../examples/figures/partD_nocollapse_variance.pdf', dpi=500)

def partD_plots_collapse():
    betas_coarse = np.array([0.2])
    betas_fine = np.arange(0.3, 0.601, 0.001)
    betas = np.sort(np.concatenate([betas_coarse, betas_fine]))
    betas *= 1000
    betas = np.round(betas)/1000

    Tc_exact = 2./(np.log(1+np.sqrt(2.)))
    T = betas**(-1)
    t = (T-Tc_exact)/Tc_exact

    Ls = [12,16,20,24]
    #Ls = [12,16]

    chis = np.zeros((len(Ls), len(betas)))

    colours = colors_from_Ls(Ls, c="Greens")

    # gamma and nu sweep
    for gamma in np.arange(1.74, 1.76, 0.005):
        for nu in np.arange(0.99, 1.01, 0.01):
            print(gamma, nu)
            # no data collapse
            fig, ax = plt.subplots(figsize=(5,2.5))
            left, bottom, width, height = [0.43, 0.25, 0.3, 0.3]
            ax2 = fig.add_axes([left, bottom, width, height])
            for (j,L) in enumerate(Ls):
                num_sites = L**2.
                for (i,beta) in enumerate(betas):
                    path = observables_json(L, beta)
                    with open(path) as f:
                        contents = json.load(f)
                    
                    # magnetization is by default per site, mult by num sites to get the desired value
                    abs_mag = contents["abs_magnetization"]["mean"]
                    mag_sqr = contents["sqr_magnetization"]["mean"]
                    chis[j,i] = (mag_sqr - abs_mag**2.)*beta*num_sites

                ax.plot(t*(L**(1./nu)), chis[j,:]*(L**(-gamma/nu)), label=r'$L = {}$'.format(L), c=colours[j])
                ax2.plot(t*(L**(1./nu)), chis[j,:]*(L**(-gamma/nu)), label=r'$L = {}$'.format(L), c=colours[j])
                ax2.set_xlim(0.5,1.5)
                ax2.set_ylim(0.04,0.046)

            ax.axvline(x=0, ls='--', color='grey', label=r'$T_{c,\mathrm{exact}}$')

            ax.set_xlabel(r'$t L^{1 / \nu}$')
            ax.set_ylabel(r'$\chi L^{-\gamma / \nu}$', rotation=0)
            ax.set_title(r'$\gamma = {}$, $\nu = {}$'.format(np.round(gamma*1000)/1000, np.round(nu*1000)/1000))
            ax.set_xlim(-1, 3)
            ax.yaxis.set_label_coords(-0.18,0.5)

            plot_name = 'gamma=%.3f' % gamma
            plot_name += '_nu=%.3f' % nu
            plot_name += ".pdf"

            ax.legend(loc=(1.01,0.45))
            #ax.tight_layout()
            fig.savefig("../examples/figures/collapse/"+plot_name, bbox_inches='tight', dpi=500)

#partA_plots()
#partB_plots()
#partC_plots()
#partD_plots_nocollapse()
partD_plots_collapse()