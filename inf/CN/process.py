import numpy as np
from scipy.sparse import diags
import scipy as sp
import scipy.linalg
import scipy.interpolate
import scipy.optimize
from scipy.sparse.linalg import gmres
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
from thermo.chemical import Chemical
import h5py

#===============================================================================
# Setup
#===============================================================================

# Thermal properties
def rho_f(T): # kg/m^3 --> kg/cc
    return Chemical('dioxouranium',T=T).rho*1E-6
def rho_w(T): # kg/m^3 --> kg/cc
    return Chemical('water',T=T,P=15.5E6).rho*1E-6
def cp_f(T): # J/kg-K
    return Chemical('dioxouranium',T=T).Cp
def cp_w(T): # J/kg-K
    return Chemical('water',T=T,P=15.5E6).Cp

# Geometry
Rf = 0.5   # cm
d  = 1.3   # cm
Z  = 360.0 # cm
Af = Z*2*np.pi*Rf
Vf = Z*np.pi*Rf**2

# Operational parameters
Pow   = 200.0 # W
h     = 0.2   # W/cm^2-K

# Neutronic properties
kappa_f  = 191.4*1.6022e-13 # MeV --> J
v        = 2.2E5            # cm/s
nu       = 2.43
xi       = 0.8097
Tf       = [565.0, 1565.0, 565.0, 1565.0]
Tw       = [565.0,  565.0, 605.0,  605.0]
SigmaT   = [0.655302, 0.653976, 0.61046]
SigmaS   = [0.632765, 0.631252, 0.589171]
nuSigmaF = [xi*0.0283063, xi*0.0277754, xi*0.0265561]
SigmaT.append(SigmaT[2]+(SigmaT[1]-SigmaT[0]))
SigmaS.append(SigmaS[2]+(SigmaS[1]-SigmaS[0]))
nuSigmaF.append(nuSigmaF[2]+(nuSigmaF[1]-nuSigmaF[0]))

# Set XS interpolators
SigmaT   = sp.interpolate.interp2d(Tf,Tw,SigmaT)
SigmaS   = sp.interpolate.interp2d(Tf,Tw,SigmaS)
nuSigmaF = sp.interpolate.interp2d(Tf,Tw,nuSigmaF)
def SigmaA(T,Tw):
    return SigmaT(T,Tw) - SigmaS(T,Tw)
def SigmaF(T,Tw):
    return nuSigmaF(T,Tw)/nu
def keff(T,Tw,SigmaL):
    return nuSigmaF(T,Tw)/(SigmaL + SigmaA(T,Tw))

# Numerical parameters
tol = 1E-9

# Simulation length
T = 60 # s

# Calculate IC (SS)
Tw     = 565.0
Tf_ss  = Pow/Af/h + Tw
SigmaL = nuSigmaF(Tf_ss,Tw) - SigmaA(Tf_ss,Tw)
phi_ss = Pow/kappa_f/SigmaF(Tf_ss,Tw)
k_ic   = keff(Tf_ss,Tw,SigmaL)
rho_ic = (k_ic-1)/k_ic
pow_ic = Pow

# Set perturbation
#h = h*0.5
phi_ss *= 10.0
pow_ic  = kappa_f*SigmaF(Tf_ss,Tw)*phi_ss

#===============================================================================
# Solve TD
#===============================================================================

K_list = [6,8,10,20,40,60,80,100,200,400,600,800,1000,2000,4000,6000,10000]

for K in K_list:
    # Time grid
    dt    = T/K*0.5 # CN average

    #===========================================================================
    # Containers
    #===========================================================================

    # Storage
    store_phi = np.zeros(K+1)
    store_Tf  = np.zeros(K+1)
    store_rho = np.zeros(K+1)
    store_pow = np.zeros(K+1)

    #===========================================================================
    # Solve
    #===========================================================================

    Tf_prv = 0.0
    q     = 0.0
    def TT(T):
        qs = Af*h*(T-Tw)
        r  = (T-Tf_prv)/dt - (q-qs)/(rho_f(T)*Vf*cp_f(T))
        return r
    
    # Initial condition
    phi = phi_ss
    Tf  = Tf_ss
    
    # Store
    store_phi[0] = phi
    store_rho[0] = rho_ic
    store_pow[0] = pow_ic
    store_Tf[0]  = Tf

    # March in time
    for k in range(K):
        # Set previous solution
        phi_prv = phi
        Tf_prv  = Tf

        # Initiate piccard iteration
        err = 1.0
        while err > tol:
            # Store old solution
            phi_old = phi
            Tf_old  = Tf
            
            # Solve neutron
            phi = phi_prv/v/dt/(1/v/dt-nuSigmaF(Tf,Tw)+SigmaA(Tf,Tw)+SigmaL)

            # Error neutron
            err_phi = abs(phi - phi_old)/phi
                
            # Solve heat transfer
            q = kappa_f*SigmaF(Tf,Tw)*phi
            Tf = sp.optimize.fsolve(TT,Tf,xtol=tol*1E-2)

            # Error T and all
            err_Tf = abs(Tf - Tf_old)/Tf
            err       = max(err_Tf,err_phi)
            print(K,k,err_phi,err_Tf)

        
        # CN step
        phi = 2*phi - phi_prv
        Tf  = 2*Tf  - Tf_prv
        
        # Store solution
        store_phi[k+1] = phi
        store_Tf[k+1]  = Tf
        store_pow[k+1] = kappa_f*SigmaF(Tf,Tw)*phi
        kf             = keff(Tf,Tw,SigmaL)
        store_rho[k+1] = (kf-1)/kf
        print(K,k,"keff",kf,(kf-1)/kf)

    t = np.linspace(0.0,T,K+1)
    '''
    plt.plot(t,store_rho)
    plt.show()
    
    plt.plot(t,store_pow)
    plt.show()

    plt.plot(t,store_phi)
    plt.show()
    
    plt.plot(t,store_Tf)
    plt.show()
    '''
    np.savez('K=%i'%K,t=t,phi=store_phi,pow=store_pow,Tf=store_Tf,rho=store_rho)