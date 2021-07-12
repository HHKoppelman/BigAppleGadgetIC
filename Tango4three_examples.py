import numpy as np
import agama

# internally use N-body units:  length=1 kpc, velocity=1 km/s, mass=1 Msun.
agama.setUnits(length=1, velocity=1, mass=1)

def generate_LMC(Nbody=1_000_000):
    """
    Generates a simple, single component, spherical LMC.
    
    Based on code provided by Vasiliev+2019 to model Sgr.
    https://arxiv.org/abs/2009.10726
    
    The LMC is described by a NFW profile with a mass of 
    1.5e11 Msun and a cut-off radius at 10 times the scale radius.
    
    """
    Mlmc = 1.5e11  # mass in physical units [Msun]

    # internally use N-body units:  length=1 kpc, velocity=1 km/s, mass=1 Msun.
    Rlmc = (Mlmc*1e-11)**0.6 * 8.5  # scale radius in kpc
    Rcut = Rlmc*10                  # outer cutoff radius
    
    pot  = agama.Potential(type='spheroid', gamma=1, beta=3, alpha=1,  # power-law indices of the NFW profile
        scaleradius=Rlmc, mass=Mlmc, outercutoffradius=Rcut)
    df   = agama.DistributionFunction(type='QuasiSpherical', density=pot, potential=pot)  # Eddington DF
    
    ic = agama.GalaxyModel(pot, df).sample(Nbody)
    return ic


def generate_Sgr(Nstar = 200_000, Nhalo = 500_000):
    """
    Based on code provided by Vasiliev+2019 to model Sgr.
    https://arxiv.org/abs/2009.10726
    NB: parameters have been modified! Not exactly same as original Vasiliev+19 Sgr model.
    """

    # units:  length=1 kpc, velocity=1 km/s, mass=1 Msun (G=1).
    
    mstar =  2e8
    mhalo  = 4e9 
    
    pot_star = agama.Potential(type='king', mass=mstar, # stars
                               scaleradius=0.8, w0=4)    # w0 = dimensionless potential depth
    pot_halo    = agama.Potential(type='spheroid', mass=mhalo, # halo
                                scaleradius=6.4, gamma=0, 
                                beta=3, alpha=1, 
                                outercutoffradius=3.6, 
                                cutoffstrength=2)
    
    pot = agama.Potential(pot_star, pot_halo)
    
    df_star = agama.DistributionFunction(type='QuasiSpherical', density=pot_star, potential=pot)
    df_halo = agama.DistributionFunction(type='QuasiSpherical', density=pot_halo, potential=pot)

    starp, starm = agama.GalaxyModel(pot, df_star).sample(Nstar)
    halop, halom = agama.GalaxyModel(pot, df_halo).sample(Nhalo) 
    
    posvel = np.vstack((starp, halop)) 
    mass = np.hstack((starm, halom)) 
    ptype = np.hstack((np.ones(Nstar)*1,np.ones(Nhalo)*2)).astype(int)
    return  posvel, mass, ptype