import numpy as np
import h5py as h5

def dummy_data(Ntypes=1, no_type0=True):
    """Returns a data container with initial conditions 
    with randomly generated pos, vel sampled from a Gaussian.
    
    :Input:
    Ntypes: int
        The number of independent particle types. By default
        particle Type 0 are gas particles in Gadget.
    no_type0: bool
        default = True. If true: skips Type 0 particles 
        (which are gas particles by default)
    """
    
    data = dict()
    
    for i in range(no_type0, Ntypes+1*no_type0):
        rng = np.random.RandomState(i)
        pos = rng.normal(size=(5,3))
        vel = rng.normal(size=(5,3))
        ID = np.arange(5) + (i - no_type0)*5
        data[f'Type{i}'] = dict(pos = pos, 
                                vel=vel, 
                                ID = ID)
        
    return data

def empty_data_dict(Ntypes=1, no_type0=True):
    """Creates an empty data container to store initial conditions.
    
    :Input:
    Ntypes: int
        The number of data types for the container to store.
    no_type0: bool (default=True)
        By default we will assume that we ignore Type0 particles, which
        is the default particle type for gass particles in Gadget.
    """
    
    data = dict()
    for i in range(no_type0, Ntypes+1*no_type0):
        data[f'Type{i}'] = dict(pos = '(N,3):float', 
                                vel='(N,3):float', 
                                ID = '(N):int')
    return data

def write_gadget_hdf(filename, data,
                     NumPart, 
                     Massarr, 
                     Time:float=0, 
                     NumFiles:int=1, 
                     Redshift:float=None, 
                     BoxSize:float=None,
                     double_precision=True):
    """Writes initial conditions to a Gadget HDF5 file.
    
    :Input:
    
    filename: str
        Name of the output file, hdf5 will be appended if given without.
    
    data: dict
        Dictionary with the following format:
        dict( Type1 = dict(pos=pos, vel=vel, ID=ID, mass=mass),
              Type2 = ...)

    NumPart: int, array(dtype=int)
        (Gadget keyword name: NumPart_ThisFile)
        The number of particles of each type in the present file.
        NB: currently it is assumed that Npart == Nall which is 
        only true if NumFiles==1.
    
    Nall: int, array(dtype=int)
        (Gadget keyword name: NumPart_Total)
        Total number of particles of each type in the simulation.

    Massarr: float, array(dtype=float)
        (Gadget keyword name: MassTable)
        The mass of each particle type. If set to 0 for a 
        type which is present, individual particle masses are stored for this type.


    Time: float
        Time of output, or expansion factor for cosmological simulations.
        Time = 0.0 by default.
        
    Redshift: float
        z=1/a-1 (only set for cosmological integrations)
        Redshift = None by default.
        
    BoxSize: float
        Gives the box size if periodic boundary conditions are used.
        BoxSize = None by default.

    NumFiles: int
        (Gadget keyword name: NumFilesPerSnapshot)
        Number of files in each snapshot. 
        NumFiles = 1 by default.
    
    Routine is based on description of: 
    https://gitlab.mpcdf.mpg.de/vrs/gadget4/-/blob/master/documentation/06_snapshotformat.md
    And with hints of inspiration from:
    https://github.com/ruggiero/galstep/blob/master/galstep/snapwrite.py
    """
    
    #Append .hdf5 if necessary
    filename += '.hdf5' if not filename.endswith('.hdf5') else ''
   
    f = h5.File(filename, 'w')
    
    ### Creating the Header
    header = f.create_group('Header')
    
    header.attrs['NumPart_ThisFile'] = np.asarray(NumPart, dtype='uint')
    header.attrs['NumPart_Total']    = np.asarray(NumPart, dtype='uint64')
    header.attrs['MassTable']          = np.asarray(Massarr, dtype='double')
    header.attrs['Time'] = float(Time)

    if Redshift is not None:
        header.attrs['Redshift'] = Redshift
    else:
        # Assumes that redshift is not important!
        header.attrs['Redshift'] = float(0)
    if BoxSize is not None:
        header.attrs['BoxSize'] = BoxSize
    else:
        # Assumes that no periodic boundary conditions are used!
        header.attrs['BoxSize'] = float(1)
    header.attrs['NumFilesPerSnapshot'] = NumFiles
    
    ### Creating particle data
    
    dtype = 'float64' if double_precision else 'float32'
    
    
    for i, N in enumerate(NumPart):
        if N==0:
            continue
        
        grp = f.create_group(f'PartType{i}')
        gdata = data[f'Type{i}']
        
        assert np.shape(gdata['pos'])  == (N, 3)
        assert np.shape(gdata['vel'])  == (N, 3)
        assert np.shape(gdata['ID'])   == (N, )
        
    
    
        grp.create_dataset('Coordinates', 
                           data = gdata['pos'],
                           dtype = dtype)
        
        grp.create_dataset('Velocities', 
                           data = gdata['vel'],
                           dtype = dtype)
        
        grp.create_dataset('ParticleIDs', 
                           data = gdata['ID'],
                           dtype = 'uint32')
        
        if Massarr[i] == 0:
            assert np.shape(gdata['mass']) == (N, )
            grp.create_dataset('Masses', 
                           data = gdata['mass'],
                           dtype = dtype)
        
    f.close()
        # Some lines copied from https://github.com/ruggiero/galstep/blob/master/galstep/snapwrite.py
        # Add if ever metallicities are necessary!
        ## TODO currently all gas+stars get the same metallicity.
        ## this should be an option in the configuration as well.
        #if i in [0, 2, 3, 4]:
        #    if raw_hdata[29] > 1:
        #        assert np.shape(Z[start:end]) == (n_part[i], raw_hdata[29])
        #
        #    p.create_dataset('Metallicity', data = Z[start:end],
        #          dtype = dtype)
        #
        #if i == 0 and N_gas > 0:
        #    assert np.shape(U_data[start:end]) == (n_part[i], )
        #    assert np.shape(rho_data[start:end]) == (n_part[i], )
        #    assert np.shape(smoothing_data[start:end]) == (n_part[i], )
        #
        #    p.create_dataset('InternalEnergy', data = U_data[start:end],
        #          dtype = dtype)
        #    p.create_dataset('Density', data = rho_data[start:end],
        #          dtype = dtype)
        #    p.create_dataset('SmoothingLength', 
        #          data = smoothing_data[start:end],
        #          dtype = dtype)   
        
        
if __name__=='__main__':
    data = dummy_data(2)

    f = write_gadget_hdf('tmp.hdf5', data=data, NumPart=[0,5,5], Massarr=[0,5,5])
    f = h5.File('tmp.hdf5',mode='r')
    header = f['Header']

    for key in header.attrs.keys():
        print(f'{key:18}: ', header.attrs[key])

    print(f['PartType1']['Coordinates'][()])