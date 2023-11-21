#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
import DB_library as DBL
import numpy as np
import os

sns.set_theme()


# In[14]:


'''

    '../../ICMAB/Data/Na-based/NaBH4/AIMD/stoichiometric/750K',
    '../../ICMAB/Data/Na-based/NaBH4/AIMD/stoichiometric/300K',
    '../../ICMAB/Data/Na-based/NaBH4/AIMD/stoichiometric/600K',
    '../../ICMAB/Data/Na-based/NaBH4/AIMD/stoichiometric/450K',
    '../../ICMAB/Data/Na-based/NaBH4/AIMD/non-stoichiometric/750K',
    '../../ICMAB/Data/Na-based/NaBH4/AIMD/non-stoichiometric/300K',
    '../../ICMAB/Data/Na-based/NaBH4/AIMD/non-stoichiometric/600K',
    '../../ICMAB/Data/Na-based/NaBH4/AIMD/non-stoichiometric/450K',
    '../../ICMAB/Data/Na-based/Na3SbS4/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Na-based/Na3SbS4/AIMD/stoichiometric/900K',
    '../../ICMAB/Data/Na-based/Na3SbS4/AIMD/stoichiometric/700K',
    '../../ICMAB/Data/Na-based/NaZr2-PO4-3/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Na-based/NaZr2-PO4-3/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Na-based/NaBO2/AIMD/stoichiometric/750K',
    '../../ICMAB/Data/Na-based/Na2B4O7/AIMD/stoichiometric/750K',
    '../../ICMAB/Data/Na-based/Na2B4O7/AIMD/stoichiometric/300K',
    '../../ICMAB/Data/Na-based/Na2B4O7/AIMD/stoichiometric/600K',
    '../../ICMAB/Data/Na-based/Na2B4O7/AIMD/stoichiometric/450K',
    '../../ICMAB/Data/Na-based/NaMn-HCO2-3/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Na-based/NaMn-HCO2-3/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Na-based/NaMn-HCO2-3/AIMD/non-stoichiometric/900K',
    '../../ICMAB/Data/Na-based/NaMn-HCO2-3/AIMD/non-stoichiometric/700K',
    '../../ICMAB/Data/Na-based/Na3P11/AIMD/stoichiometric/1000K',
    '../../ICMAB/Data/Na-based/Na3P11/AIMD/non-stoichiometric/1000K',
    '../../ICMAB/Data/Na-based/Na3P11/AIMD/non-stoichiometric/-900K',
    '../../ICMAB/Data/Cu-based/Cu2Se/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Cu-based/Cu2Se/AIMD/non-stoichiometric/900K',
    '../../ICMAB/Data/Cu-based/Cu2Se/AIMD/non-stoichiometric/700K',
    '../../ICMAB/Data/Cu-based/Cu2Se/AIMD/non-stoichiometric/300K',
    '../../ICMAB/Data/Cu-based/CuI/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Cu-based/CuI/AIMD/stoichiometric/900K',
    '../../ICMAB/Data/Cu-based/CuI/AIMD/stoichiometric/700K',
    '../../ICMAB/Data/Halide-based/SrCl2/AIMD/stoichiometric/2000K',
    '../../ICMAB/Data/Halide-based/SrCl2/AIMD/stoichiometric/2500K',
    '../../ICMAB/Data/Halide-based/SrCl2/AIMD/stoichiometric/1500K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbBr3/AIMD/stoichiometric/400K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbBr3/AIMD/stoichiometric/250K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbBr3/AIMD/non-stoichiometric/400K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbBr3/AIMD/non-stoichiometric/800K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbBr3/AIMD/non-stoichiometric/600K',
    '../../ICMAB/Data/Halide-based/CaF2/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Halide-based/CaF2/AIMD/stoichiometric/1250K',
    '../../ICMAB/Data/Halide-based/CaF2/AIMD/stoichiometric/1000K',
    '../../ICMAB/Data/Halide-based/CaF2/AIMD/stoichiometric/1500K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI3/AIMD/stoichiometric/100K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI3/AIMD/stoichiometric/400K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI3/AIMD/stoichiometric/250K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI3/AIMD/non-stoichiometric/400K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI3/AIMD/non-stoichiometric/800K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI3/AIMD/non-stoichiometric/600K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI1Br2/AIMD/stoichiometric/400K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI1Br2/AIMD/stoichiometric/250K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI1Br2/AIMD/non-stoichiometric/400K',
    '../../ICMAB/Data/Halide-based/BaF2/AIMD/stoichiometric/1250K',
    '../../ICMAB/Data/Halide-based/BaF2/AIMD/stoichiometric/1000K',
    '../../ICMAB/Data/Halide-based/BaF2/AIMD/stoichiometric/1375K',
    '../../ICMAB/Data/Halide-based/BaF2/AIMD/stoichiometric/1500K',
    '../../ICMAB/Data/Halide-based/CsPbBr3/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Halide-based/CsPbBr3/AIMD/stoichiometric/850K',
    '../../ICMAB/Data/Halide-based/CsPbBr3/AIMD/stoichiometric/700K',
    '../../ICMAB/Data/Halide-based/CsPbBr3/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Halide-based/CsPbBr3/AIMD/non-stoichiometric/900K',
    '../../ICMAB/Data/Halide-based/CsPbBr3/AIMD/non-stoichiometric/700K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI2Br1/AIMD/stoichiometric/100K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI2Br1/AIMD/stoichiometric/400K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI2Br1/AIMD/stoichiometric/250K',
    '../../ICMAB/Data/Halide-based/CH3NH3-PbI2Br1/AIMD/non-stoichiometric/400K',
    '../../ICMAB/Data/Halide-based/SrF2/AIMD/stoichiometric/1250K',
    '../../ICMAB/Data/Halide-based/SrF2/AIMD/stoichiometric/1375K',
    '../../ICMAB/Data/Halide-based/SrF2/AIMD/stoichiometric/1750K',
    '../../ICMAB/Data/Halide-based/SrF2/AIMD/stoichiometric/1500K',
    '../../ICMAB/Data/Ag-based/AgI/AIMD/stoichiometric/350K',
    '../../ICMAB/Data/Ag-based/AgI/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Ag-based/AgI/AIMD/stoichiometric/650K',
    '../../ICMAB/Data/Ag-based/AgI/AIMD/stoichiometric/200K',
    '../../ICMAB/Data/Ag-based/Ag3SI/AIMD/stoichiometric/600K',
    '../../ICMAB/Data/Ag-based/AgCrSe2/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Ag-based/AgCrSe2/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Ag-based/Ag3SBr/AIMD/stoichiometric/600K',
    '../../ICMAB/Data/O-based/SrTiO3/AIMD/non-stoichiometric/2000K',
    '../../ICMAB/Data/O-based/SrTiO3/AIMD/non-stoichiometric/1000K',
    '../../ICMAB/Data/O-based/SrTiO3/AIMD/non-stoichiometric/1500K',
    '../../ICMAB/Data/O-based/SrCoO3/AIMD/stoichiometric/400K',
    '../../ICMAB/Data/O-based/SrCoO3/AIMD/stoichiometric/200K',
    '../../ICMAB/Data/O-based/SrCoO3/AIMD/non-stoichiometric/400K',
    '../../ICMAB/Data/O-based/SrCoO3/AIMD/non-stoichiometric/200K',
    '../../ICMAB/Data/O-based/LaGaO3/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/O-based/LaGaO3/AIMD/non-stoichiometric/900K',
    '../../ICMAB/Data/O-based/LaGaO3/AIMD/non-stoichiometric/700K',
    '../../ICMAB/Data/O-based/d-Bi2O3/AIMD/stoichiometric/500K',
'''
paths_to_XDATCAR = [
    '../../ICMAB/Data/Li-based/Li3La4Ti8O24/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Li-based/Li3La4Ti8O24/AIMD/stoichiometric/900K',
    '../../ICMAB/Data/Li-based/Li3La4Ti8O24/AIMD/stoichiometric/1100K',
    '../../ICMAB/Data/Li-based/Li3La4Ti8O24/AIMD/stoichiometric/700K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/stoichiometric/1300K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/stoichiometric/900K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/stoichiometric/1100K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/stoichiometric/700K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/stoichiometric/300K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/non-stoichiometric/1300K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/non-stoichiometric/900K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/non-stoichiometric/1100K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/non-stoichiometric/1500K',
    '../../ICMAB/Data/Li-based/LiGaO2/AIMD/non-stoichiometric/700K',
    '../../ICMAB/Data/Li-based/Li7P3S11/AIMD/stoichiometric/400K',
    '../../ICMAB/Data/Li-based/Li7P3S11/AIMD/stoichiometric/800K',
    '../../ICMAB/Data/Li-based/Li7P3S11/AIMD/stoichiometric/600K',
    '../../ICMAB/Data/Li-based/Li7P3S11/AIMD/non-stoichiometric/400K',
    '../../ICMAB/Data/Li-based/Li7P3S11/AIMD/non-stoichiometric/800K',
    '../../ICMAB/Data/Li-based/Li7P3S11/AIMD/non-stoichiometric/600K',
    '../../ICMAB/Data/Li-based/LiIO3/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Li-based/LiIO3/AIMD/stoichiometric/900K',
    '../../ICMAB/Data/Li-based/LiIO3/AIMD/stoichiometric/1100K',
    '../../ICMAB/Data/Li-based/LiIO3/AIMD/stoichiometric/700K',
    '../../ICMAB/Data/Li-based/LiIO3/AIMD/stoichiometric/300K',
    '../../ICMAB/Data/Li-based/LiIO3/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Li-based/LiIO3/AIMD/non-stoichiometric/900K',
    '../../ICMAB/Data/Li-based/LiIO3/AIMD/non-stoichiometric/700K',
    '../../ICMAB/Data/Li-based/LiIO3/AIMD/non-stoichiometric/300K',
    '../../ICMAB/Data/Li-based/LiF/AIMD/stoichiometric/1250K',
    '../../ICMAB/Data/Li-based/LiF/AIMD/stoichiometric/1000K',
    '../../ICMAB/Data/Li-based/LiF/AIMD/stoichiometric/800K',
    '../../ICMAB/Data/Li-based/LiF/AIMD/non-stoichiometric/400K',
    '../../ICMAB/Data/Li-based/LiF/AIMD/non-stoichiometric/800K',
    '../../ICMAB/Data/Li-based/LiF/AIMD/non-stoichiometric/600K',
    '../../ICMAB/Data/Li-based/Li2SnS3/AIMD/stoichiometric/400K',
    '../../ICMAB/Data/Li-based/Li2SnS3/AIMD/stoichiometric/800K',
    '../../ICMAB/Data/Li-based/Li2SnS3/AIMD/stoichiometric/600K',
    '../../ICMAB/Data/Li-based/Li2SnS3/AIMD/non-stoichiometric/400K',
    '../../ICMAB/Data/Li-based/Li2SnS3/AIMD/non-stoichiometric/1200K',
    '../../ICMAB/Data/Li-based/Li2SnS3/AIMD/non-stoichiometric/1000K',
    '../../ICMAB/Data/Li-based/Li2SnS3/AIMD/non-stoichiometric/800K',
    '../../ICMAB/Data/Li-based/Li2SnS3/AIMD/non-stoichiometric/600K',
    '../../ICMAB/Data/Li-based/LiTi2-PO4-3/AIMD/stoichiometric/400K',
    '../../ICMAB/Data/Li-based/LiTi2-PO4-3/AIMD/stoichiometric/800K',
    '../../ICMAB/Data/Li-based/LiTi2-PO4-3/AIMD/stoichiometric/600K',
    '../../ICMAB/Data/Li-based/LiTi2-PO4-3/AIMD/non-stoichiometric/400K',
    '../../ICMAB/Data/Li-based/LiTi2-PO4-3/AIMD/non-stoichiometric/800K',
    '../../ICMAB/Data/Li-based/LiTi2-PO4-3/AIMD/non-stoichiometric/600K',
    '../../ICMAB/Data/Li-based/Li3OCl/AIMD/stoichiometric/1250K',
    '../../ICMAB/Data/Li-based/Li3OCl/AIMD/non-stoichiometric/1650K',
    '../../ICMAB/Data/Li-based/Li3OCl/AIMD/non-stoichiometric/1250K',
    '../../ICMAB/Data/Li-based/Li3OCl/AIMD/non-stoichiometric/1000K',
    '../../ICMAB/Data/Li-based/Li3OCl/AIMD/non-stoichiometric/1500K',
    '../../ICMAB/Data/Li-based/LiMn-HCO2-3/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Li-based/LiMn-HCO2-3/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Li-based/LiMn-HCO2-3/AIMD/non-stoichiometric/900K',
    '../../ICMAB/Data/Li-based/LiMn-HCO2-3/AIMD/non-stoichiometric/700K',
    '../../ICMAB/Data/Li-based/Li10GeS2P12/AIMD/stoichiometric/650K',
    '../../ICMAB/Data/Li-based/Li10GeS2P12/AIMD/stoichiometric/1150K',
    '../../ICMAB/Data/Li-based/Li10GeS2P12/AIMD/stoichiometric/1400K',
    '../../ICMAB/Data/Li-based/Li10GeS2P12/AIMD/stoichiometric/900K',
    '../../ICMAB/Data/Li-based/Li6PS5Br/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Li-based/Li6PS5Br/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Li-based/Li3N/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Li-based/Li3N/AIMD/stoichiometric/900K',
    '../../ICMAB/Data/Li-based/Li3N/AIMD/stoichiometric/1100K',
    '../../ICMAB/Data/Li-based/Li3N/AIMD/stoichiometric/700K',
    '../../ICMAB/Data/Li-based/Li3N/AIMD/stoichiometric/300K',
    '../../ICMAB/Data/Li-based/Li3N/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Li-based/Li3N/AIMD/non-stoichiometric/900K',
    '../../ICMAB/Data/Li-based/Li3N/AIMD/non-stoichiometric/700K',
    '../../ICMAB/Data/Li-based/Li3N/AIMD/non-stoichiometric/300K',
    '../../ICMAB/Data/Li-based/Li7La3Zr2O12/AIMD/stoichiometric/400K',
    '../../ICMAB/Data/Li-based/Li7La3Zr2O12/AIMD/stoichiometric/800K',
    '../../ICMAB/Data/Li-based/Li7La3Zr2O12/AIMD/stoichiometric/600K',
    '../../ICMAB/Data/Li-based/Li7La3Zr2O12/AIMD/non-stoichiometric/400K',
    '../../ICMAB/Data/Li-based/Li7La3Zr2O12/AIMD/non-stoichiometric/800K',
    '../../ICMAB/Data/Li-based/Li7La3Zr2O12/AIMD/non-stoichiometric/600K',
    '../../ICMAB/Data/Li-based/Li2B12H12/AIMD/stoichiometric/600K',
    '../../ICMAB/Data/Li-based/Li2B12H12/AIMD/non-stoichiometric/600K',
    '../../ICMAB/Data/Li-based/LiNbO3/AIMD/stoichiometric/500K',
    '../../ICMAB/Data/Li-based/LiNbO3/AIMD/stoichiometric/900K',
    '../../ICMAB/Data/Li-based/LiNbO3/AIMD/stoichiometric/700K',
    '../../ICMAB/Data/Li-based/LiNbO3/AIMD/non-stoichiometric/500K',
    '../../ICMAB/Data/Li-based/LiNbO3/AIMD/non-stoichiometric/900K',
    '../../ICMAB/Data/Li-based/LiNbO3/AIMD/non-stoichiometric/700K'
]


# ### Extract data from INCAR file at path_to_XDATCAR

# In[15]:


def read_INCAR(path_to_simulation):
    """Reads VASP INCAR files. It is always expected to find these parameters.
    Read VASP INCAR Settings

    Args:
        path_to_simulation (str): Path to the simulation directory.

    Returns:
        tuple: A tuple containing delta_t and n_steps.
    """
    
    # Predefining the variable, so later we check if they were found
    
    delta_t = None
    n_steps = None
    
    # Loading the INCAR file
    
    if not os.path.exists(f'{path_to_simulation}/INCAR'):
        exit('INCAR file is not available.')
    
    with open(f'{path_to_simulation}/INCAR', 'r') as INCAR_file:
        INCAR_lines = INCAR_file.readlines()
    
    # Looking for delta_t and n_steps
    
    for line in INCAR_lines:
        split_line = line.split('=')
        if len(split_line) > 1:  # Skipping empty lines
            label = split_line[0].split()[0]
            value = split_line[1].split()[0]
            
            if   label == 'POTIM':  delta_t = float(value)
            elif label == 'NBLOCK': n_steps = float(value)
    
    # Checking if they were found
    
    if (delta_t is None) or (n_steps is None):
        exit('POTIM or NBLOCK are not correctly defined in the INCAR file.')
    return delta_t, n_steps



all_calculations = []
for path_to_XDATCAR in paths_to_XDATCAR:
    print(path_to_XDATCAR)
    
    INCAR_delta_t, INCAR_n_steps = read_INCAR(path_to_XDATCAR)
    INCAR_delta_t, INCAR_n_steps


    # ### Extract data from XDATCAR file at path_to_XDATCAR

    # In[16]:


    # Import the XDATCAR file
    XDATCAR_lines = [line for line in open(f'{path_to_XDATCAR}/XDATCAR') if line.strip()]

    # Extract initial XDATCAR data
    compound = XDATCAR_lines[0][:-1]
    scale    = float(XDATCAR_lines[1])

    lattice_vectors = np.array([line.split() for line in XDATCAR_lines[2:5]], dtype=float)
    lattice_vectors *= scale  # Use scaling

    composition   = XDATCAR_lines[5].split()
    concentration = np.array(XDATCAR_lines[6].split(), dtype=int)

    n_atoms = np.sum(concentration)  # Number of particles within the simulation box

    print_name = ' '.join(composition)
    print(f'Compound: {compound}')
    print(f'Composition: {print_name}')
    print(f'Concentration: {concentration}')

    # Shape the configurations data into the positions attribute
    direct_coordinates = np.array([line.split() for line in XDATCAR_lines[8:] if not line.split()[0][0].isalpha()], dtype=float)

    direct_coordinates = direct_coordinates.ravel().reshape((-1, n_atoms, 3))  # (n_conf, n_atoms, 3) tensor
    n_conf = direct_coordinates.shape[0]  # Number of configurations or simulation steps

    print(f'Number of configurations: {n_conf}')


    # ### Identify diffusive particles

    # In[17]:


    diffusive_particle = 'Li'

    for i in range(n_atoms):
        if composition[i] == diffusive_particle:
            diffusive_idx = i
            break

    concentration_cumsum = np.insert(np.cumsum(concentration), 0, 0)

    # Exact indexes of the diffive particles
    diffusive_indexes = np.arange(concentration_cumsum[diffusive_idx],
                                  concentration_cumsum[diffusive_idx+1],
                                 dtype=int)

    # Index where the diffive particles start
    diffusive_indexes_start = diffusive_indexes[0]

    # Number of diffive particles
    n_diffusive_indexes = len(diffusive_indexes)

    print(diffusive_indexes)


    # ### Revert periodic boundary conditions and pass to cartesian coordinates

    # In[18]:


    # Get the variation in positions
    dpos = np.diff(direct_coordinates, axis=0)

    # Revert periodic boundary condition
    dpos[dpos > 0.5]  -= 1.0
    dpos[dpos < -0.5] += 1.0

    # Pass to cartesian
    for i in range(n_conf-1):
        dpos[i] = np.dot(dpos[i], lattice_vectors)

    # Copy direct to cartesian and pass first frame (configuration)
    cartesian_coordinates    = direct_coordinates.copy()
    cartesian_coordinates[0] = np.dot(cartesian_coordinates[0], lattice_vectors)

    # Expand dimensions and sum every dpos
    expanded_dimensions   = np.expand_dims(cartesian_coordinates[0], 0)
    cartesian_coordinates = np.concatenate([expanded_dimensions, dpos], axis=0)
    cartesian_coordinates = np.cumsum(cartesian_coordinates, axis=0)


    # ### Generate tensor

    # $$
    # Diff (\Delta t, atom, dim) = \frac{1}{t_{sim} - \Delta t} \sum_{t_0 = 0}^{t_{sim} - \Delta t} cc (t_0 + \Delta t, atom) - cc (t_0, atom)
    # $$
    #
    # $$
    # D_{self} (\Delta t, dim) = \frac{1}{n_{atoms}} \sum_{dim} \sum_{i = 1}^{n_{atoms}} Diff (\Delta t, atom_i, dim) * Diff (\Delta t, atom_i, dim)
    # $$
    #
    # $$
    # D_{distinc} (\Delta t, dim) = \frac{1}{n_{atoms} (n_{atoms}-1)} \sum_{dim} \sum_{i = 1}^{n_{atoms}} \sum_{j = i+1}^{n_{atoms}} Diff (\Delta t, atom_i, dim) * Diff (\Delta t, atom_j, dim)
    # $$

    # In[19]:


    differences_tensor_mean = np.zeros((n_conf, n_atoms, 3))
    differences_tensor_std  = np.zeros((n_conf, n_atoms, 3))

    # We vectorize in terms of n_atoms (only possibility here)
    for delta_t in np.arange(1, n_conf):  # delta_t = 0 gives 0 by definition
        # Number of windows which are used for screening distances
        n_windows = n_conf - delta_t
        
        # Generate mean over windows
        temp_mean = np.zeros((n_atoms, 3))
        for t_0 in range(n_windows):
            # Distance between two configurations of a same particle
            # td (atom_i, dim_i) = cc (t_0 + delta_t, atom_i, dim_k) - cc (t_0, atom_i, dim_k)
            temporal_dist = cartesian_coordinates[t_0 + delta_t] - cartesian_coordinates[t_0]

            # Add to temporal variable
            temp_mean += temporal_dist
        temp_mean /= n_windows
        
        # Generate std over windows
        temp_std  = np.zeros((n_atoms, 3))
        for t_0 in range(n_windows):
            # Distance between two configurations of a same particle
            # td (atom_i, dim_i) = cc (t_0 + delta_t, atom_i, dim_k) - cc (t_0, atom_i, dim_k)
            temporal_dist = cartesian_coordinates[t_0 + delta_t] - cartesian_coordinates[t_0]

            # Add to temporal variable
            temp_std += np.power(temporal_dist - temp_mean, 2)
        
        # If n_windows == 1 then the std is zero
        if n_windows > 1:
            temp_std = np.sqrt(temp_std / (n_windows * (n_windows-1)))
        
        # Update values
        differences_tensor_mean[delta_t] = temp_mean
        differences_tensor_std[delta_t]  = temp_std


    # ### Generate MSD

    # In[27]:


    # Define the array of time simulation in pico-seconds
    delta_t_array = np.arange(n_conf) * (INCAR_n_steps * INCAR_delta_t * 1e-3)

    n_self     = 0
    n_distinct = 0

    MSD_self     = np.zeros(n_conf)
    MSD_distinct = np.zeros(n_conf)
    for i in np.arange(n_diffusive_indexes):
        idx_i = diffusive_indexes_start + i
        
        # Self product
        for dim_k in range(3):
            dtm_i = differences_tensor_mean[:, idx_i, dim_k]
            
            # MSD_self (delta_n_conf) = dtm (delta_n_conf, atom_i, dim_k) * dtm (delta_n_conf, atom_i, dim_k)
            MSD_self += (dtm_i * dtm_i)
        n_self += 1
        
        for j in np.arange(i+1, n_diffusive_indexes):
            idx_j = diffusive_indexes_start + j
            
            # Distinct (cross) product
            for dim_k in range(3):
                dtm_i = differences_tensor_mean[:, idx_i, dim_k]
                dtm_j = differences_tensor_mean[:, idx_j, dim_k]
                
                # MSD_distinct (delta_n_conf) = dtm (delta_n_conf, atom_i, dim_k) * dtm (delta_n_conf, atom_j, dim_k)
                MSD_distinct += (dtm_i * dtm_j)
            n_distinct += 1

    MSD_self     /= n_self
    MSD_distinct /= n_distinct

    MSD_self_var     = np.zeros(n_conf)
    MSD_distinct_var = np.zeros(n_conf)
    for i in np.arange(n_diffusive_indexes):
        idx_i = diffusive_indexes_start + i
        
        # Self product
        var_prod = 0
        temp_MSD_self = np.zeros(n_conf)
        for dim_k in range(3):
            dtm_i = differences_tensor_mean[:, idx_i, dim_k]

            dts_i = differences_tensor_std[:, idx_i, dim_k]
            
            # MSD_self (delta_n_conf) = dtm (delta_n_conf, atom_i, dim_k) * dtm (delta_n_conf, atom_i, dim_k)
            temp_MSD_self += (dtm_i * dtm_i)
            
            # Uncertainty of the product (variance here instead of std)
            var_ii = np.power(dtm_i, 2) * np.power(dts_i, 2)
            var_prod += var_ii + var_ii
        MSD_self_var += np.power(temp_MSD_self - MSD_self, 2) * var_prod
        
        for j in np.arange(i+1, n_diffusive_indexes):
            idx_j = diffusive_indexes_start + j
            
            # Distinct (cross) product
            var_prod = 0
            temp_MSD_distinct = np.zeros(n_conf)
            for dim_k in range(3):
                dtm_i = differences_tensor_mean[:, idx_i, dim_k]
                dtm_j = differences_tensor_mean[:, idx_j, dim_k]
                
                dts_i = differences_tensor_std[:, idx_i, dim_k]
                dts_j = differences_tensor_std[:, idx_j, dim_k]
                
                # MSD_distinct (delta_n_conf) = dtm (delta_n_conf, atom_i, dim_k) * dtm (delta_n_conf, atom_j, dim_k)
                temp_MSD_distinct += (dtm_i * dtm_j)
                
                # Uncertainty of the product (variance here instead of std)
                var_ij = np.power(dtm_i, 2) * np.power(dts_j, 2)
                var_ji = np.power(dtm_j, 2) * np.power(dts_i, 2)
                var_prod += var_ij + var_ji
            MSD_distinct_var += np.power(temp_MSD_distinct - MSD_distinct, 2) * var_prod

    MSD_self_std     = np.sqrt(MSD_self_var     / (n_self     * (n_self    -1)))
    MSD_distinct_std = np.sqrt(MSD_distinct_var / (n_distinct * (n_distinct-1)))

    print(MSD_self, MSD_distinct)


    # In[41]:


    MSD_full     = MSD_self + MSD_distinct
    MSD_full_std = np.sqrt(np.power(MSD_self_std, 2) + np.power(MSD_distinct_std, 2))


    # ### Compute diffusion coefficient

    # In[44]:


    initial_point = None

    x_fit = delta_t_array

    if initial_point is None:
        initial_point = int(0.1 * n_conf)
    else:
        initial_point = int(initial_point * n_conf)

    y_fit    = MSD_self
    yerr_fit = None  #yerr_fit = MSD_self_std
    _beta_ = DBL.weighted_regression(x_fit[initial_point:],
                                     y_fit[initial_point:],
                                     DBL.linear_function,
                                     yerr=yerr_fit).beta
    coef_D_self = _beta_[1]

    y_fit    = MSD_full
    yerr_fit = None  #yerr_fit = MSD_full_std
    _beta_ = DBL.weighted_regression(x_fit[initial_point:],
                                     y_fit[initial_point:],
                                     DBL.linear_function,
                                     yerr=yerr_fit).beta
    coef_D_full = _beta_[1]

    print(coef_D_self, coef_D_full)
    all_calculations.append([path_to_XDATCAR, coef_D_self, coef_D_full])

    # ### Plot MSD

    # In[40]:


    plt.errorbar(delta_t_array,
                 MSD_self,
                 yerr=None,
                 label='Self')
    plt.errorbar(delta_t_array,
                 MSD_full,
                 yerr=None,
                 label='Self')

    plt.xlabel(r'$\Delta t$ (ps)')
    plt.ylabel(r'$MSD$ ($\mathregular{Å^2}$)')
    plt.legend(loc='best')
    plt.show()

np.savetxt('all_calculations.txt', all_calculations)
