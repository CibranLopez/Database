import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq



### Writing in a file ###



def write(path, line, text, value, length=0, name='Summary.dat', mode='write'):
    with open(f'{path}/{name}', 'r+') as summary_file:
        summary_lines = summary_file.readlines()
        summary_file.seek(0)
        
        for i in range(line):
            summary_file.write(summary_lines[i])
        
        if not length:
            summary_file.write(text + str(value))
        else:
            summary_file.write(text)
            for i in range(length):
                summary_file.write(' ' + str(value[i]))
        
        aux = 0
        if mode == 'insert':
            aux = -1
        
        summary_file.write('\n')
        for i in np.arange(line + 1 + aux, len(summary_lines)):
            summary_file.write(summary_lines[i])

        summary_file.truncate()


### Vibrational properties ###


k_B = 8.617333262145e-2 # meV / K

def normalized_trapezoidal_integral(x, fx, rho):
    return np.trapz(fx * rho, x) / np.trapz(rho, x)

def calculate_harmonic_phonon_energy(omega, T):
    auxiliar = omega / (k_B * T)
    return omega * (0.5  + 1 / (np.exp(auxiliar) - 1))

def calculate_constant_volume_heat_capacity(omega, T):
    auxiliar = omega / (k_B * T)
    return k_B * np.power(auxiliar, 2) * np.exp(auxiliar) / np.power(np.exp(auxiliar) - 1, 2)

def calculate_helmholtz_free_energy(omega, T):
    auxiliar = omega / (k_B * T)
    return 0.5 * omega + k_B * T * np.log(1 - np.exp(- auxiliar))

def calculate_entropy(omega, T):
    auxiliar = omega / (k_B * T)
    return omega / (2 * T * np.tanh(0.5 * auxiliar)) - k_B * np.log(2 * np.sinh(0.5 * auxiliar))

def getVPROP(path, omega_data, rho_omega_data, T, name, f_cut_off_line, f_line, hpe_line, cvhc_line, hfe_line, entropy_line):
    # Until 30 meV
    cut_off = np.where(omega_data <= 30)[0][-1]
    integral = normalized_trapezoidal_integral(omega_data[:cut_off], omega_data[:cut_off], rho_omega_data[:cut_off])
    write(path, f_cut_off_line, f'Mean frequency (cut-off 30meV, {name}): ', integral)

    # Until maximum energy available
    integral = normalized_trapezoidal_integral(omega_data, omega_data, rho_omega_data)
    write(path, f_line, f'Mean frequency {name}): ', integral)

    omega_data = omega_data[1:]
    rho_omega_data = rho_omega_data[1:]

    # Harmonic phonon energy
    harmonic_phonon_energy = calculate_harmonic_phonon_energy(omega_data, T)
    integral = normalized_trapezoidal_integral(omega_data, harmonic_phonon_energy, rho_omega_data)
    write(path, hpe_line, f'Harmonic phonon energy ({name}): ', integral)

    # Constant volume heat capacity
    constant_volume_heat_capacity = calculate_constant_volume_heat_capacity(omega_data, T)
    integral = normalized_trapezoidal_integral(omega_data, constant_volume_heat_capacity, rho_omega_data)
    write(path, cvhc_line, f'Constant volume heat capacity ({name}): ', integral)

    # Helmholtz free energy
    helmholtz_free_energy = calculate_helmholtz_free_energy(omega_data, T)
    integral = normalized_trapezoidal_integral(omega_data, helmholtz_free_energy, rho_omega_data)
    write(path, hfe_line, f'Helmholtz free energy ({name}): ', integral)

    # Entropy
    entropy = calculate_entropy(omega_data, T)
    integral = normalized_trapezoidal_integral(omega_data, entropy, rho_omega_data)
    write(path, entropy_line, f'Entropy ({name}): ', integral)



### Phonon density of states ###


def getPDOS(Niter, time_step, Nions, velocity, atoms='all_atoms', DiffStart=None, DiffEnd=None, unit='mev'):
    '''
    Phonon density of states from VAF.
    Velocity Autocorrelation Function. It is the sum over all atoms and dimensions of the correlation of velocities divided by the square sum of the velociy. Depending on the type of atom we select one part of the tensor or another.
    Velocity Autocorrelation Function tensor (correlation of the velocity of each particle in each dimension with itself), from which we may compute VAF.
    '''
    
    N = Niter - 1
    omega = fftfreq(2*N-1, time_step) * 1e3 # Frequency in THz

    if unit == 'cm-1': # Frequency in cm^-1 or mev (omega * hbar)
        omega *= 33.35640951981521
    elif unit == 'mev':
        omega *= 4.13567
    
    
    # Obtaining correlation
    
    CORR = np.zeros(((Niter-1)*2 - 1, Nions, 3))
    
    for i in range(Nions):
        for j in range(3):
            CORR[:, i, j] = np.correlate(velocity[:, i, j], velocity[:, i, j], 'full')
    
    
    # Defining correlation and velocity
    
    if   (atoms == 'all_atoms'): # All the tensor is used
        aux_corr = CORR
        aux_vel  = velocity

    elif (atoms == 'diffusive_atoms'): # The diffusive part is used
        aux_corr = CORR[:, DiffStart:DiffEnd]
        aux_vel  = velocity[:, DiffStart:DiffEnd]

    elif (atoms == 'non-diffusive_atoms'): # The non-diffusive part is used
        ac = CORR[:, :DiffStart]
        bc = CORR[:, DiffEnd:]

        av = velocity[:, :DiffStart]
        bv = velocity[:, DiffEnd:]

        aux_corr = np.concatenate((ac, bc), axis=1)
        aux_vel  = np.concatenate((av, bv), axis=1)

    
    # Computing the two-sided VAF
    
    VAF = np.sum(aux_corr, axis=(1, 2)) / np.sum(aux_vel**2)
    
    pdos = np.abs(fft(VAF - np.average(VAF)))
    
    return omega[:N], pdos[:N], VAF[Niter-2:]

def getDiff(TypeName, Nelem, DiffTypeName=None):
    '''
    Gets the diffusive and non-diffusive elements. The data is stored in the class. This
    assumes that there is only one diffusive element except for the combinations I-Br, and that
    in this latter case the follow each other.
    '''
    
    if not DiffTypeName:
        # Getting the diffusive element follwing the implicit order of preference
        
        for diff_component in ['Li', 'Na', 'Ag', 'Cu', 'Cl', 'I', 'Br', 'F', 'O']:
            if (diff_component in TypeName):
                DiffTypeName = diff_component
                break
    
    
    # Getting its position regarding the xdatcar file (allowing various, consecutive diffusive types)
    
    position = 0
    for element in TypeName:
        if (TypeName[position] == DiffTypeName):
            break
        position += 1

    
    # Saving the number of the starting diffusive particle and their number
    
    DiffStart = np.sum(Nelem[:position])
    DiffNelem = Nelem[position]

    
    # Checking for the I-Br case
    
    if (('I' in TypeName) & ('Br' in TypeName)):
        Br_position = 0
        for element in TypeName:
            if (TypeName[Br_position] == 'Br'):
                break
            Br_position += 1

        DiffTypeName += '_Br' # Concatenating the new diffusive element
        DiffNelem += Nelem[Br_position] # Adding the new diffusive particles
        DiffStart = min(DiffStart, np.sum(Nelem[:Br_position])) # Starts in the first one

    
    # Supressing the diffusive elements from the hole list in order to get the non-diffusive elements
    
    aux = TypeName.copy()

    for diff_element in DiffTypeName.split('_'):
        aux.remove(diff_element)

    NonDiffTypeName = '_'.join(aux)

    
    # Getting the end of the diffusive particles and the number of non-diffusive ones
    
    DiffEnd = DiffStart + DiffNelem
    NonDiffNelem = np.sum(Nelem) - DiffNelem

    
    # Printing the results for visual corroboration
    
    print(f'{TypeName}: {Nelem}')
    print(f'Diff: {DiffTypeName}: {DiffStart} - {DiffEnd} ({DiffNelem})  /  Non diff: {NonDiffTypeName}')
    
    return DiffTypeName, NonDiffTypeName, DiffStart, DiffEnd

def get_VACF_VDOS(path):
    # Intervals step and number of simulation steps between records
    
    with open(f'{path}/Summary.dat', 'r') as summary_file:
        summary_lines = summary_file.readlines()

    delta_t   = float(summary_lines[4].split(': ')[1][:-1])
    N_steps   = float(summary_lines[5].split(': ')[1][:-1])
    time_step = N_steps * delta_t
    
    
    # Importing the XDATCAR file
    
    XDATCAR_lines = [line for line in open(f'{path}/XDATCAR') if line.strip()]
    
    scale = float(XDATCAR_lines[1])
    
    cell = np.array([line.split() for line in XDATCAR_lines[2:5]], dtype=float)
    cell *= scale
    
    TypeName = XDATCAR_lines[5].split()
    Nelem    = np.array(XDATCAR_lines[6].split(), dtype=int)
    
    Ntype = len(TypeName)
    Nions = np.sum(Nelem)

    
    # Shaping the configurations data into the positions attribute
    
    pos = np.array([line.split() for line in XDATCAR_lines[8:] if not line.split()[0][0].isalpha()], dtype=float)

    position = pos.ravel().reshape((-1, Nions, 3))
    positionC = np.zeros_like(position)
    Niter = position.shape[0]

    
    # Getting the variation in positions and applying periodic boundary condition
    
    dpos = np.diff(position, axis=0)
    dpos[dpos > 0.5]  -= 1.0
    dpos[dpos < -0.5] += 1.0

    
    # Getting the positions and variations in cell units
    
    for i in range(Niter-1):
        positionC[i] = np.dot(position[i], cell)
        dpos[i]      = np.dot(dpos[i],     cell)

    positionC[-1] = np.dot(position[-1], cell)

    
    # Defining the attribute of window=1 variation in position and the velocity
    
    velocity = dpos / time_step
    
    
    # Calculating the diffusive and non-diffusive elements
    
    Time = np.arange(Niter-1) * time_step
    xVAF = 1e-3 * Time
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    atoms_types = ['all_atoms', 'diffusive_atoms', 'non-diffusive_atoms']
    
    DiffTypeName, NonDiffTypeName, DiffStart, DiffEnd = getDiff(TypeName, Nelem)
    
    for i in range(len(atoms_types)):
        atoms = atoms_types[i]
        
        xPDOS, yPDOS, yVAF = getPDOS(Niter, time_step, Nions, velocity, atoms=atoms, DiffStart=DiffStart, DiffEnd=DiffEnd)
        
        ax[i].plot(xPDOS, yPDOS, label=atoms)
        ax[i].set_xlabel(r'$\omega$ (meV)')
        ax[i].set_ylabel('VDOS (arb units)')
        ax[i].legend(loc='best')
        
        name = ''
        if   atoms == 'diffusive_atoms':
            name = f'_{DiffTypeName}'
        elif atoms == 'non-diffusive_atoms':
            name = f'_{NonDiffTypeName}'
        
        np.savetxt(f'{path}/VACF{name}.dat', np.column_stack((xVAF,  yVAF)))
        np.savetxt(f'{path}/VDOS{name}.dat', np.column_stack((xPDOS, yPDOS)))
    
    plt.suptitle(f'VDOS ({TypeName})')
    plt.show()

def get_vibrational_properties(path):
    # Temperature of simulation
    
    with open(f'{path}/Summary.dat', 'r') as summary_file:
        summary_lines = summary_file.readlines()

    temperature = float(summary_lines[9].split(': ')[1][:-1])
    
    
    # Importing the POSCAR file
    
    with open(f'{path}/POSCAR', 'r') as POSCAR_file:
        POSCAR_lines = POSCAR_file.readlines()
    
    TypeName = POSCAR_lines[5].split()
    Nelem    = np.array(POSCAR_lines[6].split(), dtype=int)
    
    
    # Calculating the diffusive and non-diffusive elements
    
    DiffTypeName, NonDiffTypeName, DiffStart, DiffEnd = getDiff(TypeName, Nelem)
    
    VDOS_data = np.loadtxt(f'{path}/VDOS.dat')
    getVPROP(path, VDOS_data[:, 0], VDOS_data[:, 1], temperature, 'all atoms', 14, 11, 17, 20, 23, 26)
    
    VDOS_data = np.loadtxt(f'{path}/VDOS_{DiffTypeName}.dat')
    getVPROP(path, VDOS_data[:, 0], VDOS_data[:, 1], temperature, 'diffusive atoms', 15, 12, 18, 21, 24, 27)

    VDOS_data = np.loadtxt(f'{path}/VDOS_{NonDiffTypeName}.dat')
    getVPROP(path, VDOS_data[:, 0], VDOS_data[:, 1], temperature, 'non-diffusive atoms', 16, 13, 19, 22, 25, 28)

def get_mean_square_displacement(path):
    system(f'cp Mean_square_displacement {path}')
    current_dir = getcwd()
    chdir(path)
    system(f'./Mean_square_displacement')
    chdir(current_dir)

def linear_funtion(x, y_0, D):
    return y_0 + 6 * D * x