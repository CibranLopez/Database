import numpy as np
import matplotlib.pyplot as plt
from os import system, getcwd, chdir
from scipy.fftpack import fft, fftfreq
from scipy.optimize import curve_fit


k_B = 8.617333262145e-2  # meV / K


# Writing in a file


def write(path, new_index, text, value, length=0, name='Summary.dat', mode='write'):
    """
    Writes in a specific position the given text (only value or array).

    'write' mode replaces the specific line with the given one.
    'insert' mode adds a new line in the specific position.
    """

    with open(f'{path}/{name}', 'r+') as file:
        lines = file.readlines()
        file.seek(0)

        for i in range(new_index):
            if i < len(lines): file.write(lines[i])
            else:              file.write('\n')

        if length:
            file.write(text)
            for i in range(length):
                file.write(f' {value[i]}')
        else:
            file.write(f'{text} {value}')

        aux = 0
        if mode == 'insert':
            aux = -1

        file.write('\n')
        for i in np.arange(new_index + 1 + aux, len(lines)):
            file.write(lines[i])

        file.truncate()


# Vibrational properties


def normalized_trapezoidal_integral(x, fx, rho):
    return np.trapz(fx * rho, x) / np.trapz(rho, x)


def calculate_harmonic_phonon_energy(omega, temperature):
    auxiliary = omega / (k_B * temperature)
    return omega * (0.5 + 1 / (np.exp(auxiliary) - 1))


def calculate_constant_volume_heat_capacity(omega, temperature):
    auxiliary = omega / (k_B * temperature)
    return k_B * np.power(auxiliary, 2) * np.exp(auxiliary) / np.power(np.exp(auxiliary) - 1, 2)


def calculate_helmholtz_free_energy(omega, temperature):
    auxiliary = omega / (k_B * temperature)
    return 0.5 * omega + k_B * temperature * np.log(1 - np.exp(- auxiliary))


def calculate_entropy(omega, temperature):
    auxiliary = omega / (k_B * temperature)
    return omega / (2 * temperature * np.tanh(0.5 * auxiliary)) - k_B * np.log(2 * np.sinh(0.5 * auxiliary))


def getVPROP(path, omega_data, rho_omega_data, temperature, name, indexes):
    [f_cut_off_line, f_line, hpe_line, cvhc_line, hfe_line, entropy_line] = indexes

    # Until 30 meV

    cut_off = np.where(omega_data <= 30)[0][-1]
    integral = normalized_trapezoidal_integral(omega_data[:cut_off], omega_data[:cut_off], rho_omega_data[:cut_off])
    write(path, f_cut_off_line, f'Mean frequency (cut-off 30meV, {name}):', integral)

    # Until maximum energy available

    integral = normalized_trapezoidal_integral(omega_data, omega_data, rho_omega_data)
    write(path, f_line, f'Mean frequency {name}):', integral)

    omega_data = omega_data[1:]
    rho_omega_data = rho_omega_data[1:]

    # Harmonic phonon energy

    harmonic_phonon_energy = calculate_harmonic_phonon_energy(omega_data, temperature)
    integral = normalized_trapezoidal_integral(omega_data, harmonic_phonon_energy, rho_omega_data)
    write(path, hpe_line, f'Harmonic phonon energy ({name}):', integral)

    # Constant volume heat capacity

    constant_volume_heat_capacity = calculate_constant_volume_heat_capacity(omega_data, temperature)
    integral = normalized_trapezoidal_integral(omega_data, constant_volume_heat_capacity, rho_omega_data)
    write(path, cvhc_line, f'Constant volume heat capacity ({name}):', integral)

    # Helmholtz free energy

    helmholtz_free_energy = calculate_helmholtz_free_energy(omega_data, temperature)
    integral = normalized_trapezoidal_integral(omega_data, helmholtz_free_energy, rho_omega_data)
    write(path, hfe_line, f'Helmholtz free energy ({name}):', integral)

    # Entropy

    entropy = calculate_entropy(omega_data, temperature)
    integral = normalized_trapezoidal_integral(omega_data, entropy, rho_omega_data)
    write(path, entropy_line, f'Entropy ({name}):', integral)


# Diffusion coefficient


def getDiff(composition, concentration, DiffTypeName=None):
    """
    Gets the diffusive and non-diffusive elements. The data is stored in the class.
    Various diffusive elements are allowedThis
    assumes that there is only one diffusive element except for the combinations I-Br, and that
    in this latter case the follow each other.
    """

    if not DiffTypeName:
        # Getting the diffusive element following the implicit order of preference

        for diff_component in ['Li', 'Na', 'Ag', 'Cu', 'Cl', 'I', 'Br', 'F', 'O']:
            if diff_component in composition:
                DiffTypeName = [diff_component]
                break

    NonDiffTypeName = composition.copy()
    for diff_element in DiffTypeName:
        NonDiffTypeName.remove(diff_element)

    # Getting the positions of the diffusive elements regarding the XDATCAR file

    diffusion_information = []
    for DiffTypeName_value in DiffTypeName:
        for TypeName_index in range(len(composition)):
            if composition[TypeName_index] == DiffTypeName_value:
                diffusion_information.append([TypeName_index, np.sum(concentration[:TypeName_index]),
                                              concentration[TypeName_index]])
    return DiffTypeName, NonDiffTypeName, diffusion_information


def linear_function(x, y_0, coef_D):
    return y_0 + 6 * coef_D * x


# Main functionalities


def get_VACF_VDOS(path, DiffTypeName=None, unit='meV'):
    """
    Velocity Density Of States from VAF.
    Velocity Auto-Correlation Function. It is the sum over all atoms and dimensions of the correlation of velocities
    divided by the square sum of the velociy. Depending on the type of atom we select one part of the tensor or another.
    Velocity Auto-correlation Function tensor (correlation of the velocity of each particle in each dimension
    with itself), from which we may compute VAF.
    """

    # Intervals step and number of simulation steps between records

    with open(f'{path}/Summary.dat', 'r') as summary_file:
        summary_lines = summary_file.readlines()

    delta_t = float(summary_lines[4].split(': ')[1][:-1])
    n_steps = float(summary_lines[5].split(': ')[1][:-1])
    time_step = n_steps * delta_t

    # Importing the XDATCAR file

    XDATCAR_lines = [line for line in open(f'{path}/XDATCAR') if line.strip()]

    compound = XDATCAR_lines[0][:-1]
    scale    = float(XDATCAR_lines[1])

    cell = np.array([line.split() for line in XDATCAR_lines[2:5]], dtype=float)
    cell *= scale

    composition   = XDATCAR_lines[5].split()
    concentration = np.array(XDATCAR_lines[6].split(), dtype=int)

    total_atoms = np.sum(concentration)

    print_name = ' '.join(composition)
    print(f'Compound: {compound}\nComposition: {print_name}\nConcentration: {concentration}')

    # Shaping the configurations data into the positions attribute

    pos = np.array([line.split() for line in XDATCAR_lines[8:] if not line.split()[0][0].isalpha()], dtype=float)

    position = pos.ravel().reshape((-1, total_atoms, 3))
    n_iter = position.shape[0]

    # Getting the variation in positions and applying periodic boundary condition

    dpos = np.diff(position, axis=0)
    dpos[dpos > 0.5]  -= 1.0
    dpos[dpos < -0.5] += 1.0

    # Getting the positions and variations in cell units

    for i in range(n_iter - 1):
        dpos[i] = np.dot(dpos[i], cell)

    # Defining the attribute of window=1 variation in position and the velocity

    velocity = dpos / time_step

    # Calculating the diffusive and non-diffusive elements

    DiffTypeName, NonDiffTypeName, diffusion_information = getDiff(composition, concentration,
                                                                   DiffTypeName=DiffTypeName)

    simulated_time = np.arange(n_iter - 1) * time_step
    xVACF = 1e-3 * simulated_time

    # Frequency from Fast Fourier Transform

    n_dpos = n_iter - 1
    omega = fftfreq(2 * n_dpos - 1, time_step) * 1e3  # Frequency in THz

    # Frequency in cm^-1 or meV (omega * hbar)

    if   unit == 'cm-1': omega *= 33.35640951981521
    elif unit == 'meV':  omega *= 4.13567

    # Obtaining correlation

    correlation = np.zeros(((n_iter - 1) * 2 - 1, total_atoms, 3))

    for i in range(total_atoms):
        for j in range(3):
            correlation[:, i, j] = np.correlate(velocity[:, i, j], velocity[:, i, j], 'full')

    diffusive_indexes     = np.zeros(total_atoms, dtype=int)
    non_diffusive_indexes = np.ones(total_atoms,  dtype=int)

    for line in diffusion_information:
        start = int(line[1])
        end   = int(start + line[2])
        diffusive_indexes[start:end]     = 1
        non_diffusive_indexes[start:end] = 0
    diffusive_indexes     = np.where(diffusive_indexes)[0]
    non_diffusive_indexes = np.where(non_diffusive_indexes)[0]

    # Obtaining VDOS and VACF for all atoms, and the diffusive and non-diffusive ones

    atoms_types = ['all atoms', 'diffusive atoms', 'non-diffusive atoms']
    fig, ax = plt.subplots(1, len(atoms_types), figsize=(5*len(atoms_types), 5))
    for i in range(len(atoms_types)):
        atoms = atoms_types[i]

        # Defining correlation and velocity

        aux_cor = correlation
        aux_vel = velocity
        if   atoms == 'diffusive atoms':  # The diffusive part is used
            aux_cor = correlation[:, diffusive_indexes]
            aux_vel = velocity[:,    diffusive_indexes]
        elif atoms == 'non-diffusive atoms':  # The non-diffusive part is used
            aux_cor = correlation[:, non_diffusive_indexes]
            aux_vel = velocity[:,    non_diffusive_indexes]

        # Computing the two-sided VACF

        VACF = np.sum(aux_cor, axis=(1, 2)) / np.sum(aux_vel ** 2)
        pdos = np.abs(fft(VACF - np.average(VACF)))

        xVDOS = omega[:n_dpos]
        yVDOS = pdos[:n_dpos]
        yVACF  = VACF[n_iter - 2:]

        print_name = ' '.join(composition)
        name = ''
        if   atoms == 'diffusive atoms':
            print_name = ' '.join(DiffTypeName)
            print(f'Diffusive atoms: {print_name}')
            name = '_' + '_'.join(DiffTypeName)
        elif atoms == 'non-diffusive atoms':
            print_name = ' '.join(NonDiffTypeName)
            print(f'Non-diffusive atoms: {print_name}')
            name = '_' + '_'.join(NonDiffTypeName)

        ax[i].plot(xVDOS, yVDOS, label=print_name)
        ax[i].set_xlabel(r'$\omega$ (meV)')
        ax[i].set_ylabel('VDOS (arb units)')
        ax[i].legend(loc='best')

        np.savetxt(f'{path}/VACF{name}.dat', np.column_stack((xVACF, yVACF)))
        np.savetxt(f'{path}/VDOS{name}.dat', np.column_stack((xVDOS, yVDOS)))

    plt.suptitle(f'VDOS ({compound})')
    plt.show()


def get_vibrational_properties(path, DiffTypeName=None):
    # Temperature of simulation

    with open(f'{path}/Summary.dat', 'r') as summary_file:
        summary_lines = summary_file.readlines()

    temperature = float(summary_lines[9].split(': ')[1][:-1])

    # Importing the POSCAR file

    with open(f'{path}/POSCAR', 'r') as POSCAR_file:
        POSCAR_lines = POSCAR_file.readlines()

    composition   = POSCAR_lines[5].split()
    concentration = np.array(POSCAR_lines[6].split(), dtype=int)

    # Calculating the diffusive and non-diffusive elements

    DiffTypeName, NonDiffTypeName, diffusion_information = getDiff(composition, concentration,
                                                                   DiffTypeName=DiffTypeName)

    indexes = np.array([14, 11, 17, 20, 23, 26], dtype=int)
    atoms_types = ['all atoms', 'diffusive atoms', 'non-diffusive atoms']
    for atoms in atoms_types:
        name = ''
        if   atoms == 'diffusive atoms':
            name = '_' + '_'.join(DiffTypeName)
            indexes += 1
        elif atoms == 'non-diffusive atoms':
            name = '_' + '_'.join(NonDiffTypeName)
            indexes += 1

        VDOS_data = np.loadtxt(f'{path}/VDOS{name}.dat')
        getVPROP(path, VDOS_data[:, 0], VDOS_data[:, 1], temperature, atoms, indexes)


def get_mean_square_displacement(path):
    system(f'cp Mean_square_displacement {path}')
    current_dir = getcwd()
    chdir(path)
    system(f'./Mean_square_displacement')
    chdir(current_dir)


def get_diffusion_coefficient(path, DiffTypeName=None):
    """
    The initial point for the linear fit is selected as the one that minimizes uncertainty of the diffusion coefficient.
    """

    # Importing the POSCAR file

    with open(f'{path}/POSCAR', 'r') as POSCAR_file:
        POSCAR_lines = POSCAR_file.readlines()

    compound      = POSCAR_lines[0][:-1]
    composition   = POSCAR_lines[5].split()
    concentration = np.array(POSCAR_lines[6].split(), dtype=int)
    n_components  = len(composition)

    # Calculating the diffusive and non-diffusive elements

    DiffTypeName, _, diffusion_information = getDiff(composition, concentration,
                                                                   DiffTypeName=DiffTypeName)

    y_0_array    = []
    coef_D_array = []
    n_NonDiff        = 0
    mean_NonDiff_msd = 0

    rows = int(np.ceil(0.5 * n_components))

    # Generating the diffusion coefficients for each component

    fig, ax = plt.subplots(rows, 2, figsize=(5 * rows, 5))
    for i in range(n_components):
        element = composition[i]

        row = int(i / 2)
        column = int(i % 2)

        data = np.loadtxt(f'{path}/msd_{i}.dat')

        x = data[:, 0]
        y = data[:, 1]

        # Looking for the best initial point

        pcov_array = []
        for k in range(len(x) - 2):
            x_fit = x[k:]
            y_fit = y[k:]

            _, pcov = curve_fit(linear_function, x_fit, y_fit)
            pcov_array.append(pcov[1, 1])

        # Implementing the fit with lowest uncertainty in the diffusion coefficient

        initial_point = np.argmin(pcov_array)
        print(f'Initial point: {initial_point}')

        x_fit = x[initial_point:]
        y_fit = y[initial_point:]

        [y_0, coef_D], pcov = curve_fit(linear_function, x_fit, y_fit)

        y_0_array.append(y_0)
        coef_D_array.append(coef_D)

        image_index = column
        if n_components > 2:
            image_index = (row, column)

        ax[image_index].plot(x, y, '.', label='Data')
        ax[image_index].plot(x_fit, linear_function(x_fit, y_0, coef_D), label=u'Linear fitting')

        # Calculating the mean square displacement for the non-diffusive atoms
        if composition[i] in DiffTypeName:
            title = f'{element}: y_0 = {y_0:.2g}, D = {coef_D:.2g}'
        else:
            n_NonDiff        += concentration[i]
            mean_NonDiff_msd += concentration[i] * np.mean(y_fit)
            title = f'{element}: msd = {np.mean(y_fit):.2g}'

        ax[image_index].set_title(title)
        ax[image_index].legend(loc='best')

    plt.suptitle(compound)
    plt.show()

    mean_NonDiff_msd /= n_NonDiff
    print(f'Mean non-diffusive msd: {mean_NonDiff_msd}')

    write(path, 30, 'y_0:',                y_0_array,    n_components)
    write(path, 31, 'D:',                  coef_D_array, n_components)
    write(path, 32, 'Non-diffusive msd:', mean_NonDiff_msd)
