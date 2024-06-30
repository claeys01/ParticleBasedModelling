import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
import scipy.stats as stats
import CoolProp.CoolProp as CP

sys.path.append('C://Users//Matth//Master//Y1//Q4//particleBasedModeling//MolecularDynamicsAss')

def get_log(file: str) -> np.ndarray:
    """
    Function to plot the energies from the energies file
    :param directory: directory of the file
    :return:
    """
    step = []
    temp = []
    pressure = []
    kineticEnergy = []
    potentialEnergy = []
    totalEnergy = []
    data = (step, temp, pressure, kineticEnergy, potentialEnergy, totalEnergy)
    with open(file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line_split = line.strip().split(' ')
            for i in range(len(data)):
                data[i].append(float(line_split[i]))

        return data
def read_lammps_log(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    time_steps = []
    temperatures = []
    KE = []
    POT = []
    total = []
    pressures = []

    for line in data:
        if re.match(r"^\s*\d+\s+[-+]?[0-9]*\.?[0-9]+\s+", line):
            parts = line.split()
            time_steps.append(int(parts[0]))
            temperatures.append(float(parts[1]))
            pressures.append(float(parts[2]))
            KE.append(float(parts[3]))
            POT.append(float(parts[4]))
            total.append(float(parts[5]))

    return time_steps, temperatures, pressures, KE, POT, total


def plot_first_Qtune(directory):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    file_path = 'C://Users//Matth//Master//Y1//Q4//particleBasedModeling//MolecularDynamicsAss//VERIFY2//log.lammps'  # Replace with the actual path to your log file
    time_steps_verify2, temperatures_verify2, pressures_verify2, kinetic_verify2, potential_verify2, total_verify2  = read_lammps_log(file_path)
    ax.plot(time_steps_verify2, temperatures_verify2, label='Verify 2', color='blue', linestyle='dashed')

    # a list with 17 different colors
    colors = [
        'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive',
        'cyan', 'magenta', 'yellow', 'teal', 'lime', 'navy', 'maroon', 'black'
    ]
    # print(len(colors))
    i = 0
    for file in sorted(os.listdir(directory)):
        if file.endswith(".txt"):
            # print(int(file[8:10]))
            # print(int(file[8:10]))
            # label ='Q: 1e'+ str(int(file[3:5]))
            power = int(file[7:10])
            print(power)
            # if power == -20:
            # power = int(file[8:10])
            label = 'Q: 1e' + str(power)
            data = get_log(directory + file)
            ax.plot(data[0], data[1], label=label)
            i += 1
    ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Temperature')
    ax.grid()
    plt.savefig('MolecularDynamicsAss/Figures/tuning.png')
    plt.show()

def rdf(xyz, LxLyLz, n_bins=100, r_range=(0.01, 10.0)):
    '''
    rarial pair distribution function

    :param xyz: coordinates in xyz format per frame
    :param LxLyLz: box length in vector format
    :param n_bins: number of bins
    :param r_range: range on which to compute rdf
    :return:
    '''

    g_r, edges = np.histogram([0], bins=n_bins, range=r_range)
    g_r[0] = 0
    g_r = g_r.astype(np.float64)
    rho = 0

    for i, xyz_i in enumerate(xyz):
        xyz_j = np.vstack([xyz[:i], xyz[i + 1:]])
        d = np.abs(xyz_i - xyz_j)
        d = np.where(d > 0.5 * LxLyLz, LxLyLz - d, d)
        d = np.sqrt(np.sum(d ** 2, axis=-1))
        temp_g_r, _ = np.histogram(d, bins=n_bins, range=r_range)
        g_r += temp_g_r

    rho += (i + 1) / np.prod(LxLyLz)
    r = 0.5 * (edges[1:] + edges[:-1])
    V = 4. / 3. * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    norm = rho * i
    g_r /= norm * V

    return r, g_r

def get_grid_info(simul):
    n_side = int(np.ceil(simul.npart ** (1 / 3)))
    spacing = simul.side_length / n_side
    lattice_range = (simul.side_length - spacing) / 2
    return n_side/simul.sigma, spacing/simul.sigma, lattice_range/simul.sigma
def write_frame(coords, L, trajectory_name, step):
    '''
    function to write trajectory file in LAMMPS format

    In VMD you can visualize the motion of particles using this trajectory file.

    :param coords: coordinates
    :param vels: velocities
    :param forces: forces
    :param trajectory_name: trajectory filename

    :return:
    '''

    nPart = len(coords[:, 0])
    nDim = len(coords[0, :])
    with open(trajectory_name, 'a') as file:
        file.write('ITEM: TIMESTEP\n')
        file.write('%i\n' % step)
        file.write('ITEM: NUMBER OF ATOMS\n')
        file.write('%i\n' % nPart)
        file.write('ITEM: BOX BOUNDS pp pp pp\n')
        for dim in range(nDim):
            file.write('%.6f %.6f\n' % (-0.5 * L[dim], 0.5 * L[dim]))
        for dim in range(3 - nDim):
            file.write('%.6f %.6f\n' % (0, 0))
        file.write('ITEM: ATOMS id type xu yu zu\n')

        temp = np.zeros((nPart, 3))
        for dim in range(nDim):
            temp[:, dim] = coords[:, dim]

        for part in range(nPart):
            file.write('%i %i %.4f %.4f %.4f\n' % (part + 1, 1, *temp[part, :]))

def read_lammps_data(data_file, verbose=False):
    """Reads a LAMMPS data file
        Atoms
        Velocities
    Returns:
        lmp_data (dict):
            'xyz': xyz (numpy.ndarray)
            'vel': vxyz (numpy.ndarray)
        box (numpy.ndarray): box dimensions
    """
    print("Reading '" + data_file + "'")
    with open(data_file, 'r') as f:
        data_lines = f.readlines()

    directives = re.compile(r"""
        ((?P<n_atoms>\s*\d+\s+atoms)
        |
        (?P<box>.+xlo)
        |
        (?P<Atoms>\s*Atoms)
        |
        (?P<Velocities>\s*Velocities))
        """, re.VERBOSE)

    i = 0
    while i < len(data_lines):
        match = directives.match(data_lines[i])
        if match:
            if verbose:
                print(match.groups())

            elif match.group('n_atoms'):
                fields = data_lines.pop(i).split()
                n_atoms = int(fields[0])
                xyz = np.empty(shape=(n_atoms, 3))
                print(xyz)
                vxyz = np.empty(shape=(n_atoms, 3))

            elif match.group('box'):
                dims = np.zeros(shape=(3, 2))
                for j in range(3):
                    fields = [float(x) for x in data_lines.pop(i).split()[:2]]
                    dims[j, 0] = fields[0]
                    dims[j, 1] = fields[1]
                L = dims[:, 1] - dims[:, 0]

            elif match.group('Atoms'):
                if verbose:
                    print('Parsing Atoms...')
                data_lines.pop(i)
                data_lines.pop(i)

                while i < len(data_lines) and data_lines[i].strip():
                    fields = data_lines.pop(i).split()
                    a_id = int(fields[0])
                    xyz[a_id - 1] = np.array([float(fields[2]),
                                         float(fields[3]),
                                         float(fields[4])])

            elif match.group('Velocities'):
                if verbose:
                    print('Parsing Velocities...')
                data_lines.pop(i)
                data_lines.pop(i)

                while i < len(data_lines) and data_lines[i].strip():
                    fields = data_lines.pop(i).split()
                    va_id = int(fields[0])
                    vxyz[va_id - 1] = np.array([float(fields[1]),
                                         float(fields[2]),
                                         float(fields[3])])

            else:
                i += 1
        else:
            i += 1

    return xyz, vxyz, L
def read_xyz_trj(file_name):

    xyz_file = open(file_name, 'r')

    frame = 0
    xyz = {}
    READING=True
    while READING:
        nparts = int(xyz_file.readline())
        print(nparts)
        xyz_file.readline()
        try:
            nparts = int(xyz_file.readline())
            print(nparts)
            xyz_file.readline()
            xyz[frame] = np.zeros([nparts, 3])
            for k in range(0, nparts):
                line = xyz_file.readline()
                line = line.split()
                print(line)
                xyz[frame][k, 0] = line[1]
                xyz[frame][k, 1] = line[2]
                xyz[frame][k, 2] = line[3]
            frame += 1
        except:
            print("Reach end of '" + file_name + "'")
            READING=False

    return xyz

def plot_rdf(rdf, title, grid_data):
    n_side, spacing, lattice_range = grid_data
    print(n_side, spacing, lattice_range)
    spacing = np.array([spacing, spacing])
    size = 18

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    ax.plot(rdf[0] / MD.sigma, rdf[1], label='g(r)')
    # print(len(rdf[1]))
    linestyle_spacing = 'dashed'
    ax.plot(spacing, (rdf[1].min(), rdf[1].max()), label='1D spacing', linestyle=linestyle_spacing)
    ax.plot(2*spacing, (rdf[1].min(), rdf[1].max()), label='2*spacing', linestyle=linestyle_spacing)
    ax.plot(np.sqrt(2 * spacing ** 2), (rdf[1].min(), rdf[1].max()), label='2D Diagonal spacing', linestyle=linestyle_spacing)
    ax.plot(np.sqrt(3 * spacing ** 2), (rdf[1].min(), rdf[1].max()), label='3D Diagonal spacing', linestyle=linestyle_spacing)

    ax.annotate('3D spacing', (np.sqrt(3 * spacing[0] ** 2)-0.2, rdf[1].max()-5), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12)
    ax.annotate('Spacing', (spacing[0] - 0.2, rdf[1].max() - 6), textcoords="offset points", xytext=(0, 10),ha='center', fontsize=12)
    ax.annotate(r'2$\cdot$Spacing', (2 * spacing[0] - 0.15, rdf[1].max() - 7), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12)
    ax.annotate('2D spacing', (np.sqrt(2 * spacing[0] ** 2)-0.2, rdf[1].max()-8), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12)


    ax.set_xlabel('r/σ', size=size)
    ax.set_ylabel('g(r)', size=size)
    ax.set_title("Rdf for "+title, size=size+2)
    # ax.legend()
    ax.grid()
    plt.savefig(f'Figures/initial_conf_{title[6:9]}.png')
    plt.show()


def plot_log(file_path, verify_path):
    time_steps_verify, temperatures_verify, pressures_verify, kinetic_verify, potential_verify, total_verify  = read_lammps_log(verify_path)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    data = get_log(file_path)
    startfrom = 0
    ax.plot(data[0][startfrom:], data[1][startfrom:], label='Own simulation', color='red', linestyle='dashed')
    ax.plot(time_steps_verify, temperatures_verify, label='Verify 1', color='blue')
    ax.legend()
    # ax.set_xlabel('Step')
    ax.set_ylabel('Temperature')
    ax.grid()

    ax2.plot(data[0], data[3], color='blue', linestyle='dashed')
    ax2.plot(time_steps_verify, kinetic_verify, label='Ke', color='blue', linestyle='solid')
    ax2.plot(data[0], data[4],color='red', linestyle='dashed')
    ax2.plot(time_steps_verify, potential_verify, label='Pot',color='red', linestyle='solid')
    ax2.plot(data[0], data[5],color='black', linestyle='dashed')
    ax2.plot(time_steps_verify, total_verify, label='Total', color='black', linestyle='solid')
    ax2.legend()
    # ax2.set_xlabel('Step')
    ax2.set_ylabel('Energy(kcal/mol)')
    ax2.grid()

    ax3.plot(data[0], np.array(data[2])/101325, label='Own simulation', color='red', linestyle='dashed')
    ax3.plot(time_steps_verify, pressures_verify, label='Verify 1', color='blue')
    ax3.set_xlabel('time (fs)')
    ax3.set_ylabel('Pressure(atm)')
    ax3.grid()
    ax3.legend()


    # ax2.set_ylim(0, 10000)

    plt.savefig('MolecularDynamicsAss/Figures/verify4.png')
    plt.show()

def question4():
    verify_path1 = 'C://Users//Matth//Master//Y1//Q4//particleBasedModeling//MolecularDynamicsAss//VERIFY1//log.lammps'  # Replace with the actual path to your log file
    own_path = 'MolecularDynamicsAss/Outputs/final_standard_log2.txt'
    half_path = 'MolecularDynamicsAss/Outputs/final_standard_log2_05_20.txt'
    new_path = 'MolecularDynamicsAss/Outputs/tuningQ3/1.3335e-22_pbc_log.txt'

    plot_log(new_path, verify_path1)
    # plot_log(wrong_path, verify_path1)

def question5():
    verify_path2 = 'C://Users//Matth//Master//Y1//Q4//particleBasedModeling//MolecularDynamicsAss//VERIFY2//log.lammps'  # Replace with the actual path to your log file
    new_path = 'MolecularDynamicsAss/Outputs/correct_weird_log5.txt'

    plot_log(new_path, verify_path2)

def question22Liquid():
    directory = 'MolecularDynamicsAss/Outputs/Question22/Fluid/'

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    pressures = []
    coolprop_pressures = []
    stdvs = []
    temperatures = []
    confidence_intervals = []
    error_margins = []

    for file in sorted(os.listdir(directory)):
        if file.endswith(".txt"):
            data = get_log(directory + file)
            temperature = np.mean(data[1])
            pressure = np.array(data[2]) * 10 ** -6

            pressure_mean = np.mean(pressure)
            pressure_std = np.std(pressure)
            pressure_error = pressure_std / np.sqrt(len(pressure))

            confidence_level = 0.95
            critical_value = stats.norm.ppf((1 + confidence_level) / 2)

            pressure_error_margin = critical_value * pressure_error
            confidence_interval = (pressure_mean - pressure_error_margin, pressure_mean + pressure_error_margin)

            pressures.append(pressure_mean)
            stdvs.append(pressure_std)
            temperatures.append(temperature)
            confidence_intervals.append(confidence_interval)
            error_margins.append(pressure_error_margin)

    coolprop_pressures = CP.PropsSI('P', 'T', np.array([100, 200, 300, 400]), 'D', 358.4, 'Methane') * 10 ** -6


    ax.plot(temperatures, pressures, linestyle='dashed', color='blue', label='Simulation')
    # ax.errorbar(temperatures, pressures, yerr=stdvs, fmt='o', color='red')
    # for i, stdv in enumerate(stdvs):
    #     ax.annotate(f"σ: {stdv:.2f}", (temperatures[i] - 10, pressures[i] + stdv), xytext=(0, 5),
    #                 textcoords='offset points')
    #
    # ax.errorbar(temperatures, pressures, yerr=error_margins, fmt='o', color='orange')
    # ax.fill_between(temperatures, np.array(pressures) - np.array(error_margins),
    #                 np.array(pressures) + np.array(error_margins), color='orange', alpha=0.7,
    #                 label=' 95% Confidence Interval')
    # ax.fill_between(temperatures, np.array(pressures) - np.array(stdvs), np.array(pressures) + np.array(stdvs),
    #                 color='blue', alpha=0.1, label='Standard Deviation Bounds')

    ax.plot(np.array([100, 200, 300, 400]), coolprop_pressures, alpha=0.7, linewidth=3.0, linestyle='dashed', color='green',
            label='CoolProp')

    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel(r'Pressure [Mpa] ')
    ax.set_title('Temperature vs Pressure')
    ax.legend()
    plt.grid()
    plt.savefig('MolecularDynamicsAss/Figures/Question22_plot1.png')
    plt.show()

def question22Gas():
    directory = 'MolecularDynamicsAss/Outputs/Question22/Gas/'

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    pressures = []
    coolprop_pressures = []
    stdvs = []
    temperatures = []
    confidence_intervals = []
    error_margins = []

    for file in sorted(os.listdir(directory)):
        if file.endswith(".txt"):
            data = get_log(directory + file)
            temperature = np.mean(data[1])
            pressure = np.array(data[2]) * 10 ** -6

            pressure_mean = np.mean(pressure)
            pressure_std = np.std(pressure)
            pressure_error = pressure_std / np.sqrt(len(pressure))

            confidence_level = 0.95
            critical_value = stats.norm.ppf((1 + confidence_level) / 2)

            pressure_error_margin = critical_value * pressure_error
            confidence_interval = (pressure_mean - pressure_error_margin, pressure_mean + pressure_error_margin)

            pressures.append(pressure_mean)
            stdvs.append(pressure_std)
            temperatures.append(temperature)
            confidence_intervals.append(confidence_interval)
            error_margins.append(pressure_error_margin)

    coolprop_pressures = CP.PropsSI('P', 'T', np.array([200, 300, 400]), 'D', 1.6, 'Methane') * 10 ** -6


    ax.plot(temperatures, pressures, linestyle='dashed', color='blue', label='Simulation')
    ax.errorbar(temperatures, pressures, yerr=stdvs, fmt='o', color='red')
    for i, stdv in enumerate(stdvs):
        ax.annotate(f"σ: {stdv:.2f}", (temperatures[i] - 10, pressures[i] + stdv), xytext=(0, 5),
                    textcoords='offset points')

    ax.errorbar(temperatures, pressures, yerr=error_margins, fmt='o', color='orange')
    ax.fill_between(temperatures, np.array(pressures) - np.array(error_margins),
                    np.array(pressures) + np.array(error_margins), color='orange', alpha=0.7,
                    label=' 95% Confidence Interval')
    ax.fill_between(temperatures, np.array(pressures) - np.array(stdvs), np.array(pressures) + np.array(stdvs),
                    color='blue', alpha=0.1, label='Standard Deviation Bounds')

    # ax.plot(np.array([200, 300, 400]), coolprop_pressures, alpha=0.7, linewidth=3.0, linestyle='dashed', color='green',
    #         label='CoolProp')

    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel(r'Pressure [Mpa] ')
    ax.set_title('Temperature vs Pressure')
    ax.legend()
    plt.grid()
    plt.savefig('MolecularDynamicsAss/Figures/Question22_Gas_plot2.png')
    plt.show()


def extract_positions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    positions = []
    read_atoms = False

    for line in lines:
        if line.startswith('ITEM: ATOMS'):
            read_atoms = True
        elif line.startswith('ITEM:'):
            read_atoms = False
        elif read_atoms:
            data = line.split()
            xu, yu, zu = float(data[2]), float(data[3]), float(data[4])
            positions.append([xu, yu, zu])

    return np.array(positions)
def question23Liquid(directory):
    state = directory.split('/')[-2]
    print(state)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    for file in os.listdir(directory):
        if file.endswith(".lammps"):
            temp = file.split('_')[0]
            positions = extract_positions(directory + file)
            pos_rdf = rdf(positions, np.array([30, 30, 30]), n_bins=100, r_range=(0.01, 10.0))
            ax.plot(np.array(pos_rdf[0])/3.72, pos_rdf[1], label=temp + ' K')

    ax.set_title(f'Radial Distribution Function for {state} state', size=16)
    ax.set_xlabel('r/σ', size=13)
    ax.set_ylabel('g(r)', size=13)
    ax.legend()
    ax.grid()
    plt.savefig(f'MolecularDynamicsAss/Figures/Question23_{state}.png')
    plt.show()







if __name__ == '__main__':
    # question5()
    # plot_first_Qtune('MolecularDynamicsAss/Outputs/tuningQ3/')
    # question22Liquid()
    # question22Gas()
    question23Liquid('MolecularDynamicsAss/Outputs/Question22/Fluid/')
    question23Liquid('MolecularDynamicsAss/Outputs/Question22/Gas/')