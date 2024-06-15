import math

import scipy.stats as stats

import Simulation
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.constants as const

import time
import os
from datetime import datetime
import pandas as pd
import CoolProp.CoolProp as CP

sigma = 3.73
eta_kb = 148


def question21(T, side_length, rcut, ncycles=500000):
    print("exercise 2.1: liquid system \n")
    delta_arr = np.linspace(0.005, side_length/2, 25)
    directory = 'Ass21Outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '_' + str(T) + '/'
    os.mkdir(directory)
    for (i, delta) in enumerate(delta_arr):
        start = time.time()
        box = 'box.xyz'
        mc_21 = Simulation.MonteCarloSimulation(temp=T, side_length=side_length, rcut=rcut,
                                                eta_kb=eta_kb, sigma=sigma, ncycle=ncycles, delta=delta, box_path=box)
        E_ave, P_ave, acceptance_ratio = mc_21.run(start_conf=True)

        filename = create_file_names(directory, delta)
        fields = ['Delta', 'Energy', 'Pressure', 'Acceptance Ratio']
        with open(filename, mode='w', newline='') as csfile:
            csv_writer = csv.writer(csfile, delimiter=',', quotechar='"')
            csv_writer.writerow(fields)
            for j in range(ncycles):
                if E_ave[j] != 0 and P_ave[j] != 0:
                    csv_writer.writerow([delta, E_ave[j], P_ave[j], acceptance_ratio])
        end = time.time()

        print(
            f"{i}: Delta= {round(delta, 5)}, acceptance ratio = {round(acceptance_ratio, 5)}, duration = {round(end - start, 5)} seconds \n")
    print("excersice 2.1 done")


def question22(T, side_length, rcut, ncycles=500000):
    print("exercise 2.2: gas system")
    delta_arr = np.linspace(0.005, side_length / 2, 25)
    density = 9.68
    num_molecules = get_num_molecules(density, side_length, 16.04 / 1000)
    print(f"num_molecules: {num_molecules}")

    directory = 'Ass22Outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '_' + str(T) + '/'
    os.mkdir(directory)

    for (i, delta) in enumerate(delta_arr):

        start = time.time()
        mc_22 = Simulation.MonteCarloSimulation(temp=T, side_length=side_length, rcut=rcut,
                                                eta_kb=eta_kb, sigma=sigma, delta=delta, npart=num_molecules)

        E_ave, P_ave, acceptance_ratio = mc_22.run()

        filename = create_file_names(directory, delta)
        fields = ['Delta', 'Energy', 'Pressure', 'Acceptance Ratio']
        with open(filename, mode='w', newline='') as csfile:
            csv_writer = csv.writer(csfile, delimiter=',', quotechar='"')
            csv_writer.writerow(fields)
            for j in range(ncycles):
                if E_ave[j] != 0 and P_ave[j] != 0:
                    csv_writer.writerow([delta, E_ave[j], P_ave[j], acceptance_ratio])
        end = time.time()

        print(
            f"{i}: Delta= {round(delta, 5)}, acceptance ratio = {round(acceptance_ratio, 5)}, duration = {round(end - start, 5)} seconds \n")
    print("excersice 2.2 done")


def question23(T, side_length, rcut, delta, ncycles=500000):
    print("exercise 2.3: liquid system")
    start = time.time()

    directory = 'Ass23Outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '/'
    os.mkdir(directory)
    mc_23 = Simulation.MonteCarloSimulation(temp=T, side_length=side_length, rcut=rcut,
                                            eta_kb=eta_kb, sigma=sigma, delta=delta, ncycle=ncycles)
    E_ave, P_ave, acceptance_ratio = mc_23.run(start_conf=False)

    filename = create_file_names(directory, delta)
    fields = ['Delta', 'Energy', 'Pressure', 'Acceptance Ratio']
    with open(filename, mode='w', newline='') as csfile:
        csv_writer = csv.writer(csfile, delimiter=',', quotechar='"')
        csv_writer.writerow(fields)
        for j in range(ncycles):
            if E_ave[j] != 0 and P_ave[j] != 0:
                csv_writer.writerow([delta, E_ave[j], P_ave[j], acceptance_ratio])
        end = time.time()

    print("excersice 2.3 done in ", end - start, " seconds \n")


def question31(rcut=14, delta=0.736, rho=358.4, ncycles=500000, side_length=30):
    print("exercise 3.1: liquid system")
    molar_mass = 16.04
    npart = get_num_molecules(rho, side_length, molar_mass/1000)
    side_length = side_length_calc(molar_mass, npart, rho) * 10**10
    print(f"side length: {side_length}, num_molecules: {npart} \n")

    temp_arr = np.array([200, 300, 400])

    directory = 'Ass31Outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '/'
    os.mkdir(directory)

    for temp in temp_arr:
        start = time.time()
        mc = Simulation.MonteCarloSimulation(temp=temp, ncycle=ncycles, side_length=side_length, rcut=rcut, eta_kb=eta_kb, sigma=sigma, delta=delta)
        energy, pressure, acceptance_ratio = mc.run(start_conf=False)
        filename = directory + f'{temp}.csv'
        fields = ['Delta', 'Energy', 'Pressure', 'Acceptance Ratio']
        with open(filename, mode='w', newline='') as csfile:
            csv_writer = csv.writer(csfile, delimiter=',', quotechar='"')
            csv_writer.writerow(fields)
            for j in range(ncycles):
                if energy[j] != 0 and pressure[j] != 0:
                    csv_writer.writerow([delta, energy[j], pressure[j], acceptance_ratio])
        end = time.time()
        print(f"Temp: {temp}, Elapsed time: {end-start} seconds")
    print("excersice 3.1 done")


def question32_tune(T, side_length, rcut, ncycles=500000) -> list[float]:
    print(f"exercise 3.2: gas system, T: {T}")
    delta_arr = np.linspace(0.005, side_length / 2, 25)
    durations = []
    directory = 'Ass32_tune_outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '_' + str(T) + '/'
    os.mkdir(directory)

    for (i, delta) in enumerate(delta_arr):

        start = time.time()
        mc_32 = Simulation.MonteCarloSimulation(temp=T, side_length=side_length, rcut=rcut,
                                                eta_kb=eta_kb, sigma=sigma, delta=delta, npart=362)

        E_ave, P_ave, acceptance_ratio = mc_32.run()

        filename = create_file_names(directory, delta)
        fields = ['Delta', 'Energy', 'Pressure', 'Acceptance Ratio']
        with open(filename, mode='w', newline='') as csfile:
            csv_writer = csv.writer(csfile, delimiter=',', quotechar='"')
            csv_writer.writerow(fields)
            for j in range(ncycles):
                if E_ave[j] != 0 and P_ave[j] != 0:
                    csv_writer.writerow([delta, E_ave[j], P_ave[j], acceptance_ratio])
        end = time.time()
        duration = end - start
        durations.append(duration)
        print(f"{i}: Delta= {round(delta, 5)}, acceptance ratio = {round(acceptance_ratio, 5)}, duration = {round(end - start, 5)} seconds \n")
    print("excersice 3.2 done")
    return durations


def question32(rho: float, rcut: float, ncycles=500000) -> None:
    print("exercise 3.2: gas system")

    start = time.time()
    molar_mass = 16.04
    npart = 362
    side_length = side_length_calc(molar_mass, npart, rho) * 10**10
    delta=side_length/2
    print(side_length)
    print(f"side length: {side_length}, num_molecules: {npart} \n")

    directory = 'Ass32Outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '/'
    os.mkdir(directory)

    temp_arr = np.array([200, 300, 400])
    for temp in temp_arr:
        start = time.time()
        mc = Simulation.MonteCarloSimulation(temp=temp, side_length=side_length, rcut=rcut, eta_kb=eta_kb, sigma=sigma,
                                             delta=delta)
        E_ave, P_ave, acceptance_ratio = mc.run(start_conf=False)
        filename = directory + f'{temp}.csv'

        fields = ['Delta', 'Energy', 'Pressure', 'Acceptance Ratio']
        with open(filename, mode='w', newline='') as csfile:
            csv_writer = csv.writer(csfile, delimiter=',', quotechar='"')
            csv_writer.writerow(fields)
            for j in range(ncycles):
                if E_ave[j] != 0 and P_ave[j] != 0:
                    csv_writer.writerow([delta, E_ave[j], P_ave[j], acceptance_ratio])
        end = time.time()
        print(f"Temp: {temp}, Elapsed time: {end - start} seconds \n")
    end2 = time.time()

    print("excersice 3.2 done in ", end2 - start, " seconds \n")


def get_acceptance_ratio(directory: str):
    acceptance_ratios = []
    deltas = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)

            acceptance_ratios.append(df['Acceptance Ratio'][0])
            deltas.append(df['Delta'][0])
    return np.array([acceptance_ratios for deltas, acceptance_ratios in sorted(zip(deltas, acceptance_ratios))]), np.sort(np.array(deltas))


def plot_acceptance_ratio_vs_delta(directory: str, prefix: str = 'q21', given_y: float = 0.35) -> None:
    acceptance_ratio, delta = get_acceptance_ratio(directory)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.scatter(delta, acceptance_ratio, marker='x')
    ax.plot(delta, acceptance_ratio, linestyle='dashed', color='red')

    # # Fit a 7th order polynomial to the data
    # coefficients = np.polyfit(delta, acceptance_ratio, 7)
    # # print(coefficients)
    #
    # # Define the polynomial equation minus the given y value
    # polynomial_minus_y = np.poly1d(coefficients - np.array([0, 0, 0, 0, 0, 0, 0, given_y]))
    #
    # # Find the roots of the polynomial equation
    # roots = polynomial_minus_y.roots
    #
    # # Filter real roots (ignore complex roots)
    # real_roots = roots[np.isreal(roots)].real
    #
    # print(f"The x values corresponding to y = {given_y} are: {real_roots}")
    # solution = real_roots.min()

    # put labels on the dots of the scatterplot but offset the labels such that they do not overlap
    # for (i, delta) in enumerate(delta):
    #     if i % 2 == 0:
    #         ax.annotate(f"{delta:.3f}", (delta, acceptance_ratio[i]), xytext=(0, 5), textcoords='offset points')
    # ax.annotate(f"Optimal Delta: {solution:.3f}", (solution, given_y), xytext=(0, 5), textcoords='offset points')
    # ax.scatter(solution, [given_y], color='red', marker='o')
    ax.set_xlabel('Delta [Å]')
    ax.set_ylabel('Acceptance Ratio')
    ax.set_title('Acceptance Ratio vs Delta')
    plt.grid()
    plt.savefig(prefix + '_' + 'AcceptanceRatioVsDelta.png')
    plt.show()


def plot_energy_vs_cycle(directory: str, prefix: str, startfrom: int) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    all_energies = []

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)
            energy = (np.array(df['Energy'][startfrom:]) / 362) * 10 ** 21
            energy_running_average = np.cumsum(energy) / (np.arange(len(energy)) + 1)
            all_energies.extend(energy)

            # Calculate the running standard deviation
            a = np.cumsum((energy - energy_running_average) ** 2)
            b = a / (np.arange(len(energy)) + 1)
            running_std = np.array([i ** 0.5 for i in b])

            # Calculate the upper and lower bounds
            lower_bound = np.array(energy_running_average - running_std, dtype=float)
            upper_bound = np.array(energy_running_average + running_std, dtype=float)

            cycle = np.linspace(0, 500000, len(energy), dtype=float)

            ax.plot(cycle, energy, alpha=0.8, label='Energy')
            ax.plot(cycle, energy_running_average, linestyle='dashed', color='red', label='Running Average')
            ax.plot(cycle, lower_bound, linestyle='dotted', color='gray')
            ax.plot(cycle, upper_bound, linestyle='dotted', color='gray')
            ax.fill_between(cycle, lower_bound, upper_bound, color='gray', alpha=0.2, label='Standard Deviation Bounds')

            # Add markers for the standard deviation at specific intervals
            interval = 43*2  # Interval for placing markers
            marker_indices = np.arange(0, len(energy), interval)
            ax.scatter(cycle[marker_indices], lower_bound[marker_indices], color='blue', marker='o', s=10,)
            ax.scatter(cycle[marker_indices], upper_bound[marker_indices], color='blue', marker='o', s=10)
            ax.scatter(cycle[marker_indices], energy_running_average[marker_indices], color='red', marker='x', s=30)

            for index in marker_indices:
                if index == 0:
                    continue
                ax.annotate(f"σ: {running_std[index]:.2f}", (cycle[index], upper_bound[index]),
                            xytext=(0, 5), textcoords='offset points')
                # annotate the avg energy
                if index % 2 == 0:
                    ax.annotate(f"e: {energy_running_average[index]:.2f}", (cycle[index], energy_running_average[index]),
                                xytext=(0, 5), textcoords='offset points', color='red')
                # else:
                #     ax.annotate(f"σ: {running_std[index]:.2f}", (cycle[index], upper_bound[index]),
                #                 xytext=(0, 5), textcoords='offset points')
                #
                ax.plot([cycle[index], cycle[index]], [lower_bound[index], upper_bound[index]], color='gray', linestyle='dotted')

    equilibrium_point = int(500000/150000)  # Example point, adjust based on your plot observation

    # Calculate average energy and error from equilibrium point to end
    equilibrium_energies = np.array(all_energies[startfrom:])[equilibrium_point:]
    average_energy = np.mean(equilibrium_energies)
    std_dev = np.std(equilibrium_energies)
    standard_error = std_dev / np.sqrt(len(equilibrium_energies))

    print(f"Average Energy: {average_energy:.5e} Zepto Joules")
    print(f"Standard Deviation: {std_dev:.5e} Zepto Joules")
    print(f"Standard Error: {standard_error:.5e} Zepto Joules")

    ax.set_xlabel('Cycle')
    ax.set_ylabel(r'Zepto Joule ($e^{-21}$) $\frac{J}{Atom}$ ')
    ax.set_title('Energy vs Cycle')
    plt.grid()
    # plt.legend()
    plt.savefig(prefix + '_EnergyVsCycle.png')
    plt.show()


def plot_energy_vs_cycle3(directory: str, prefix: str, startfrom: int) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    all_energies = []

    colours = ['black', 'blue', 'red']
    for i, file in enumerate(os.listdir(directory)):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)
            energy = (np.array(df['Energy'][startfrom:]) / 364) * 10 ** 21
            pressure = np.array(df['Pressure'][startfrom:])

            energy_running_average = np.cumsum(energy) / (np.arange(len(energy)) + 1)
            pressure_running_average = np.cumsum(pressure) / (np.arange(len(pressure)) + 1)
            all_energies.extend(energy)

            cycle = np.linspace(0, 500000, len(energy), dtype=float)

            # coolprop_energy = -CP.PropsSI('U', 'T', int(file[:3]), 'P', pressure_running_average[-1], 'Methane') * 358.4 * (30e-10)**3 / 364 * 10 ** 21
            coolprop_energy = -CP.PropsSI('U', 'D', 358.4, 'P', pressure_running_average[-1],
                                          'Methane') * 358.4 * (30e-10) ** 3 / 364 * 10 ** 21
            print(int(file[:3]))
            print(f"CoolProp Energy: {coolprop_energy:.5e} \n")

            coolprop_energy = np.array([coolprop_energy for _ in range(len(energy))])
            ax.plot(cycle, coolprop_energy, color=colours[i], label=f"{file[:3]} K")
            ax.plot(cycle, energy, alpha=0.3, color=colours[i])
            ax.plot(cycle, energy_running_average, linestyle='dashed', color=colours[i], label=f"{file[:3]} K")


    equilibrium_point = int(500000/150000)  # Example point, adjust based on your plot observation

    # Calculate average energy and error from equilibrium point to end
    equilibrium_energies = np.array(all_energies[startfrom:])[equilibrium_point:]
    average_energy = np.mean(equilibrium_energies)
    std_dev = np.std(equilibrium_energies)
    standard_error = std_dev / np.sqrt(len(equilibrium_energies))

    print(f"Average Energy: {average_energy:.5e} Zepto Joules")
    print(f"Standard Deviation: {std_dev:.5e} Zepto Joules")
    print(f"Standard Error: {standard_error:.5e} Zepto Joules")

    ax.set_xlabel('Cycle')
    ax.set_ylabel(r'Zepto Joule ($e^{-21}$) $\frac{J}{Atom}$ ')
    ax.set_title('Energy vs Cycle')
    plt.grid()
    plt.legend()
    plt.savefig(prefix + '_EnergyVsCycle.png')
    plt.show()


def plot_temperature_vs_pressure(directory: str, prefix: str, startfrom: int) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)


    pressures = []
    coolprop_pressures = []
    stdvs = []
    temperatures = []
    confidence_intervals = []
    error_margins = []

    nist_data = [43.178, 120.2, 190.18]

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)

            pressure = np.array(df['Pressure'][startfrom:])*10**-6
            temperature = int(file[:3])
            # print(f"Temperature: {temperature} K, acceptance ratio: {df['Acceptance Ratio'][0]}")

            pressure_mean = np.mean(pressure)
            pressure_std = np.std(pressure)
            pressure_error = pressure_std / np.sqrt(len(pressure))

            confidence_level = 0.95
            critical_value = stats.norm.ppf((1 + confidence_level) / 2)
            # print(len(pressure))

            pressure_error_margin = critical_value * pressure_error
            confidence_interval = (pressure_mean - pressure_error_margin, pressure_mean + pressure_error_margin)

            pressures.append(pressure_mean)
            stdvs.append(pressure_std)
            temperatures.append(temperature)
            confidence_intervals.append(confidence_interval)
            error_margins.append(pressure_error_margin)

            # coolprop calculations
            coolprop_pressure = CP.PropsSI('P', 'T', temperature, 'D', 358.4, 'Methane')*10**-6
            coolprop_pressures.append(coolprop_pressure)
            print(f"""
            Temperature: {temperature} K, acceptance ratio: {df['Acceptance Ratio'][0]}, difference: {(1 - coolprop_pressure/pressure_mean) *100} %
            Pressure: {pressure_mean:.2f} MPa, CoolProp Pressure: {coolprop_pressure:.2f} MPa
            Running Standard Deviation: {pressure_std:.2f}
            running Error: {pressure_error_margin:.2f}
            Confidence Interval: {pressure_mean} ± {critical_value}({pressure_std}/sqrt({len(pressure)}))
            """)

    ax.plot(temperatures, pressures, linestyle='dashed', color='blue', label='Simulation')
    ax.plot(temperatures, coolprop_pressures, alpha=0.7, linewidth=3.0, linestyle='dashed', color='green',
            label='CoolProp')
    ax.plot(temperatures, nist_data, alpha=0.7, linestyle='solid', color='black', label='NIST Data')
    ax.errorbar(temperatures, pressures, yerr=stdvs, fmt='o', color='red')
    for i, stdv in enumerate(stdvs):
        ax.annotate(f"σ: {stdv:.2f}", (temperatures[i] - 10, pressures[i] + stdv), xytext=(0, 5), textcoords='offset points')

    print(f"Confidence Intervals: {confidence_intervals}")
    # ax.errorbar(temperatures, pressures, yerr=error_margins, fmt='o', color='orange')
    ax.fill_between(temperatures, np.array(pressures) - np.array(error_margins), np.array(pressures) + np.array(error_margins), color='orange', alpha=0.7, label=' 95% Confidence Interval')
    ax.fill_between(temperatures, np.array(pressures) - np.array(stdvs), np.array(pressures) + np.array(stdvs), color='blue', alpha=0.1, label='Standard Deviation Bounds')


    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel(r'Pressure [Mpa] ')
    ax.set_title('Temperature vs Pressure')
    ax.legend()
    plt.grid()
    plt.savefig(prefix + '_TemperatureVsPressure.png')
    plt.show()


def plot_pressure_vs_cycle(directory: str, prefix: str, startfrom: int) -> None:
    fig = plt.figure(figsize=(10, 5), layout='tight')
    ax3 = fig.add_subplot(313)
    ax2 = fig.add_subplot(312, sharex=ax3)
    ax1 = fig.add_subplot(311, sharex=ax3)


    axis = [ax1, ax2, ax3]


    colours = ['black', 'blue', 'red']
    nist_data = [0.16412, 0.24773, 0.33119]
    coolprop_pressures = []

    for i, file in enumerate(os.listdir(directory)):
        if file.endswith(".csv"):
            ax = axis[i]
            df = pd.read_csv(directory + file)
            pressure = np.array(df['Pressure'][startfrom:]) * 10 ** -5
            pressure_running_average = np.cumsum(pressure) / (np.arange(len(pressure)) + 1)

            cycle = np.linspace(0, 500000, len(pressure), dtype=float)

            pressure_mean = np.mean(pressure)
            pressure_std = np.std(pressure)
            pressure_error = pressure_std / np.sqrt(len(pressure))

            confidence_level = 0.95
            critical_value = stats.norm.ppf((1 + confidence_level) / 2)
            # print(len(pressure))

            pressure_error_margin = critical_value * pressure_error
            # print(f"Pressure Error Margin: {pressure_error_margin:.5e}")
            confidence_interval = (pressure_mean - pressure_error_margin, pressure_mean + pressure_error_margin)

            coolprop_pressure_val = CP.PropsSI('P', 'T', int(file[:3]), 'D', 1.6, 'Methane') * 10 ** -5
            coolprop_pressures.append(coolprop_pressure_val)
            coolprop_pressure = np.array([coolprop_pressure_val for _ in range(len(cycle))])
            # print(f"CoolProp pressure: {coolprop_pressure_val:.5e} \n")

            print(f"""
                    Temperature: {int(file[:3])} K, acceptance ratio: {df['Acceptance Ratio'][0]}, difference: {((1 - coolprop_pressure_val / pressure_mean) * 100):.4f} %
                    Pressure: {pressure_mean:.4f} MPa, CoolProp Pressure: {coolprop_pressure_val:.4f} MPa
                    Running Standard Deviation: {pressure_std:.4f}
                    running Error: {pressure_error_margin:.4f}
                    Confidence Interval: {pressure_mean:.4f} ± {critical_value:.4f}({pressure_std:.4f}/sqrt({len(pressure)}) = )
                    """)

            ax.plot(cycle, pressure, alpha=0.3, color=colours[i])
            ax.plot(cycle, pressure_running_average, linestyle='dashed', color=colours[i], label=f"MC {file[:3]} K")
            ax.fill_between(cycle, pressure_running_average - pressure_error_margin, pressure_running_average + pressure_error_margin, color='orange', alpha=0.3, label='Error margin')

            ax.plot(cycle, coolprop_pressure, color=colours[i], linestyle='solid', label=f"CoolProp {file[:3]} K")
            # put the legend on the upper right
            ax.legend(loc='upper right', fontsize='small')
            ax.grid()
            ax.set_ylabel('Pressure [Bar]')
    ax3.set_xlabel('Cycle')
    plt.show()
    fig.savefig(prefix + '_PressureVsCycle.png')


def plot_temperature_vs_pressure32(directory: str, prefix: str, startfrom: int) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    pressures = []
    coolprop_pressures = []
    stdvs = []
    temperatures = []
    confidence_intervals = []
    error_margins = []

    nist_data = np.array([0.16412, 0.24773, 0.33119])*10**6

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)

            pressure = np.array(df['Pressure'][startfrom:])
            temperature = int(file[:3])
            # print(f"Temperature: {temperature} K, acceptance ratio: {df['Acceptance Ratio'][0]}")

            pressure_mean = np.mean(pressure)
            pressure_std = np.std(pressure)
            pressure_error = pressure_std / np.sqrt(len(pressure))

            confidence_level = 0.95
            critical_value = stats.norm.ppf((1 + confidence_level) / 2)
            # print(len(pressure))

            pressure_error_margin = critical_value * pressure_error
            confidence_interval = (pressure_mean - pressure_error_margin, pressure_mean + pressure_error_margin)

            pressures.append(pressure_mean)
            stdvs.append(pressure_std)
            temperatures.append(temperature)
            confidence_intervals.append(confidence_interval)
            error_margins.append(pressure_error_margin)

            # coolprop calculations
            coolprop_pressure = CP.PropsSI('P', 'T', temperature, 'D', 1.6, 'Methane')
            coolprop_pressures.append(coolprop_pressure)

            print(f"""
            Temperature: {temperature} K, acceptance ratio: {df['Acceptance Ratio'][0]}, difference: {(1 - coolprop_pressure/pressure_mean) *100} %
            Pressure: {pressure_mean:.2f} MPa, CoolProp Pressure: {coolprop_pressure:.2f} MPa
            Running Standard Deviation: {pressure_std:.2f}
            running Error: {pressure_error_margin:.2f}
            Confidence Interval: {pressure_mean} ± {critical_value}({pressure_std}/sqrt({len(pressure)}))
            """)

    ax.plot(temperatures, pressures, linestyle='dashed', color='blue', label='Simulation')
    ax.errorbar(temperatures, pressures, yerr=stdvs, fmt='o', color='red')
    ax.fill_between(temperatures, np.array(pressures) - np.array(stdvs), np.array(pressures) + np.array(stdvs), color='blue', alpha=0.1, label='Standard Deviation Bounds')
    for i, stdv in enumerate(stdvs):
        ax.annotate(f"σ: {stdv:.2f}", (temperatures[i] - 10, pressures[i] + stdv), xytext=(0, 5), textcoords='offset points')

    print(f"Confidence Intervals: {confidence_intervals}")
    # ax.errorbar(temperatures, pressures, yerr=error_margins, fmt='o', color='orange')
    ax.fill_between(temperatures, np.array(pressures) - np.array(error_margins), np.array(pressures) + np.array(error_margins), color='orange', alpha=0.7, label=' 95% Confidence Interval')

    ax.plot(temperatures, coolprop_pressures, alpha=0.7,linewidth=3.0, linestyle='dashed', color='green', label='CoolProp')
    ax.plot(temperatures, nist_data, alpha=0.7, linestyle='solid', color='black', label='NIST Data')
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel(r'Pressure [Mpa] ')
    ax.set_title('Temperature vs Pressure')
    ax.legend()
    plt.grid()
    plt.savefig(prefix + '_TemperatureVsPressure.png')
    plt.show()


def get_num_molecules(density: float, side_length: float, molar_mass: float) -> int:
    return math.ceil((density * (side_length * 10 ** -10) ** 3 / molar_mass) * const.N_A)


def side_length_calc(molar_mass: float, npart: int, rho: float) -> float:
    return (molar_mass/1000*npart/(rho*const.N_A))**(1/3)


def create_file_names(folder: str, delta: float):
    rounded = f"{delta:.5f}"  # Round to 7 decimal places
    number = '_'.join(rounded.split('.'))  # Replace the "." with "_"
    formatted_name = folder + number + ".csv"  # Pad with zeros if necessary and add ".csv"
    return formatted_name


if __name__ == '__main__':
    # number = 34.12345678
    # print(number)
    # string = f"{number:.5f}"
    # print(string)
    # split_str = string.split('.')
    # print(split_str)
    # str_join = '_'.join(split_str)
    # print(str_join)
    # print(((16.04 / 1000 * 362)/(358.4*const.N_A))**(1/3)*10**10)
    # question21(T_21, side_length_21, rcut_21, ncycles21)
    # question22(T_22, side_length_22, rcut_22)
    # question23(T_21, side_length_21, rcut_21, 0.461)
    # plot_energy_vs_cycle('Ass23Outputs/2024-06-03_14-06-40/')

    # plot_acceptance_ratio_vs_delta('Ass32_tune_outputs/2024-06-11_14-10-06_200/', 'question32_tune_200', 0.35)
    # plot_acceptance_ratio_vs_delta('Ass32_tune_outputs/2024-06-11_15-08-01_300/', 'question32_tune_300', 0.35)
    # plot_acceptance_ratio_vs_delta('Ass32_tune_outputs/2024-06-11_16-11-48_400/', 'question32_tune_400', 0.35)

    # question31(14, 0.736, 358.4)
    # plot_energy_vs_cycle3('Ass31Outputs/2024-06-13_20-53-27/', 'question31', 12)
    # plot_temperature_vs_pressure('Ass31Outputs/2024-06-13_20-53-27/', 'question31', 12)
    plot_pressure_vs_cycle('Ass32Outputs/2024-06-13_22-16-17/', 'question32', 12)
    # question32(1.6, 50, 181.97595888345467 / 2)

    # plot_pressure_vs_cycle('Ass32Outputs/2024-06-11_17-09-27/', 'question32', 12)
    # plot_temperature_vs_pressure32('Ass32Outputs/2024-06-13_22-16-17/', 'question32', 12)


    # plot_temperature_vs_energy('Ass31Outputs/2024-06-13_20-53-27/', 'question31')

