import math

import Simulation
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.constants as const

import time
import os
from datetime import datetime
import pandas as pd

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
                                            eta_kb=eta_kb, sigma=sigma, delta=delta)
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


def question31(rcut, delta, rho, ncycles=500000):
    print("exercise 3.1: liquid system")
    molar_mass = 16.04
    npart = 362
    side_length = side_length_calc(molar_mass, npart, rho) * 10**10
    print(side_length)

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

def question32_tune(T, side_length, rcut, num_molecules, ncycles=500000) -> list[float]:
    print(f"exercise 3.2: gas system, T: {T}")
    delta_arr = np.linspace(0.005, side_length / 2, 25)
    print(f"num_molecules: {num_molecules}")
    durations = []
    directory = 'Ass32_tune_outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '_' + str(T) + '/'
    os.mkdir(directory)

    for (i, delta) in enumerate(delta_arr):

        start = time.time()
        mc_32 = Simulation.MonteCarloSimulation(temp=T, side_length=side_length, rcut=rcut,
                                                eta_kb=eta_kb, sigma=sigma, delta=delta, npart=num_molecules)

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

def question32(rho: float, rcut: float, delta: float, ncycles=500000) -> None:
    print("exercise 3.2: gas system")
    start = time.time()

    molar_mass = 16.04
    npart = 362
    # side_length = (molar_mass * npart / (rho * const.N_A)) ** (1 / 3)
    side_length = side_length_calc(molar_mass, npart, rho) * 10**10
    print(side_length)
    temp_arr = np.array([200, 300, 400])


    directory = 'Ass32Outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '/'
    os.mkdir(directory)

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

def plot_acceptance_ratio_vs_delta(directory: str, prefix: str = 'q21'):
    acceptance_ratio, delta = get_acceptance_ratio(directory)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.scatter(delta, acceptance_ratio, marker='x')
    ax.plot(delta, acceptance_ratio, linestyle='dashed', color='red')

    # put labels on the dots of the scatterplot but offset the labels such that they do not overlap
    for (i, delta) in enumerate(delta):
        if i % 2 == 0:
            ax.annotate(f"{delta:.3f}", (delta, acceptance_ratio[i]), xytext=(0, 5), textcoords='offset points')

    ax.set_xlabel('Delta [Å]')
    ax.set_ylabel('Acceptance Ratio')
    ax.set_title('Acceptance Ratio vs Delta')
    plt.grid()
    plt.savefig(prefix + '_' + 'AcceptanceRatioVsDelta.png')
    plt.show()

def plot_energy_vs_cycle(directory: str, prefix: str, startfrom: int) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)
            energy = (np.array(df['Energy'][startfrom:])/362)*10**21
            energy_running_average = np.cumsum(energy) / (np.arange(len(energy)) + 1)

            # Calculate the running standard deviation
            a = np.cumsum((energy - energy_running_average)**2)
            b = a / (np.arange(len(energy)) + 1)

            running_std = np.array([i**0.5 for i in b])
            # running_std = np.array(running_std.tolist())

            # Calculate the upper and lower bounds
            lower_bound = np.array(energy_running_average - running_std, dtype=float)
            upper_bound = np.array(energy_running_average + running_std, dtype=float)

            for item in upper_bound:
                if type(item) != float:
                    print(item)

            cycle = np.linspace(0, 500000, len(energy), dtype=float)
            # cycle = [float(i) for i in cycle]


            for i in range(len(energy)):
                if math.isnan(energy[i]) or math.isnan(energy_running_average[i]) or math.isnan(lower_bound[i]) or math.isnan(upper_bound[i]):
                    print(f"energy: {energy[i]}, energy_running_average: {energy_running_average[i]}, lower_bound: {lower_bound[i]}, upper_bound: {upper_bound[i]}")

                    if not isinstance(energy[i], float) or not isinstance(energy_running_average[i], float) or not isinstance(lower_bound[i], float) or not isinstance(upper_bound[i], float):
                        print("Not a float")

                # print(np.isinfinite(lower_bound[i])

            ax.plot(cycle, energy, alpha=0.7, label='Energy')
            ax.plot(cycle, energy_running_average, linestyle='dashed', color='red', label='Running Average')
            ax.plot(cycle, lower_bound, linestyle='dotted', color='gray', label='Lower Bound')
            ax.plot(cycle, upper_bound, linestyle='dotted', color='gray', label='Upper Bound')
            ax.fill_between(cycle, lower_bound, upper_bound, color='gray', alpha=0.3, label='Standard Deviation Bounds')



    ax.set_xlabel('Cycle')
    ax.set_ylabel('Zepto Joule [10^-21] J/Atom')
    ax.set_title('Energy vs Cycle')
    plt.grid()
    plt.savefig(prefix + '_EnergyVsCycle.png')
    plt.show()

def get_num_molecules(density: float, side_length: float, molar_mass: float) -> float:
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

    # plot_acceptance_ratio_vs_delta('Ass22Outputs/2024-06-05_10-20-41_400/')
    plot_energy_vs_cycle('Ass23Outputs/2024-06-03_14-06-40/', 'test', 15)

