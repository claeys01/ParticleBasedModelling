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


def create_file_names(folder: str, delta: float):
    rounded = f"{delta:.7f}"  # Round to 7 decimal places
    without_leading_zero = rounded[2:]  # Remove the leading "0."
    formatted_name = folder + without_leading_zero.ljust(7, '0') + ".csv"  # Pad with zeros if necessary and add ".csv"
    return formatted_name


def question21(T, side_length, rcut, ncycles=500000):
    print("exercise 2.1: liquid system")
    delta_arr = np.linspace(0.005, side_length/2, 25)
    directory = 'Ass21Outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '_' + str(T) + '_' + '/'
    os.mkdir(directory)
    for (i, delta) in enumerate(delta_arr):
        print(i, delta)
        start = time.time()
        # box = 'box.xyz'
        mc_21 = Simulation.MonteCarloSimulation(temp=T, side_length=side_length, rcut=rcut,
                                                eta_kb=eta_kb, sigma=sigma, ncycle=ncycles, delta=delta)
        E_ave, P_ave, acceptance_ratio = mc_21.run()

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


# question 2.3
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


def get_acceptance_ratio(directory: str):
    acceptance_ratios = []
    deltas = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)

            acceptance_ratios.append(df['Acceptance Ratio'][0])
            deltas.append(df['Delta'][0])
    return np.array(acceptance_ratios), np.array(deltas)

# plot acceptance ratio vs delta
def plot_acceptance_ratio_vs_delta(directory: str, savefig: bool = False) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    acceptance_ratio, delta = get_acceptance_ratio(directory)
    ax.scatter(delta, acceptance_ratio)

    for (i, delta) in enumerate(delta):
        ax.annotate(f"{delta:.3f}", (delta, acceptance_ratio[i]))
    ax.set_xlabel('Delta')
    ax.set_ylabel('Acceptance Ratio')
    ax.set_title('Acceptance Ratio vs Delta')
    if savefig:
        plt.savefig(str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + 'AcceptanceRatioVsDelta.png')
    plt.grid()
    plt.show()

def plot_energy_vs_cycle(directory: str, savefig: bool = False) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)
            ax.plot(df.index[10:], (df['Energy'][10:]/362)*10**21, label=file)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Zepto Joule [10^-21] J/Atom')
    ax.set_title('Energy vs Cycle')
    ax.legend()
    if savefig:
        plt.savefig(str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + 'EnergyVsCycle.png')
    plt.grid()
    plt.show()
# plot_acceptance_ratio_vs_delta('Ass21Outputs/2024-05-31_16-51-56/')
def get_num_molecules(density: float, side_length: float, molar_mass: float) -> float:
    return math.ceil((density * (side_length * 10 ** -10) ** 3 / molar_mass) * const.N_A)

def side_length_calc(molar_mass: float, npart: int, rho: float) -> float:
    return (molar_mass/1000*npart/(rho*const.N_A))**(1/3)


def question31(rcut, delta, rho, ncycles=500000):
    molar_mass = 16.04
    npart = 362
    side_length = (molar_mass*npart/(rho*const.N_A))**(1/3)

    temp_arr = np.array([200, 300, 400])

    directory = 'Ass31Outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '/'
    os.mkdir(directory)

    for temp in temp_arr:
        start = time.time()
        mc = Simulation.MonteCarloSimulation(temp=temp, side_length=side_length, rcut=rcut, eta_kb=eta_kb, sigma=sigma, delta=delta)
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
        print(f"Temp: {temp}, Elapsed time: {end-start} seconds")


if __name__ == '__main__':
    print(((16.04 / 1000 * 362)/(358.4*const.N_A))**(1/3)*10**10)
    # question21(T_21, side_length_21, rcut_21, ncycles21)
    # question22(T_22, side_length_22, rcut_22)
    # question23(T_21, side_length_21, rcut_21, 0.461)
    # plot_energy_vs_cycle('Ass23Outputs/2024-06-03_14-06-40/')

    # plot_acceptance_ratio_vs_delta('Ass21Outputs/2024-05-31_16-51-56/')
