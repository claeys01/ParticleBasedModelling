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

# question 2.1
T_21 = 150
side_length_21 = 30
rcut_21 = 14
ncycles21 = 500000
npart = 362

delta_arr = np.linspace(0.005, 1, 25)

def create_file_names(folder: str, delta: float, acceptance_ratio: float):
    rounded = f"{delta:.7f}"  # Round to 7 decimal places
    without_leading_zero = rounded[2:]  # Remove the leading "0."
    formatted_name = folder + without_leading_zero.ljust(7, '0') + ".csv"  # Pad with zeros if necessary and add ".csv"
    return formatted_name

def question21(T, side_length, rcut, ncycles):
    delta_arr = np.linspace(0.005, 1, 25)

    directory = 'Ass21Outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '/'
    os.mkdir(directory)
    for (i, delta) in enumerate(delta_arr):
        print(i, delta)
        start = time.time()
        box = 'box.xyz'
        mc_21 = Simulation.MonteCarloSimulation(temp=T, side_length=side_length, rcut=rcut,
                                                eta_kb=eta_kb, sigma=sigma, ncycle=ncycles, delta=delta, box_path=box)
        E_ave, P_ave, acceptance_ratio = mc_21.run()

        filename = create_file_names(directory, delta)
        fields = ['Delta', 'Energy', 'Pressure', 'Acceptance Ratio']
        with open(filename, mode='w', newline='') as csfile:
            csv_writer = csv.writer(csfile, delimiter=',', quotechar='"')
            csv_writer.writerow(fields)
            for j in range(ncycles21):
                if E_ave[j] != 0 and P_ave[j] != 0:
                    csv_writer.writerow([delta, E_ave[j], P_ave[j], acceptance_ratio])
        end = time.time()

        print(f"{i}: Delta= {round(delta, 5)}, acceptance ratio = {round(acceptance_ratio, 5)}, duration = {round(end - start, 5)} seconds \n")
    print("excersice 2.1 done")


def get_num_molecules(density: float, side_length: float, molar_mass: float) -> float:
    return math.ceil((density * side_length**3 / molar_mass) * const.N_A)


def question22(T, side_length, rcut):
    print("exercise 2.2: liquid system")
    delta_arr = np.linspace(0.005, 1, 25)
    density = 9.68
    num_molecules = get_num_molecules(density, side_length, 16.04)

    directory = 'Ass22Outputs/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '/'
    os.mkdir(directory)

    for (i, delta) in enumerate(delta_arr):
        print(i, delta)
        start = time.time()
        mc_21 = Simulation.MonteCarloSimulation(temp=T, side_length=side_length, rcut=rcut,
                                                eta_kb=eta_kb, sigma=sigma, delta=delta, npart=num_molecules)


        E_ave, P_ave, acceptance_ratio = mc_21.run()

        filename = create_file_names(directory, delta)
        fields = ['Delta', 'Energy', 'Pressure', 'Acceptance Ratio']
        with open(filename, mode='w', newline='') as csfile:
            csv_writer = csv.writer(csfile, delimiter=',', quotechar='"')
            csv_writer.writerow(fields)
            for j in range(ncycles21):
                if E_ave[j] != 0 and P_ave[j] != 0:
                    csv_writer.writerow([delta, E_ave[j], P_ave[j], acceptance_ratio])
        end = time.time()

        print(
            f"{i}: Delta= {round(delta, 5)}, acceptance ratio = {round(acceptance_ratio, 5)}, duration = {round(end - start, 5)} seconds \n")
    print("excersice 2.2 done")


def get_acceptance_ratio(directory: str):
    acceptance_ratios = []
    deltas = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)

            acceptance_ratios.append(df['Acceptance Ratio'][0])
            deltas.append(df['Delta'][0])
    return np.array(acceptance_ratios), np.array(deltas)



