import Simulation
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

sigma = 3.73
eta_kb = 148

# question 2.1
T_21 = 150
side_length_21 = 30
rcut_21 = 14
ncycles21 = 500000
npart = 362

delta_arr = np.linspace(0.005, 1, 25)
energy_arr = np.array([])
pressure_arr = np.array([])

def create_file_names(delta):
    rounded = f"{delta:.7f}"  # Round to 7 decimal places
    without_leading_zero = rounded[2:]  # Remove the leading "0."
    formatted_name = without_leading_zero.ljust(7, '0') + ".csv"  # Pad with zeros if necessary and add ".csv"
    return formatted_name


for delta in delta_arr:
    start = time.time()
    mc_21 = Simulation.MonteCarloSimulation(temp=T_21, npart=npart, side_length=side_length_21, rcut=rcut_21, eta_kb=eta_kb, sigma=sigma, ncycle=ncycles21, delta=delta)
    E_ave, P_ave, acceptance_ratio = mc_21.run()
    # energy_arr = np.append(energy_arr, E_ave[-1])
    # pressure_arr = np.append(pressure_arr, P_ave[-1])
    # print(f"delta = {delta}, acceptance ratio = {acceptance_ratio}")

    # filename = 'Ass21Outputs/first_run/' + str(round(delta, 6)).split('.')[-1] + '.csv'
    filename = create_file_names(delta)
    fields = ['Delta', 'Energy', 'Pressure', 'Acceptance Ratio']
    with open(filename, mode='w', newline='') as csfile:
        csv_writer = csv.writer(csfile, delimiter=',', quotechar='"')
        csv_writer.writerow(fields)
        for i in range(ncycles21):
            if E_ave[i] != 0 and P_ave[i] != 0:
                csv_writer.writerow([delta, E_ave[i], P_ave[i], acceptance_ratio])
    end = time.time()

    print(f"Delta: {round(delta, 5)}, acceptance ratio = {round(acceptance_ratio, 5)}, duration: {round(end - start, 5)} seconds \n")
print("excersice 2.1 done")




