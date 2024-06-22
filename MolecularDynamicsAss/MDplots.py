import numpy as np
import matplotlib.pyplot as plt
import os

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

data = get_log('MolecularDynamicsAss/Outputs/tuningQ/1e+04thermostat_log.txt')

def plot_first_Qtune(directory):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    # a list with 17 different colors
    colors = [
        'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive',
        'cyan', 'magenta', 'yellow', 'teal', 'lime', 'navy', 'maroon', 'black'
    ]
    print(len(colors))
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            label ='Q: 1e'+ str(int(file[3:5]))
            data = get_log(directory + file)
            ax.plot(data[0], data[1], label=label, color=colors[int(file[3:5])-1])
    ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Temperature')
    ax.set_ylim(0, 500)
    plt.show()


plot_first_Qtune('MolecularDynamicsAss/Outputs/tuningQ/')