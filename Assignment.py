import Assignment_funcs
import matplotlib.pyplot as plt
from datetime import datetime
"""
Question 2.1 modeling a liquid CH4 system and running the mc simulation 
for multiple dela values to determine the aoptimal displacement value
 
Output: csv files with the energy, pressure and acceptance ratio for each delta value
 """
T_21 = 150
side_length_21 = 30
rcut_21 = 14
npart = 362

# Assignment_funcs.question21(T_21, side_length_21, rcut_21)

directory_21 = 'Ass21Outputs/2024-06-12_10-10-01_150/'

Assignment_funcs.plot_acceptance_ratio_vs_delta(directory_21, 'question21')




#%%
import Assignment_funcs

"""
Question 2.2 modeling a gas CH4 system and running the mc simulation 
for multiple dela values to determine the aoptimal displacement value

Output: csv files with the energy, pressure and acceptance ratio for each delta value
"""

T_22 = 400
side_length_22 = 75
rcut_22 = 30

# Assignment_funcs.question22(T_22, side_length_22, rcut_22)
directory_22 = 'Ass22Outputs/2024-06-05_10-20-41_400/'
Assignment_funcs.plot_acceptance_ratio_vs_delta(directory_22, 'question22')

# %%
import Assignment_funcs
"""
Question 2.3 modeling a liquid CH4 using the optimal delta value from question 2.1
and running the mc simulation to find the equilibrium of the system

Output: csv files with the energy, pressure and acceptance ratio
"""

T_23 = 150
side_length_23 = 30
rcut_23 = 14
delta_23 = 0.736


directory_23 = 'Ass23Outputs/2024-06-13_11-37-00/'
directory_23_mil = 'Ass23Outputs/2024-06-13_12-08-46/'

Assignment_funcs.plot_energy_vs_cycle(directory_23, 'question23', 14)
# Assignment_funcs.question23(T_23, side_length_23, rcut_23, delta_23, ncycles=1000000)


# %%
import Assignment_funcs

"""
Question 3.1 modeling a liquid CH4 system and running the mc simulation for different temperatures

Output: csv files with the energy, pressure and acceptance ratio for each temperature
"""
rcut_31 = 14
delta_31 = 0.736
rho31 = 358.4

Assignment_funcs.question31(rcut_31, delta_31, rho31)

directory_31 = 'Ass31Outputs/2024-06-13_20-53-27/'
# Assignment_funcs.plot_acceptance_ratio_vs_delta(directory_31, 'question31')
Assignment_funcs.plot_energy_vs_cycle(directory_31, 'question31', 15)




 # %%
import Assignment_funcs
import matplotlib.pyplot as plt
import os
import pandas as pd
"""
Question 3.2: modeling a gas CH4 system and running the mc simulation for different temperatures

Output: csv files with the energy, pressure and acceptance ratio for each temperature
"""


# tuning the new gas system first to find optimal delta
durations = [[169.54517650604248, 153.81104350090027, 148.8055922985077, 143.9403874874115, 141.1909544467926, 138.36850571632385, 157.0816888809204, 176.8054313659668, 140.51300764083862, 130.89160013198853, 132.80848383903503, 134.63036394119263, 133.99095726013184, 147.77583813667297, 140.18448424339294, 122.97052812576294, 129.7872760295868, 125.84979510307312, 126.77502822875977, 116.8486590385437, 127.39922404289246, 134.37216901779175, 134.32950401306152, 129.74159622192383, 136.481427192688], [172.0035753250122, 161.49578189849854, 174.5022735595703, 168.17542958259583, 143.3199577331543, 145.09136295318604, 139.68718600273132, 146.08571076393127, 129.00637078285217, 139.89176177978516, 197.7860176563263, 179.5569031238556, 160.81261563301086, 140.99705934524536, 176.04460835456848, 192.4294204711914, 156.38732314109802, 174.48443388938904, 136.27572107315063, 136.61940026283264, 131.25800681114197, 130.8469307422638, 131.64679408073425, 133.77717423439026, 128.56399202346802], [156.22976183891296, 152.51491045951843, 142.97871804237366, 141.9008231163025, 136.3796546459198, 138.89945316314697, 133.2537009716034, 133.74190163612366, 136.52301692962646, 135.48944211006165, 133.5197787284851, 136.15683150291443, 137.34443140029907, 144.8232524394989, 152.4843192100525, 134.6722445487976, 136.86120510101318, 137.22769832611084, 133.9440836906433, 131.24449253082275, 133.3521318435669, 133.71751832962036, 134.34743118286133, 137.6072235107422, 133.88652753829956]]

# a function that puts 3 boxplots in the same figure and gives each one a label 200, 300 or 400 respectively
def boxplot(data):
    fig = plt.figure(figsize =(7, 5))
    ax = fig.add_subplot(111)
    bplot = ax.boxplot(data, patch_artist = True, labels=['0.40138512', '0.48107184', '0.5197459999999999'], widths=0.4)
    ax.set_title('Duration of the simulation for different temperatures')
    ax.set_xlabel('Acceptance ratio')
    ax.set_ylabel('Duration [s]')

    colours = [[0, 0, 0, 0.5],
               [0, 0, 1, 0.5],
               [1, 0, 0, 0.5]]
    for patch, color in zip(bplot['boxes'], colours):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    fig.savefig('durations_boxplot.png')
    plt.show()

boxplot(durations)

directory = 'Ass32_tune_outputs'
accetance_ratios = []
for dir in os.listdir(directory):
    accetance_ratios_temp = []
    for file in os.listdir(directory + '/' + dir):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + '/' + dir + '/' + file)
            accetance_ratios_temp.append(df['Acceptance Ratio'][0])
    accetance_ratios.append(accetance_ratios_temp)
print(accetance_ratios)

print([sum(x) / len(x) for x in accetance_ratios])


# acceptance_ratios = []
# deltas = []
# for file in os.listdir(directory):
#     if file.endswith(".csv"):
#         df = pd.read_csv(directory + file)
#
#         acceptance_ratios.append(df['Acceptance Ratio'][0])
#         deltas.append(df['Delta'][0])


