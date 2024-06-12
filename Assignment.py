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

Assignment_funcs.question21(T_21, side_length_21, rcut_21)

# directory_21 = 'Ass21Outputs/2024-06-05_11-58-44_150/'

# Assignment_funcs.plot_acceptance_ratio_vs_delta(directory_21, 'question21')


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
delta_23 = 0.461



directory_23 = 'Ass23Outputs/2024-06-03_14-06-40/'
#
Assignment_funcs.plot_energy_vs_cycle(directory_23, 'question23', 15)
# Assignment_funcs.question23(T_23, side_length_23, rcut_23, delta_23)


# %%
import Assignment_funcs

"""
Question 3.1 modeling a liquid CH4 system and running the mc simulation for different temperatures

Output: csv files with the energy, pressure and acceptance ratio for each temperature
"""
rcut_31 = 14
delta_31 = 0.461
rho31 = 358

directory_31 = 'Ass31Outputs/2024-06-11_12-03-23/'
Assignment_funcs.plot_energy_vs_cycle(directory_31, 'question31', 15)
# Assignment_funcs.question31(rcut_31, delta_31, rho31)

# %%
import Assignment_funcs
import matplotlib.pyplot as plt
"""
Question 3.2: modeling a gas CH4 system and running the mc simulation for different temperatures

Output: csv files with the energy, pressure and acceptance ratio for each temperature
"""

rcut_32 = 50
delta_32 = 21.87708
rho32 = 1.6
molar_mass = 16.04
npart32 = 362
side_length32 = Assignment_funcs.side_length_calc(molar_mass, npart32, rho32) * 10**10
# print(side_length32)

# acceptance_ratio_200, delta_200 = Assignment_funcs.get_acceptance_ratio('Ass32_tune_outputs/2024-06-11_14-10-06_200/')
# acceptance_ratio_300, delta_300 = Assignment_funcs.get_acceptance_ratio('Ass32_tune_outputs/2024-06-11_15-08-01_300/')
# acceptance_ratio_400, delta_400 = Assignment_funcs.get_acceptance_ratio('Ass32_tune_outputs/2024-06-11_16-11-48_400/')
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# ax.scatter(delta_200, acceptance_ratio_200, marker='x')
# ax.plot(delta_200, acceptance_ratio_200, linestyle='dashed')
#
# ax.scatter(delta_300, acceptance_ratio_300, marker='x')
# ax.plot(delta_300, acceptance_ratio_300, linestyle='dashed')
#
# ax.scatter(delta_400, acceptance_ratio_400, marker='x')
# ax.plot(delta_400, acceptance_ratio_400, linestyle='dashed')
#
#
# # put labels on the dots of the scatterplot but offset the labels such that they do not overlap
# # for (i, delta) in enumerate(delta):
# #     if i % 2 == 0:
# #         ax.annotate(f"{delta:.3f}", (delta, acceptance_ratio[i]), xytext=(0, 5), textcoords='offset points')

ax.set_xlabel('Delta')
ax.set_ylabel('Acceptance Ratio')
ax.set_title('Acceptance Ratio vs Delta')
plt.savefig('question32_tune.png')
plt.grid()
plt.show()


# tuning the new gas system first to find optimal delta
durations = [[169.54517650604248, 153.81104350090027, 148.8055922985077, 143.9403874874115, 141.1909544467926, 138.36850571632385, 157.0816888809204, 176.8054313659668, 140.51300764083862, 130.89160013198853, 132.80848383903503, 134.63036394119263, 133.99095726013184, 147.77583813667297, 140.18448424339294, 122.97052812576294, 129.7872760295868, 125.84979510307312, 126.77502822875977, 116.8486590385437, 127.39922404289246, 134.37216901779175, 134.32950401306152, 129.74159622192383, 136.481427192688], [172.0035753250122, 161.49578189849854, 174.5022735595703, 168.17542958259583, 143.3199577331543, 145.09136295318604, 139.68718600273132, 146.08571076393127, 129.00637078285217, 139.89176177978516, 197.7860176563263, 179.5569031238556, 160.81261563301086, 140.99705934524536, 176.04460835456848, 192.4294204711914, 156.38732314109802, 174.48443388938904, 136.27572107315063, 136.61940026283264, 131.25800681114197, 130.8469307422638, 131.64679408073425, 133.77717423439026, 128.56399202346802], [156.22976183891296, 152.51491045951843, 142.97871804237366, 141.9008231163025, 136.3796546459198, 138.89945316314697, 133.2537009716034, 133.74190163612366, 136.52301692962646, 135.48944211006165, 133.5197787284851, 136.15683150291443, 137.34443140029907, 144.8232524394989, 152.4843192100525, 134.6722445487976, 136.86120510101318, 137.22769832611084, 133.9440836906433, 131.24449253082275, 133.3521318435669, 133.71751832962036, 134.34743118286133, 137.6072235107422, 133.88652753829956]]

# make a boxplot of the durations
plt.boxplot(durations)
plt.show()


directory_32 = 'Ass32Outputs/2024-06-11_12-19-32/'
# Assignment_funcs.plot_energy_vs_cycle(directory_32, 'question32', 15)

# Assignment_funcs.question32(rho32, rcut_32, delta_32)

