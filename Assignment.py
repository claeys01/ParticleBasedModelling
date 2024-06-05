import Assignment_funcs
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

Assignment_funcs.question22(T_22, side_length_22, rcut_22)

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

Assignment_funcs.question23(T_23, side_length_23, rcut_23, delta_23)


# %%
import Assignment_funcs

"""
Question 3.1 modeling a liquid CH4 system and running the mc simulation for different temperatures

Output: csv files with the energy, pressure and acceptance ratio for each temperature
"""
rcut_31 = 30
delta_31 = 0.461
rho31 = 358

Assignment_funcs.question31(rcut_31, delta_31, rho31)

# %%
import Assignment_funcs

"""
Question 3.2: modeling a gas CH4 system and running the mc simulation for different temperatures

Output: csv files with the energy, pressure and acceptance ratio for each temperature
"""

rcut_32 = 50
delta_32 = 21.87708
rho32 = 1.6

Assignment_funcs.question32(rho32, rcut_32, delta_32)

