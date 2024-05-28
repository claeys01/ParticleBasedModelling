import numpy as np
import scipy.constants as const
from typing import Tuple
import time
# from numba import njit, prange  # TODO: It will speed up for loops, but numpy array operations are much, much faster

"""
OVERAL COMMENTS:
    Your code looks pretty good. I love to see classes, functions and clear type definitions. Do not do it for this assignment,
    however, it would be better if you make your forcefield (and the energies and pressure calculations) a separate class.
    This would make it even more object oriented as that would make it flexible to add different forcefield types as well and
    even molecular bonds. Besides that, classes will not make your code slower (at least not in the way you are noticing).

    Then performance wise, it is not too bad. I added some comments, but I believe that the main problems boil down to three things:
        1. You double count in your total_energy_pressure() function.
        2. You place to np.inf, (divide by that later). This works fine, however you can drop the elements in the array larger than r_cut all together with a mask.
        3. You sometimes have code inside the sum that would be nice outside it.

    Besides that, there are some small comments. However, overall it is already quite good. It should take a few minutes for a single simulation of 1e9 trial moves.

    Then the final comment. There is something wrong with your final results. I changed some small things and can see the following problems/good things:
        1. At least the energy goes down from a random starting configuration (That is why I turned of the equilibration). So it looks like normal MC works as it minimizes energy a bit.
        2. However, your final pressure and energy are wrong. You should get something like -10 zJ/atom and approximately 100 bar (that can vary quite a bit though). There is either something wrong in the single_particle_energy() function or the total_energy_pressure() implementation.
        3. There are some features missing:
            a. You should be able to start form a box.xyz file (make this a class option).
            b. The assignment tells you that the input will be: density + temperature, so do also write a function that calculates the boxsize depending on the density.
"""

np.random.seed(0)


class MonteCarloSimulation:
    def __init__(self, npart: int = 362, ncycle: int = 500000, temp: int = 150, side_length: float = 30.0, rcut: float = 14,
                 sigma: float = 3.73, eta_kb: int = 148, delta: float = 0.5, molar_m: float = 16.04) -> None:
        self.ncycle = ncycle  # Number of cycles
        self.npart = npart  # Number of particles
        self.molar_m = molar_m / 1000  # Molar mass of the system [kg]
        self.temp = temp  # Temperature
        self.side_length = side_length * 10 ** -10  # Side length of the box [m]
        self.volume = self.side_length ** 3  # Volume of the box [m^3]
        self.rcut = rcut * 10 ** -10  # Cutoff distance [m]
        self.rcut3 = self.rcut ** 3  # Cutoff distance cubed
        self.rcut9 = self.rcut ** 9

        self._particles = np.random.uniform(0, self.side_length, (self.npart, 3))  # Position of the particles

        self.sigma6 = (sigma * 10 ** -10) ** 6  # Sigma6 value
        self.sigma12 = self.sigma6 * self.sigma6  # Sigma12 value

        self.kb = const.physical_constants['Boltzmann constant'][0]  # Boltzmann constant  # you can also just use `const.k` :) 
        self.rho_num = self.npart / self.volume  # Density of the system [m^-3]
        self.epsilon = eta_kb * self.kb  # Epsilon value [J]
        self.tail_correction = 8 * np.pi * self.rho_num * self.epsilon * (
                    self.sigma12 / (3 * self.rcut9) - self.sigma6 / (3 * self.rcut3))

        # self.tail_correction = 0

        self.delta = delta * 10 ** -10  # Maximum displacement of the particles [m]
        self.beta = 1 / (self.kb * self.temp)  # Beta value
        self.density = (self.npart / const.Avogadro * self.molar_m) / (self.volume)  # Density of the system [kg/m^3]

    @property
    def total_energy_pressure(self) -> Tuple[float, float]:
        energy = 0.0
        virial = 0.0
        for (i, pos_i) in enumerate(self._particles):
            delta = self.pbc(self._particles - pos_i)  # TODO: you take too many interactions
            d_sq = delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2  # TODO: See single energy comments
            d_sq[i] = np.inf  # TODO: You do not need this if you solve comment two lines up
            d_sq[i:] = np.inf  # TODO: The same goes for this
            d_sq[d_sq > self.rcut ** 2] = np.inf  # TODO: This works, but you can do it in a way where you mask away parts of the array

            d6 = d_sq ** 3
            d12 = d6 * d6
            energy += np.sum(4 * self.epsilon * (self.sigma12 / d12 - self.sigma6 / d6)) + self.tail_correction  # TODO: move multiplications out of sum
            virial += np.sum(24 * self.epsilon * (self.sigma6 / d6 - 2 * self.sigma12 / d12))  # TODO: move multiplications out of sum
        pressure = self.rho_num * self.kb * self.temp - virial / (3 * self.volume)
        energy *= 1e21/self.npart
        pressure *= 1e-5  # Return energy in zJ per atom and pressure in bar (makes comparisons easier)
        print(f"The total energy is: {energy}, the total pressure is: {pressure}")

        return energy, pressure

    def single_particle_energy(self, particle_index: int) -> float:
        delta = self.pbc(self._particles - self._particles[particle_index])
        d_sq = delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2  # TODO: Try d_sq = np.sum(delta**2, axis=1) and figure what it does
        d_sq[particle_index] = np.inf  # TODO: this is true, but it is cheaper to set it to the cutoff distance and remove that later as np.inf takes additional checks
        d_sq[particle_index:] = np.inf # TODO: This is wrong
        d_sq[d_sq > self.rcut ** 2] = np.inf # TODO: This can be faster
        """
        Replace your code block from the d_sq = delta with this:
            `
            d_sq = np.sum(delta**2, axis=1)  # dotproduct with itself
            d_sq[particle_index] = self.side_length  # set to boxlength
            sr2 = self.sigma**2/d_sq[d_sq < self.rcut**2]  # only use part of the array where the values are lower than the cutoff. To safe time, you can create a self.rcut2 and a self.sigma2 at startup
            sr6 = sr2**3
            sr12 = sr6**2
            ...ect..
            `
        """

        d6 = d_sq ** 3
        d12 = d6 * d6
        return np.sum(4 * self.epsilon * (self.sigma12 / d12 - self.sigma6 / d6)) + self.tail_correction  # TODO: 4*self.epsilon outside the sum safes a lot of computation time.
        # GOOD: tail correction only needed once per particle

    def pbc(self, delta: np.ndarray) -> np.ndarray:
        return delta - self.side_length * np.round(delta / self.side_length)

        # TODO: This is a oneline function you use in 1 location you can integrate it in your code
        # Additionally, there is a faster way of doing this (round is cheap, but your division is fast)
        # Try the following `return (delta + self.side_length/2) % self.side_length - self.side_length/2`
        # And figure out why this works.
    
    def translate_particle(self) -> bool:
        # print("Translating Particle")
        i = np.random.randint(self.npart)
        old_position = self._particles[i].copy()
        # TODO: Yes, you need a copy if you do it like that. HOWEVER, you do not need this.
        # You can use a single self._particles[i] for the entire function.
        # If you reject, you can just do self._particles[i] -= displacement to go back.
        # This saves memory allocations. See comment below.

        displacement = np.random.uniform(-self.delta, self.delta, 3)
        new_position = self.pbc(old_position + displacement) # TODO: See comment below

        e_old = self.single_particle_energy(i)
        self._particles[i] = new_position
        e_new = self.single_particle_energy(i)
        
        # that would mean the folowing:
        # e_old = self.single_particle_energy(i)
        # displacement = np.random.uniform(-self.delta, self.delta, 3)
        # self._particles[i] = (self._particles[i] + displacement) % self.side_length  # Why does this work?
        # e_new = self.single_particle_energy(i)

        delta_e = e_new - e_old
        if delta_e < 0 or np.random.uniform(0, 1) < np.exp(-self.beta * delta_e):
            return True
        else:
            self._particles[i] = old_position  # TODO: then it would be  self._particles[i] = (self._particles[i] - displacement) % self.side_length
            return False

    def start_conf(self, nsteps: int = 50) -> np.ndarray:
        for i in range(nsteps * self.npart):
            print(f"Translation {i + 1}/{nsteps * self.npart}", end='\r', flush=True)
            self.translate_particle()
        return self._particles

    def run(self) -> Tuple[np.ndarray, np.ndarray, float]:
        print("Running Monte Carlo Simulation")
        E_ave = np.zeros(self.ncycle)
        P_ave = np.zeros(self.ncycle)
        accepted_moves = 0

        # self.start_conf(50)  # TODO: This is double, you allso call it in the main code or make it a class option to enable this or not.€ý,€ý,
        for i in range(self.ncycle):
            start = time.time()
            if self.translate_particle():  # TODO: It is smarter to make self.translate_particle either return a 1 or a 0 for the accepted_moves or add that to the class as well
                accepted_moves += 1
            if i % 500 == 0:  # TODO: Although i is outside your True or False part, you still only sample if you have accepted.
                # TODO: THis is wrong. You should put this part indented back to a lower level.
                E_ave[i], P_ave[i] = self.total_energy_pressure
                end = time.time()
                print(f"Itera€ý,€ý,tion {i + 1}/{self.ncycle}, time: {end-start} seconds", end='\r', flush=True)
        acceptance_ratio = accepted_moves / self.ncycle
        return E_ave, P_ave, acceptance_ratio


if __name__ == '__main__':
    mc_sim = MonteCarloSimulation(npart=362, ncycle=500000, side_length=30, rcut=14)
    print(mc_sim._particles)
    # mc_sim.start_conf(50)  
    E_ave, P_ave, acceptance_ratio = mc_sim.run()
    # mc_sim.plot(E_ave, P_ave)
    print(f"Acceptance Ratio: {acceptance_ratio}")

