import numpy as np
import scipy.constants as const
from typing import Tuple
import time
from numba import njit, prange

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

        self.kb = const.physical_constants['Boltzmann constant'][0]  # Boltzmann constant
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
            delta = self.pbc(self._particles - pos_i)
            d_sq = delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2
            d_sq[i] = np.inf
            d_sq[i:] = np.inf
            d_sq[d_sq > self.rcut ** 2] = np.inf

            d6 = d_sq ** 3
            d12 = d6 * d6
            energy += np.sum(4 * self.epsilon * (self.sigma12 / d12 - self.sigma6 / d6)) + self.tail_correction
            virial += np.sum(24 * self.epsilon * (self.sigma6 / d6 - 2 * self.sigma12 / d12))
        pressure = self.rho_num * self.kb * self.temp - virial / (3 * self.volume)
        # print(f"The total energy is: {energy}, the total pressure is: {pressure}")
        return energy, pressure

    def single_particle_energy(self, particle_index: int) -> float:
        delta = self.pbc(self._particles - self._particles[particle_index])
        d_sq = delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2
        d_sq[particle_index] = np.inf
        d_sq[particle_index:] = np.inf
        d_sq[d_sq > self.rcut ** 2] = np.inf
        d6 = d_sq ** 3
        d12 = d6 * d6
        return np.sum(4 * self.epsilon * (self.sigma12 / d12 - self.sigma6 / d6)) + self.tail_correction

    # @property
    # def energy(self) -> float:
    #     return self.total_energy_pressure[0]
    #
    # @property
    # def pressure(self) -> float:
    #     return self.total_energy_pressure[1]

    def pbc(self, delta: np.ndarray) -> np.ndarray:
        return delta - self.side_length * np.round(delta / self.side_length)

    def translate_particle(self) -> bool:
        # print("Translating Particle")
        i = np.random.randint(self.npart)
        old_position = self._particles[i].copy()

        displacement = np.random.uniform(-self.delta, self.delta, 3)

        new_position = self.pbc(old_position + displacement)

        e_old = self.single_particle_energy(i)
        self._particles[i] = new_position
        e_new = self.single_particle_energy(i)

        delta_e = e_new - e_old

        if delta_e < 0 or np.random.uniform(0, 1) < np.exp(-self.beta * delta_e):
            return True
        else:
            self._particles[i] = old_position
            return False

    def start_conf(self, nsteps: int = 50) -> np.ndarray:
        for i in range(nsteps * self.npart):
            print(f"Translation {i + 1}/{nsteps * self.npart}", end='\r', flush=True)
            self.translate_particle()
        return self._particles

    def run(self) -> Tuple[np.ndarray, np.ndarray, float]:
        # print("Running Monte Carlo Simulation")
        E_ave = np.zeros(self.ncycle)
        P_ave = np.zeros(self.ncycle)
        accepted_moves = 0

        self.start_conf(50)

        for i in range(self.ncycle):
            if self.translate_particle():
                accepted_moves += 1
            if i % self.ncycle == 0:
                start = time.time()
                E_ave[i], P_ave[i] = self.total_energy_pressure
                end = time.time()
                print(f"Iteration {i + 1}/{self.ncycle}, time: {end-start} seconds", end='\r', flush=True)
        acceptance_ratio = accepted_moves / self.ncycle

        return E_ave, P_ave, acceptance_ratio


if __name__ == '__main__':
    mc_sim = MonteCarloSimulation(npart=362, ncycle=5000, side_length=30, rcut=14)
    # print(mc_sim._particles)
    mc_sim.start_conf(50)
    E_ave, P_ave, acceptance_ratio = mc_sim.run()
    # mc_sim.plot(E_ave, P_ave)
    print(f"Acceptance Ratio: {acceptance_ratio}")
