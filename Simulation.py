import numpy as np
import scipy.constants as const
from typing import Tuple
import time

# np.random.seed(0)


class MonteCarloSimulation:
    def __init__(self, npart: int = 362, ncycle: int = 500000, temp: int = 150, side_length: float = 30.0, rcut: float = 14,
                 sigma: float = 3.73, eta_kb: int = 148, delta: float = 0.5, molar_m: float = 16.04, box_path: str = '') -> None:
        self.ncycle = ncycle  # Number of cycles
        self.npart = npart  # Number of particles
        self.molar_m = molar_m / 1000  # Molar mass of the system [kg]
        self.temp = temp  # Temperature
        self.side_length = side_length * 10 ** -10  # Side length of the box [m]
        self.volume = self.side_length ** 3  # Volume of the box [m^3]
        self.rcut = rcut * 10 ** -10  # Cutoff distance [m]
        self.rcut3 = self.rcut ** 3  # Cutoff distance cubed
        self.rcut9 = self.rcut ** 9

        self.box_path = box_path
        if self.box_path:
            print("Box path given")
            self._particles = self.get_coordinates() * 10 ** -10
            self.npart = len(self._particles)
            print("Particles created")
        else:
            print("No box path given")
            self._particles = np.round(np.random.uniform(0, self.side_length, (self.npart, 3)), 3)  # Position of the particles
            print("Particles created")

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
            delta = self.pbc(self._particles[i+1:, :] - pos_i)  # alleen de onderste driehoek van de matrix
            d_sq = np.sum((delta ** 2), axis=1)
            d_sq = d_sq[d_sq < self.rcut**2]
            d6 = d_sq ** 3
            d12 = d6 * d6
            energy += 4 * self.epsilon * np.sum(self.sigma12 / d12 - self.sigma6 / d6) + self.tail_correction
            virial += 24 * self.epsilon * np.sum(self.sigma6 / d6 - 2 * self.sigma12 / d12)
        pressure = self.rho_num * self.kb * self.temp - virial / (3 * self.volume)
        # print(f"The total energy is: {energy}, the total pressure is: {pressure}")
        return energy, pressure

    def single_particle_energy(self, particle_index: int) -> float:
        delta = self.pbc(self._particles - self._particles[particle_index])
        d_sq = np.sum((delta ** 2), axis=1)
        d_sq[particle_index] = self.side_length ** 2
        d_sq = d_sq[d_sq < self.rcut ** 2]
        d6 = d_sq ** 3
        d12 = d6 * d6
        return 4 * self.epsilon * np.sum(self.sigma12 / d12 - self.sigma6 / d6) + self.tail_correction

    def pbc(self, delta: np.ndarray) -> np.ndarray:
        return (delta + self.side_length/2) % self.side_length - self.side_length/2

    def translate_particle(self) -> bool:
        # print("Translating Particle")
        i = np.random.randint(self.npart)
        e_old = self.single_particle_energy(i)
        displacement = np.random.uniform(-self.delta, self.delta, 3)
        self._particles[i] = (self._particles[i] + displacement) % self.side_length
        e_new = self.single_particle_energy(i)

        delta_e = e_new - e_old

        if delta_e < 0 or np.random.uniform(0, 1) < np.exp(-self.beta * delta_e):
            return True
        else:
            self._particles[i] = (self._particles[i] - displacement) % self.side_length
            return False

    def get_coordinates(self) -> np.ndarray:
        coordinates = []
        with open(self.box_path, 'r') as file:
            lines = file.readlines()
            for line in lines[2:]:
                parts = line.split()
                if len(parts) == 4 and all(part.replace('.', '').isdigit() for part in parts[1:]):
                    x, y, z = map(float, parts[1:])
                    coordinates.append([x, y, z])
        return np.array(coordinates)

    def start_conf(self, nsteps: int = 50) -> np.ndarray:
        for i in range(nsteps * self.npart):
            print(f"Translation {i + 1}/{nsteps * self.npart}", end='\r', flush=True)
            self.translate_particle()
        return self._particles

    def run(self, start_conf: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        # print("Running Monte Carlo Simulation")
        E_ave = np.zeros(self.ncycle)
        P_ave = np.zeros(self.ncycle)
        accepted_moves = 0

        if self.box_path or not start_conf:
            self.start_conf(50)

        start = time.time()
        for i in range(self.ncycle):
            if self.translate_particle():
                accepted_moves += 1
                if (i % self.npart) == 0:
                    E_ave[i], P_ave[i] = self.total_energy_pressure
                end = time.time()
                print(f"Iteration {i + 1}/{self.ncycle}, time: {end-start} seconds", end='\r', flush=True)
        acceptance_ratio = accepted_moves / self.ncycle

        return E_ave, P_ave, acceptance_ratio


if __name__ == '__main__':
    pass
    # mc_sim = MonteCarloSimulation(npart=6, ncycle=5000, side_length=30, rcut=14)
    # print(mc_sim._particles)
    # mc_sim.start_conf(50)
    # E_ave, P_ave, acceptance_ratio = mc_sim.run()
    # mc_sim.plot(E_ave, P_ave)
    # print(f"Acceptance Ratio: {acceptance_ratio}")
