# import MD.py
import MonteCarloAss.Simulation as MC
import scipy.constants as const

import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime


class MolecularDynamicsSimulation:

    def __init__(self, side_length: float = 30, rho: float = 358.4, molar_mass: float = 16.04, inital_temp: float = 150,
                 ndim: int = 3, rcut: float = 10, dt: int = 1, eta_kb: float = 148, t_end: int = 3000, Q: float = 100.0) -> None:
        self.sigma = 3.73 * 1e-10  # [m]
        self.sigma6 = self.sigma ** 6
        self.sigma12 = self.sigma6 ** 2

        self.kb = const.physical_constants['Boltzmann constant'][0]  # J/K
        # print(f"Boltzmann constant: {self.kb} kJ/K")
        self.NA = const.Avogadro  # 1/mol
        self.epsilon = eta_kb * self.kb  # [J]
        self.M = molar_mass * 1e-3  # kg/mol
        self.ndim = ndim
        self.R = self.NA * self.kb  # J/(mol K)
        self.initial_temp = inital_temp  # K
        self.rcut = rcut * 1e-10  # [m]
        self.rcut3 = self.rcut ** 3
        self.rcut9 = self.rcut ** 9

        self.dt = dt * 1e-15  # timestep in femto seconds
        self.side_length = side_length * 1e-10  # [m]
        self.volume = self.side_length ** 3  # [m^3]
        self.t_end = t_end

        self.npart = MolecularDynamicsSimulation.get_num_molecules(rho, self.side_length * 1e10, self.M * 1e3)
        self.rho_num = self.npart / self.volume  # Density of the system [m^-3]
        self.tail_correction = 8 * np.pi * self.rho_num * self.epsilon * (
                self.sigma12 / (3 * self.rcut9) - self.sigma6 / (3 * self.rcut3))  # Tail correction

        self.rho = rho  # kg/m^3
        self.mass = self.rho * self.side_length ** 3 / self.npart  # kg

        self.zeta = 0  # Thermostat variable
        self.Q = Q  # Damping parameter

        print(f"""Simulation parameters:
    Boltzmann constant: {self.kb * 1e-3} kJ/K
    Avogadro constant: {self.NA} 1/mol
    Side length: {self.side_length * 1e10} Å
    Density: {self.rho} kg/m^3
    mass: {self.mass} kg
    Initial temperature: {self.initial_temp} K
    Number of dimensions: {self.ndim}
    Number of particles: {self.npart}
    Cutoff distance: {self.rcut * 1e10} Å
    Timestep: {dt} fs
    Epsilon: {self.epsilon} J
    End time: {t_end} fs
        """)

        self._particles = self.initGrid()
        self._velocities = self.initVel()
        self._forces = self.LJ_forces()

    @staticmethod
    def get_num_molecules(density: float, side_length: float, molar_mass: float) -> int:
        return math.floor((density * (side_length * 10 ** -10) ** 3 / (molar_mass / 1000)) * const.N_A)

    def density_conversion(self, rho: float) -> float:
        """
        Convert density to particles per Å^3
        rho in kg/m^3
        M in g/mol

        Returns: density in particles per Å^3
        """
        # Convert rho to g/Å^3
        # rho_g_per_A3 = rho * 1e-24  # 1 kg/m^3 = 1e-24 g/Å^3
        # return rho_g_per_A3 * self.NA / self.M  # particles per Å^3
        return rho * 1000 / 10 ** 30 * self.NA

    def rdf(self, n_bins: int = 100, r_range: tuple[float] = (0.01, 10)):
        pass

    def initGrid(self) -> np.ndarray:
        print("Initalising grid")

        # Initialize particle positions
        positions = np.zeros((self.npart, self.ndim))

        # Simple lattice filling (needs improvement to avoid overlap)
        spacing = self.side_length / (self.npart) ** (1 / self.ndim)  # Assume cubic root for spacing
        for i in range(self.npart):
            if self.ndim == 1:
                positions[i] = [i * spacing]
            elif self.ndim == 2:
                positions[i] = [(i % int(np.sqrt(self.npart))) * spacing,
                                (i // int(np.sqrt(self.npart))) * spacing]
            elif self.ndim == 3:
                positions[i] = [(i % int(np.cbrt(self.npart))) * spacing,
                                ((i // int(np.cbrt(self.npart))) % int(np.cbrt(self.npart))) * spacing,
                                (i // (int(np.cbrt(self.npart)) ** 2)) * spacing]
        print(f"{self.ndim}D grid initalised with {self.npart} particles")
        return positions - self.side_length / 2

    def initVel(self) -> np.ndarray[float]:
        print("Initalising velocities")

        # determine direction of velocities
        velocities_directions = np.random.normal(size=(self.npart, self.ndim))
        velocities_norms = np.linalg.norm(velocities_directions, axis=1)
        velocities_directions = velocities_directions / velocities_norms[:, np.newaxis]

        # determine magnitude of velocities
        velocities_magnitude = np.sqrt(self.ndim * self.kb * self.initial_temp / self.mass)

        velocities = velocities_directions * velocities_magnitude
        velocities -= np.mean(velocities, axis=0)  # Ensure zero net momentum
        print(f"Mean absolute velocity: {np.mean(np.abs(velocities))} m/s")
        print(f"Velocities initalised with a temperature of {self.initial_temp} K")
        # print(velocities)
        return velocities

    def pbc(self, delta: np.ndarray) -> np.ndarray:
        return (delta + self.side_length / 2) % self.side_length - self.side_length / 2

    def LJ_forces(self) -> np.ndarray:
        forces = np.zeros((self.npart, self.ndim))
        for (i, pos_i) in enumerate(self._particles):
            delta = self.pbc(self._particles[:, :] - pos_i)
            d_sq = np.sum((delta ** 2), axis=1).reshape(self.npart, 1)
            d_sq[d_sq > self.rcut ** 2] = np.inf
            d_sq[i] = np.inf
            d_sq[d_sq < self.sigma ** 2] = self.sigma ** 2
            d6 = d_sq ** 3
            d12 = d6 * d6
            forces[i] += np.sum((self.sigma6 / d6 - 2 * self.sigma12 / d12)
                                .reshape(self.npart, 1) * delta / np.sqrt(d_sq), axis=0)
        return forces

    @property
    def temperature(self) -> float:
        """
        :return: the temperature of the system in Kelvin
        """
        vmag = np.linalg.norm(self._velocities, axis=1)  # m/s
        return self.mass * np.mean(vmag ** 2) / (self.ndim * self.kb)  # K
        # return self.mass * np.mean(np.linalg.norm(self._velocities, axis=1) ** 2) / (self.ndim * self.kb)  # K


    @property
    def kineticEnergy(self) -> float:
        """
        Calculate the kinetic energy of the system in J
        """
        return 0.5 * self.mass * np.sum(np.linalg.norm(self._velocities,axis=1) ** 2)  # J
        # return 0.5 * np.sum(self.mass * np.sum(self._velocities ** 2, axis=1))  # J

    @property
    def potentialEnergy(self) -> float:
        energy = 0.0
        for (i, pos_i) in enumerate(self._particles):
            delta = self.pbc(self._particles[i + 1:, :] - pos_i)
            d_sq = np.sum((delta ** 2), axis=1)
            d_sq = d_sq[d_sq < self.rcut ** 2]
            d_sq[d_sq < self.sigma ** 2] = self.sigma ** 2
            d6 = d_sq ** 3
            d12 = d6 * d6
            energy += 4 * self.epsilon * np.sum(self.sigma12 / d12 - self.sigma6 / d6) + self.tail_correction
        return energy

    @property
    def temperature(self) -> float:
        """

        :return: the temperature of the system in Kelvin
        """

        return 2.0 * self.kineticEnergy / (self.ndim * self.kb * self.npart)  # K

    @property
    def pressure(self) -> float:
        virial_contribution = 0.0
        kinetic_contribution = 0.0
        for i, pos_i in enumerate(self._particles):
            delta = self.pbc(self._particles[i + 1:, :] - pos_i)
            d_sq = np.sum((delta ** 2), axis=1)
            d_sq[d_sq > self.rcut ** 2] = np.inf
            d_sq[d_sq < self.sigma ** 2] = self.sigma ** 2
            d6 = d_sq ** 3
            d12 = d6 * d6
            virial_contribution += 24 * self.epsilon * np.sum(self.sigma6 / d6 - 2 * self.sigma12 / d12)
            kinetic_contribution += np.sum(self._velocities[i] ** 2) * self.mass
        return (virial_contribution / 6 + kinetic_contribution) / self.volume

    def write_frame(self, trajectory_name, step):
        '''
        function to write trajectory file in LAMMPS format

        In VMD you can visualize the motion of particles using this trajectory file.

        :param coords: coordinates
        :param vels: velocities
        :param forces: forces
        :param trajectory_name: trajectory filename

        :return:
        '''

        with open(trajectory_name, 'a') as file:
            file.write('ITEM: TIMESTEP\n')
            file.write('%i\n' % step)
            file.write('ITEM: NUMBER OF ATOMS\n')
            file.write('%i\n' % self.npart)
            file.write('ITEM: BOX BOUNDS pp pp pp\n')
            for dim in range(self.ndim):
                file.write('%.6f %.6f\n' % (-0.5 * self.side_length * 1e10, 0.5 * self.side_length * 1e10))
            for dim in range(3 - self.ndim):
                file.write('%.6f %.6f\n' % (0, 0))
            file.write('ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz\n')

            temp = np.zeros((self.npart, 9))
            for dim in range(self.ndim):
                temp[:, dim] = self._particles[:, dim] * 1e10
                temp[:, dim + 3] = self._velocities[:, dim] * 1e-5
                temp[:, dim + 6] = self._forces[:, dim] / 4184 * self.NA

            for part in range(self.npart):
                file.write('%i %i %.4e %.4e %.4e %.6e %.6e %.6e %.4e %.4e %.4e\n' % (part + 1, 1, *temp[part, :]))

    def velocityVerlet(self) -> None:
        """
        Performs 1 step of the verlet algorithm
        """
        # print("Performing velocity verlet")
        # retrieve the LJ forces for the current positions
        forces = self._forces

        # Update positions
        self._particles += self._velocities * self.dt + 0.5 * (forces / self.mass) * (self.dt ** 2)

        # self._particles = self.pbc(self._particles)

        # Compute new forces
        new_forces = self.LJ_forces()

        # Update velocities
        self._velocities += 0.5 * (new_forces + forces) / self.mass * self.dt

        # Update forces
        self._forces = new_forces

    def velocityVerletThermostat(self) -> None:
        """
        Performs 1 step of the velocity verlet algorithm with Nosé-Hoover thermostat
        """
        # Retrieve the LJ forces for the current positions
        forces = self._forces
        energy = self.kineticEnergy
        temp = self.temperature

        # Update positions
        self._particles += self._velocities * self.dt + 0.5 * (forces / self.mass - self.zeta * self._velocities) * (self.dt ** 2)

        # Intermediate velocity update
        velocities_t2 = self._velocities + 0.5 * (forces / self.mass - self.zeta * self._velocities) * self.dt

        # intermediate thermostat variable
        zeta_t2 = self.zeta + 0.5 * self.dt / self.Q * (energy - 1.5 * (self.npart + 1) * self.kb * temp)

        # self._particles = self.pbc(self._particles)

        # Compute new forces
        new_forces = self.LJ_forces()

        # Update thermostat variable again
        self.zeta = zeta_t2 + 0.5 * self.dt / self.Q * (energy - 1.5 * (self.npart + 1) * self.kb * temp)

        # Final velocity update
        self._velocities = (velocities_t2 + 0.5 * self.dt * (new_forces / self.mass)) / (1 + 0.5 * self.dt * self.zeta)

        # Update forces
        self._forces = new_forces

    def run(self):
        # write outputs to a file in outputs directory
        filename = 'C:\\Users\\Matth\\Master\\Y1\\Q4\\particleBasedModeling\\MolecularDynamicsAss\\Outputs\\' + "_trajectory.lammps_nopbc_temp_correct"
        for t in range(self.t_end):
            self.velocityVerlet()
            if t % 100 == 0:
                print(
f"""
\r{t}|  Temperature: {self.temperature:.6} Kelvin
        Kinetic energy: {self.kineticEnergy / 4184 * self.NA:.4f} kCa/mol
        Mean absolute velocity  [{np.mean(np.abs(self._velocities)) * 1e-5:.3e}] Å/fs
        Mean absolute force: [{np.mean(np.abs(self._forces))/ 4184 * self.NA:.3e}] kcal/mol/Å
        """, end='', flush=True)
                self.write_frame(filename, t)

        print("Simulation completed")

    def runThermostat(self):
        # write outputs to a file in outputs directory
        filename = 'C:\\Users\\Matth\\Master\\Y1\\Q4\\particleBasedModeling\\MolecularDynamicsAss\\Outputs\\' + "thermostat_trajectory.lammps"
        for t in range(self.t_end):
            self.velocityVerletThermostat()
            if t % 100 == 0:
                print(
    f"""
\r{t}|
Temperature: {self.temperature:.6} Kelvin
Kinetic energy: {self.kineticEnergy / 4184 * self.NA:.4f} kCa/mol
Mean absolute velocity: [{np.mean(np.abs(self._velocities)) * 1e-5:.3e}] Å/fs
Mean absolute force: [{np.mean(np.abs(self._forces))/ 4184 * self.NA:.3e}] kcal/mol/Å
""", end='', flush=True)
                self.write_frame(filename, t)

        print("Simulation completed")

if __name__ == "__main__":
    # MD = MolecularDynamicsSimulation(10, 1000, 16.04, 150, 3, 6, 0.001, 1, 1000)
    inital_temp = 150
    npart = 363

    MD = MolecularDynamicsSimulation(Q=0)
    MD.run()
    # MD.runThermostat()
