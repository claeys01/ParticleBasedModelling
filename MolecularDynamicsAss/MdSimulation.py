# import MD.py
import MonteCarloAss.Simulation as MC
import scipy.constants as const

import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime



class MolecularDynamicsSimulation:

    def __init__(self, side_length: float = 30, rho: float = 358.4, molar_mass: float = 16.04, inital_temp: float = 150,
                 ndim: int = 3, rcut: float = 10, dt: int = 1, eta_kb: float = 148, t_end: int = 3000) -> None:
        self.sigma = 3.73 * 1e-10  # [m]
        self.sigma6 = self.sigma ** 6
        self.sigma12 = self.sigma6 ** 2

        self.kb = const.physical_constants['Boltzmann constant'][0] # J/K
        # print(f"Boltzmann constant: {self.kb} kJ/K")
        self.NA = const.Avogadro  # 1/mol
        self.epsilon = eta_kb * self.kb # [J]
        self.M = molar_mass * 1e-3  # kg/mol
        self.ndim = ndim
        self.R = self.NA * self.kb  # J/(mol K)
        self.initial_temp = inital_temp # K
        self.rcut = rcut * 1e-10  # [m]
        self.rcut3 = self.rcut ** 3
        self.rcut9 = self.rcut ** 9

        self.dt = dt  # timestep in femto seconds
        self.side_length = side_length * 1e-10  # [m]
        self.volume = self.side_length ** 3  # [m^3]
        self.t_end = t_end

        self.npart = MolecularDynamicsSimulation.get_num_molecules(rho, self.side_length*1e10, self.M*1e3)
        self.rho_num = self.npart / self.volume                         # Density of the system [m^-3]
        self.tail_correction = 8 * np.pi * self.rho_num * self.epsilon * (
                self.sigma12 / (3 * self.rcut9) - self.sigma6 / (3 * self.rcut3))  # Tail correction

        # print(f"Number of particles: {self.npart}")
        # self.rho = self.density_conversion(rho)  # g/(mol Å^3)
        self.rho = rho
        # print(f"Density of the system: {self.rho} g/(mol Å^3)")
        self.mass = self.rho * 1e3 * self.side_length ** 3/self.npart  # kg
        print(f"Mass of the system: {self.mass} kg")
        # print(f"Mass of the system: {self.mass} g/mol")

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
        # show the particles in a 3d plots
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self._particles[:, 0], self._particles[:, 1], self._particles[:, 2])
        # plt.show()

        self._velocities = self.initVel()
        self._forces = self.LJ_forces()

    @staticmethod
    def get_num_molecules(density: float, side_length: float, molar_mass: float) -> int:
        return math.floor((density * (side_length * 10 ** -10) ** 3 / (molar_mass/1000)) * const.N_A)

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
        # # Initialize velocities
        # velocities = np.random.normal(0, 1, (self.npart, self.ndim))
        #
        # # Desired kinetic energy per particle
        # desired_energy = 0.5 * self.ndim * self.kb * self.initial_temp  # kJ/mol per particle
        #
        # # Current kinetic energy per particle
        # current_energy = 0.5 * self.mass * np.sum(velocities ** 2, axis=1) * 1e4  #  kJ/mol
        #
        # scaling_factor = np.sqrt(desired_energy / current_energy)
        # velocities *= scaling_factor[:, np.newaxis] # scale to Å/fs

        velocities = np.random.normal(0, np.sqrt(self.ndim*self.kb*self.initial_temp/self.mass), (self.npart, self.ndim))
        # velocities_mean = np.mean(velocities, axis=0)  # Ensure zero net momentum
        print(f"Velocities initalised with a temperature of {self.initial_temp} K")
        # print(velocities)
        return velocities

    def pbc(self, delta: np.ndarray) -> np.ndarray:
        return (delta + self.side_length/2) % self.side_length - self.side_length/2


    def LJ_forces(self) -> np.ndarray:
        forces = np.zeros((self.npart, self.ndim))
        for (i, pos_i) in enumerate(self._particles):
            delta = self.pbc(self._particles[:, :] - pos_i)
            d_sq = np.sum((delta ** 2), axis=1).reshape(self.npart,1)
            # d_sq = np.sum((delta ** 2), axis=0)
            d_sq[d_sq > self.rcut ** 2] = np.inf
            d_sq[d_sq< self.sigma**2] = self.sigma**2
            d_sq[i] = np.inf
            d6 = d_sq ** 3
            d12 = d6 * d6

            forces[i] += -24 * self.epsilon * np.sum((self.sigma6 / d6 -
                                                  2 * self.sigma12 / d12).reshape(self.npart, 1)*delta/np.sqrt(d_sq), axis=0)
        return forces


    @property
    def kineticEnergy(self) -> float:
        """
        Calculate the kinetic energy of the system in J
        """
        return 0.5 * self.mass * np.sum(self._velocities ** 2)  # J/mol

    @property
    def potentialEnergy(self) -> float:
        energy = 0.0
        for (i, pos_i) in enumerate(self._particles):
            delta = self.pbc(self._particles[i+1:, :] - pos_i)
            d_sq = np.sum((delta ** 2), axis=1)
            d_sq = d_sq[d_sq < self.rcut**2]
            d_sq[d_sq < self.sigma**2] = self.sigma**2
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
            d_sq[d_sq< self.sigma**2] = self.sigma**2
            d6 = d_sq ** 3
            d12 = d6 * d6
            virial_contribution += 24 * self.epsilon * np.sum(self.sigma6 / d6 - 2 * self.sigma12 / d12)
            kinetic_contribution += np.sum(self._velocities[i] ** 2) * self.mass
        return (virial_contribution/6 + kinetic_contribution) / self.volume

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
                file.write('%.6f %.6f\n' % (-0.5 * self.side_length*1e10, 0.5 * self.side_length*1e10))
            for dim in range(3 - self.ndim):
                file.write('%.6f %.6f\n' % (0, 0))
            file.write('ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz\n')

            temp = np.zeros((self.npart, 9))
            for dim in range(self.ndim):
                temp[:, dim] = self._particles[:, dim] * 1e10
                temp[:, dim + 3] = self._velocities[:, dim] * 1e10/1e-15
                temp[:, dim + 6] = self._forces[:, dim] * 1e10/4184/self.NA

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

        self._particles = self.pbc(self._particles)

        # Compute new forces
        new_forces = self.LJ_forces()

        # Update velocities
        self._velocities += 0.5 * (new_forces + forces) / (self.mass) * self.dt

        # Update forces
        self._forces = new_forces

    def run(self):

        # write outputs to a file in outputs directory
        filename = 'C:\\Users\\Matth\\Master\\Y1\\Q4\\particleBasedModeling\\MolecularDynamicsAss\\Outputs\\' + "_trajectory.lammps2"
        for t in range(self.t_end):
            self.velocityVerlet()
            if t % 100 == 0:
                print(
                    f"""\r{t}| 
                    Temperature: {self.temperature} Kelvin
                    Kinetic energy: {self.kineticEnergy * 1e-3 / 4184 * self.NA} kCa/mol
                    Step max velocity: {np.linalg.norm(self._velocities, axis=1).max() * 1e10/1e15} Å/fs
                    Step max force: {(self._forces).max()} N
                            """, end='', flush=True)
                self.write_frame(filename, t)

        print("Simulation completed")

if __name__ == "__main__":
    # MD = MolecularDynamicsSimulation(10, 1000, 16.04, 150, 3, 6, 0.001, 1, 1000)
    MD = MolecularDynamicsSimulation()
    MD.run()



