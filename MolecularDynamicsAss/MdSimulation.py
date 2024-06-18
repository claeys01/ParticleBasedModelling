# import MD.py
import MonteCarloAss.Simulation as MC
import scipy.constants as const

import numpy as np
import matplotlib.pyplot as plt
import math


class MolecularDynamicsSimulation:

    def __init__(self, side_length: float = 30, rho: float = 358.4, molar_mass: float = 16.04, inital_temp: float = 150,
                 ndim: int = 3, rcut: float = 10, dt: int = 1, eta_kb: float = 148, t_end: int = 3000) -> None:
        self.sigma = 3.73   # [Å]
        self.sigma6 = self.sigma ** 6
        self.sigma12 = self.sigma6 ** 2

        self.kb = const.physical_constants['Boltzmann constant'][0]  # /mol
        self.NA = const.Avogadro
        self.epsilon = eta_kb * self.kb
        self.M = molar_mass  # g/mol
        self.ndim = ndim
        self.R = 8.314  # J/(mol K)
        self.initial_temp = inital_temp
        self.rcut = rcut  # [Å]
        self.dt = dt  # timestep in femto seconds
        self.side_length = side_length  # [Å]
        self.volume = self.side_length **3  # [Å^3]
        self.t_end = t_end

        self.rho = MolecularDynamicsSimulation.density_conversion(rho, molar_mass)  # g/(mol Å^3)
        self.mass = self.rho * self.side_length ** 3  # g/mol
        print(f"Mass of the system: {self.mass} g/mol")
        self.npart = MolecularDynamicsSimulation.get_num_molecules(rho, self.side_length, self.M)
        print(f"Number of particles: {self.npart}")
        self._particles = self.initGrid()
        # show the particles in a 3d plots
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self._particles[:, 0], self._particles[:, 1], self._particles[:, 2])
        # plt.show()

        self._velocities = self.initVel()
        self._forces = np.zeros((self.npart, self.ndim))
        self._forces = self.LJ_forces()

    @staticmethod
    def get_num_molecules(density: float, side_length: float, molar_mass: float) -> int:
        return math.floor((density * (side_length * 10 ** -10) ** 3 / (molar_mass/1000)) * const.N_A)

    @staticmethod
    def density_conversion(rho: float, M: float) -> float:
        """
        Convert density to particles per Å^3
        rho in kg/m^3
        M in g/mol

        Returns: density in g/(mol Å^3)
        """
        return (rho / M)

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
        """
        nog nadenken over additional considerations


        :return: velocities of the system in m/s
        """

        print("Initalising velocities")
        # Initialize velocities
        velocities = np.random.normal(0, 1, (self.npart, self.ndim))

        energy_per_particle = 0.5 * self.M * np.sum(velocities ** 2, axis=1) / self.npart

        desired_energy = 0.5 * self.ndim * self.kb * self.initial_temp

        scaling_factor = np.sqrt(desired_energy / energy_per_particle)

        velocities *= scaling_factor[:, np.newaxis]
        print(f"Velocities initalised with a temperature of {self.initial_temp} K")
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
            d_sq[i] = np.inf
            d6 = d_sq ** 3
            d12 = d6 * d6

            forces[i] += -24 * self.epsilon * np.sum((self.sigma6 / d6 -
                                                  2 * self.sigma12 / d12).reshape(self.npart, 1)*delta/np.sqrt(d_sq), axis=0)
        return forces


    def velocityVerlet(self) -> None:
        """
        Performs 1 step of the verlet algorithm
        """
        print("Performing velocity verlet")
        # retrieve the LJ forces for the current positions
        # forces = self.LJ_forces

        # update the positions
        self._particles += self._velocities * self.dt + 0.5*(self._forces/self.mass)*self.dt**2
        # update forces
        new_forces = self.LJ_forces()
        # print(self._particles)

        # update velocities
        self._velocities += 0.5 * (self._forces+new_forces) / self.mass * self.dt

        self._forces = new_forces
        print(self._forces)

    @property
    def kineticEenrgy(self) -> float:
        return 0.5 * self.mass * np.sum(self._velocities ** 2)

    @property
    def potentialEnergy(self) -> float:
        energy = 0.0
        for (i, pos_i) in enumerate(self._particles):
            delta = self.pbc(self._particles[i+1:, :] - pos_i)
            d_sq = np.sum((delta ** 2), axis=1)
            d_sq = d_sq[d_sq < self.rcut**2]
            d6 = d_sq ** 3
            d12 = d6 * d6
            energy += 4 * self.epsilon * np.sum(self.sigma12 / d12 - self.sigma6 / d6)
        return energy

    @property
    def temperature(self) -> float:
        """
        mogelijks nog delen door npart

        :return:
        """

        return 2.0 * self.kineticEenrgy / (self.ndim * self.kb * self.npart)

    # @property
    # def pressure1(self) -> float:
    #     virial_contribution = 0.0
    #     kinetic_contribution = 0.0
    #     for i in range(self.npart):
    #         for j in range(i + 1, self.npart):
    #             delta = self.pbc(self._particles[j] - self._particles[i])
    #             d_sq = np.sum(delta ** 2)
    #             if d_sq < self.rcut ** 2:
    #                 force_vect = self.lj_force_magnitude(d_sq) * delta/np.linalg.norm(delta)
    #                 virial_contribution += np.dot(delta, force_vect)
    #
    #         kinetic_contribution += np.sum(self._velocities[i] ** 2)
    #     return (virial_contribution + kinetic_contribution) / self.volume

    @property
    def pressure(self) -> float:
        virial_contribution = 0.0
        kinetic_contribution = 0.0
        for i, pos_i in enumerate(self._particles):
            delta = self.pbc(self._particles[i + 1:, :] - pos_i)
            d_sq = np.sum((delta ** 2), axis=1)
            d_sq[d_sq > self.rcut ** 2] = np.inf
            d6 = d_sq ** 3
            d12 = d6 * d6
            virial_contribution += 24 * self.epsilon * np.sum(self.sigma6 / d6 - 2 * self.sigma12 / d12)
            kinetic_contribution += np.sum(self._velocities[i] ** 2) * self.M
        return (virial_contribution + kinetic_contribution) / self.volume

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
                file.write('%.6f %.6f\n' % (-0.5 * self.side_length, 0.5 * self.side_length))
            for dim in range(3 - self.ndim):
                file.write('%.6f %.6f\n' % (0, 0))
            file.write('ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz\n')

            temp = np.zeros((self.npart, 9))
            for dim in range(self.ndim):
                temp[:, dim] = self._particles[:, dim]
                temp[:, dim + 3] = self._velocities[:, dim]
                temp[:, dim + 6] = self._forces[:, dim]

            for part in range(self.npart):
                file.write('%i %i %.4f %.4f %.4f %.6f %.6f %.6f %.4f %.4f %.4f\n' % (part + 1, 1, *temp[part, :]))

    def run(self):
        filename = "first_trajectory2.lammps"
        for t in range(self.t_end):
            self.velocityVerlet()
            if t % 100 == 0:
                # print(self._forces)
                self.write_frame(filename, t)
                print(f"Step {t} of {self.t_end} completed")

        print("Simulation completed")

if __name__ == "__main__":
    # MD = MolecularDynamicsSimulation(10, 1000, 16.04, 150, 3, 6, 0.001, 1, 1000)
    MD = MolecularDynamicsSimulation()
    MD.run()



