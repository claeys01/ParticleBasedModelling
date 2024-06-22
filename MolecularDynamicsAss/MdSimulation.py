import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as const
import math


class MolecularDynamicsSimulation:

    def __init__(self, side_length: float = 30, rho: float = 358.4, molar_mass: float = 16.04,
                 initial_temp: float = 150,
                 ndim: int = 3, rcut: float = 10, dt: int = 1, eta_kb: float = 148, t_end: int = 3000,
                 Q: float = 1e10) -> None:
        self.sigma = 3.73  # [Å]
        self.sigma6 = self.sigma ** 6
        self.sigma12 = self.sigma6 ** 2

        self.kb = const.k  # J/K
        self.NA = const.Avogadro  # 1/mol
        self.R = const.R  # J/(mol K)
        self.epsilon = eta_kb * self.kb  # [J]
        self.M = molar_mass  # g/mol
        self.ndim = ndim
        self.rcut = rcut  # [Å]
        self.rcut3 = self.rcut ** 3
        self.rcut9 = self.rcut ** 9

        self.dt = dt  # timestep in fs
        self.side_length = side_length  # [Å]
        self.volume = self.side_length ** 3  # [Å^3]
        self.t_end = t_end

        self.npart = MolecularDynamicsSimulation.get_num_molecules(rho, self.side_length, self.M)
        self.rho_num = self.npart / self.volume  # Density of the system [particles/Å^3]
        self.tail_correction = (8 / 3) * np.pi * self.rho_num * self.epsilon * (
                self.sigma12 / (3 * self.rcut9) - self.sigma6 / (3 * self.rcut3))  # Tail correction
        self.initial_temp = initial_temp  # K

        self.rho = rho  # kg/m^3
        # self.mass = self.rho * (self.side_length * 1e-10) ** 3 / self.npart  # kg per particle
        self.mass = self.M / self.NA / 1000  # kg
        self.zeta = 0  # Thermostat variable
        self.Q = Q  # Damping parameter

        print(f"""Simulation parameters:
    Boltzmann constant: {self.kb} J/K
    Avogadro constant: {self.NA} 1/mol
    Side length: {self.side_length} Å
    Density: {self.rho} kg/m^3
    Mass: {self.mass} kg
    Initial temperature: {self.initial_temp} K
    Number of dimensions: {self.ndim}
    Number of particles: {self.npart}
    Cutoff distance: {self.rcut} Å
    Timestep: {dt} fs
    Epsilon: {self.epsilon} J
    End time: {t_end} fs
        """)

        self._particles = self.initGrid()
        self._velocities = self.initVel()
        self._forces = self.LJ_forces()

    @staticmethod
    def get_num_molecules(density: float, side_length: float, molar_mass: float) -> int:
        volume_m3 = (side_length * 1e-10) ** 3  # Convert Å^3 to m^3
        mass_kg = density * volume_m3
        moles = mass_kg / (molar_mass * 1e-3)  # kg / (g/mol) -> moles
        return int(moles * const.Avogadro)

    def initGrid(self) -> np.ndarray:
        print("Initializing grid")

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
        print(f"{self.ndim}D grid initialized with {self.npart} particles")
        return positions - self.side_length / 2

    def initVel(self) -> np.ndarray:
        print("Initializing velocities")

        # determine direction of velocities
        velocities_directions = np.random.normal(size=(self.npart, self.ndim))
        velocities_norms = np.linalg.norm(velocities_directions, axis=1)
        velocities_directions = velocities_directions / velocities_norms[:, np.newaxis]

        # determine magnitude of velocities
        velocities_magnitude = np.sqrt(self.ndim * self.R * self.initial_temp / (self.M / 1000))  # Convert to Å/fs

        velocities = velocities_directions * velocities_magnitude

        # Calculate the linear momentum in each direction
        linear_momentum = np.sum(velocities * self.mass, axis=0)  # kg m/s

        # Remove the linear momentum from the velocities
        velocities -= linear_momentum / (self.npart * self.mass)

        print(f"Mean velocity: {np.mean(np.abs(velocities))} m/s")
        print(f"Velocities initialized with a temperature of {self.initial_temp} K")
        return velocities * 1e-5  # Convert to Å/fs

    def pbc(self, delta: np.ndarray) -> np.ndarray:
        return (delta + self.side_length / 2) % self.side_length - self.side_length / 2

    def LJ_forces(self) -> np.ndarray:
        forces = np.zeros((self.npart, self.ndim))
        for (i, pos_i) in enumerate(self._particles):
            delta = self.pbc(self._particles[:, :] - pos_i)
            d_sq = np.sum((delta ** 2), axis=1).reshape(self.npart, 1)
            d_sq[d_sq > self.rcut ** 2] = np.inf
            d_sq[d_sq <= self.sigma ** 2] = self.sigma ** 2
            d_sq[i] = np.inf
            d7 = d_sq ** 3.5
            d13 = d_sq ** 6.5
            forces[i] += np.sum((self.sigma6 / d7 - 2 * self.sigma12 / d13)
                                .reshape(self.npart, 1) * delta / np.sqrt(d_sq), axis=0)
        return -24 * self.epsilon * forces * 1e-10  # Convert forces to J/Å

    @property
    def kineticEnergy(self) -> float:
        """
        Calculate the kinetic energy of the system in J
        """
        return 10 ** 4 * 0.5 * self.M * np.sum(np.linalg.norm(self._velocities, axis=1) ** 2)
        # return 10**5* 0.5 * self.M * np.sum((np.linalg.norm(self._velocities,axis=1))** 2)

    @property
    def temperature(self) -> float:
        """
        :return: the temperature of the system in Kelvin
        """
        vmag = np.linalg.norm(self._velocities, axis=1) * 10 ** 5  # m/s
        return self.mass * np.mean(vmag ** 2) / (self.ndim * self.kb)  # kg m^2/s^2 / J/K = K
        # return 2.0 * self.kineticEnergy / (self.ndim * self.kb * self.NA * self.npart)  # K

    @property
    def potentialEnergyPressure(self) -> float:
        energy = 0.0
        virial = 0.0
        for (i, pos_i) in enumerate(self._particles):
            delta = self.pbc(self._particles[i + 1:, :] - pos_i)
            d_sq = np.sum((delta ** 2), axis=1)
            d_sq = d_sq[d_sq < self.rcut ** 2]
            # d_sq[d_sq > self.rcut**2] = np.inf
            # d1[d_sq > self.rcut**2]
            d_sq[d_sq < self.sigma ** 2] = self.sigma ** 2
            d6 = d_sq ** 3
            d12 = d6 * d6
            energy += 4 * self.epsilon * np.sum(self.sigma12 / d12 - self.sigma6 / d6) + self.tail_correction
            virial += 24 * self.epsilon * np.sum(self.sigma6 / d6 - 2 * self.sigma12 / d12)
        pressure = self.rho_num * self.kb * self.temperature - virial / (6 * self.volume)
        return energy, pressure

    def write_frame(self, trajectory_name, step):
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
                temp[:, dim + 6] = self._forces[:, dim] * 1e10 / 4184 * self.NA  # Convert forces to kcal/mol/Å

            for part in range(self.npart):
                file.write('%i %i %.4e %.4e %.4e %.6e %.6e %.6e %.4e %.4e %.4e\n' % (part + 1, 1, *temp[part, :]))

    def velocityVerlet(self) -> None:
        forces = self._forces

        # Update positions
        self._particles += self._velocities * self.dt + 0.5 * (forces / self.mass) * (self.dt ** 2)

        # Compute new forces
        new_forces = self.LJ_forces()

        # Update velocities
        self._velocities += 0.5 * (new_forces + forces) / self.mass * self.dt

        # Update forces
        self._forces = new_forces

        self._particles = self.pbc(self._particles)

    def velocityVerletThermostat(self) -> None:
        forces = self._forces
        energy = self.kineticEnergy
        temp = self.temperature

        # Update positions
        self._particles += self._velocities * self.dt + 0.5 * (forces / self.mass - self.zeta * self._velocities) * (
                    self.dt ** 2)

        # Intermediate velocity update
        velocities_t2 = self._velocities + 0.5 * (forces / self.mass - self.zeta * self._velocities) * self.dt

        # Intermediate thermostat variable
        zeta_t2 = self.zeta + 0.5 * self.dt / self.Q * (energy - 1.5 * (self.npart + 1) * self.kb * temp)

        # Compute new forces
        new_forces = self.LJ_forces()

        # Update thermostat variable again
        self.zeta = zeta_t2 + 0.5 * self.dt / self.Q * (energy - 1.5 * (self.npart + 1) * self.kb * temp)

        # Final velocity update
        self._velocities = (velocities_t2 + 0.5 * self.dt * (new_forces / self.mass)) / (1 + 0.5 * self.dt * self.zeta)

        # Update forces
        self._forces = new_forces

        self._particles = self.pbc(self._particles)

    def log(self, log_name, step):
        potentialEnergy, pressure = self.potentialEnergyPressure
        with open(log_name, 'a') as file:
            file.write(
                f"{step} {self.temperature:.6f} {pressure:.6f} {self.kineticEnergy / 4184:.6f} {potentialEnergy / 4184:.6f} {(self.kineticEnergy + potentialEnergy) / 4184:.6f}\n")

    def run(self, filename, logname):
        print("Running simulation \n")
        for t in range(self.t_end):
            self.velocityVerlet()
            if t % 100 == 0:
                print(
                    f"\rtimestep: {t} | Temperature: {self.temperature:.3f} Kelvin | KE: {self.kineticEnergy / 4.184:.4f} kcal/mol | Mean U: [{np.mean(np.abs(self._velocities)):.3e}] Å/fs | Mean f: [{np.mean(np.abs(self._forces)) * 10 ** 10 / 4184 * self.NA:.3e}] kcal/(mol Å)",
                    flush=True, end='')
                self.log(logname, t)
                self.write_frame(filename, t)
        print("...")
        print("Simulation completed")

    def runThermostat(self, filename: str, logname):
        print("Running thermostat simulation \n")
        for t in range(self.t_end):
            self.velocityVerletThermostat()
            if t % 100 == 0:
                print(
                    f"\rtimestep: {t} | Temperature: {self.temperature:.3f} Kelvin | KE: {self.kineticEnergy / 4.184:.4f} kcal/mol | Mean U: [{np.mean(np.abs(self._velocities)):.3e}] Å/fs | Mean f: [{np.mean(np.abs(self._forces)) * 10 ** 10 / 4184 * self.NA:.3e}] kcal/(mol Å)",
                    flush=True, end='')
                self.write_frame(filename, t)
                self.log(logname, t)
        print('...')
        print("Simulation completed")


if __name__ == "__main__":
    initial_temp = 150
    npart = 363
    rcut = 1000
    t_end = 3000
    # print(initial_temp*const.Avogadro*npart)

    # MD_comp = MolecularDynamicsSimulation(Q=initial_temp * const.k * npart, t_end=t_end, rcut=rcut)
    # MD_comp.run('MolecularDynamicsAss/Outputs/pbc_standard_trajectory.lammps',
    #             'MolecularDynamicsAss/Outputs/pbc_standard_log.txt')
    #
    # MD_comp_thermostat = MolecularDynamicsSimulation(Q=initial_temp * const.k * npart, t_end=t_end, rcut=rcut)
    # MD_comp_thermostat.runThermostat('MolecularDynamicsAss/Outputs/pbc_standard_thermostat_trajectory.lammps',
    #                                  'MolecularDynamicsAss/Outputs/pbc_standard_thermostat_log.txt')


    def tune_Q():
        Q_values = np.logspace(1, 25, num=25, base=10)
        for Q in Q_values:
            Q_str = f"{Q:.0e}"
            print(Q_str)
            MD = MolecularDynamicsSimulation(Q=Q, t_end=t_end, rcut=rcut)
            MD.runThermostat('MolecularDynamicsAss/Outputs/' + Q_str + '_thermostat_trajectory.lammps',
                             'MolecularDynamicsAss/Outputs/' + Q_str + 'thermostat_log.txt')
            print(f"Q = {Q:.0e} completed")

    def tune_Q2():
        Q_values = np.logspace(-15, -25, num=25, base=10)
        for Q in Q_values:
            Q_str = f"{Q:.4e}"
            # print(Q_str)
            MD = MolecularDynamicsSimulation(Q=Q, t_end=t_end, rcut=rcut)
            MD.run('MolecularDynamicsAss/Outputs/tuningQ2/' + Q_str + '_pbc_trajectory.lammps',
                   'MolecularDynamicsAss/Outputs/tuningQ2/' + Q_str + '_pbc_log.txt')
            print(f"Q = {Q:.4e} completed")
    tune_Q2()

