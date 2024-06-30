import time
from typing import Tuple

import numpy as np
from sympy.stats import Maxwell, sample
from scipy import constants as const
import os
from MonteCarloAss.Simulation import MonteCarloSimulation


class MolecularDynamicsSimulation:

    def __init__(self, side_length: float = 30, rho: float = 358.4, molar_mass: float = 16.04,
                 initial_temp: float = 150,
                 ndim: int = 3, rcut: float | None = None, dt: int = 1, eta_kb: float = 148, t_end: int = 3000,
                 Q: float = 1e10, input_conf: float | None = None) -> None:
        self.sigma = 3.730  # [Å]
        self.sigma6 = self.sigma ** 6
        self.sigma12 = self.sigma6 ** 2

        self.kb = const.k  # J/K
        self.NA = const.Avogadro  # 1/mol
        self.R = const.R  # J/(mol K)
        self.epsilon = eta_kb * self.kb  # [J]
        self.M = molar_mass  # g/mol
        self.ndim = ndim
        self.dt = dt  # timestep in fs
        self.side_length = side_length  # [Å]
        self.volume = self.side_length ** 3  # [Å^3]
        self.t_end = t_end

        self.npart = MolecularDynamicsSimulation.get_num_molecules(rho, self.side_length, self.M)
        self.rho_num = self.npart / (self.side_length*10**-10)**3  # Density of the system [particles/Å^3]
        if rcut:
            self.rcut = rcut
            self.rcut3 = self.rcut ** 3
            self.rcut9 = self.rcut3 ** 3
            self.tail_correction = (8 / 3) * np.pi * self.rho_num * self.epsilon * (
                    self.sigma12 / (3 * self.rcut9) - self.sigma6 / (3 * self.rcut3))  # Tail correction
        else:
            self.rcut = 10000 # arbitrary large value
            self.rcut3 = self.rcut ** 3
            self.rcut9 = self.rcut ** 9
            self.tail_correction = 0

        self.initial_temp = initial_temp  # K
        self.rho = rho  # kg/m^3
        self.mass = self.M / self.NA / 1000  # kg
        self.zeta = 0  # Thermostat variable
        self.Q = Q  # Damping parameter

        if input_conf:
            self.input_conf = input_conf
            self._particles, self.npart = self.read_xyz()
            self._velocities = self.initVel()
            self._forces = self.LJ_forces()
        else:
            self._particles = self.initGrid()
            self._velocities = self.initVel()
            self._forces = self.LJ_forces()

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


    def read_xyz(self):
        '''
            Function to read coordinates from xyz format.
        '''
        print('reading input file')
        xyz_file = open(self.input_conf, 'r')

        nparts = int(xyz_file.readline())
        xyz_file.readline()
        coord = np.zeros([nparts, 3])
        for k in range(0, nparts):
            line = xyz_file.readline()
            line = line.split()
            coord[k, 0] = line[1]
            coord[k, 1] = line[2]
            coord[k, 2] = line[3]
        print(f'{nparts} particles read from {self.input_conf}')
        return coord, nparts

    @staticmethod
    def get_num_molecules(density: float, side_length: float, molar_mass: float) -> int:
        volume_m3 = (side_length * 10**-10) ** 3  # Convert Å^3 to m^3
        mass_kg = density * volume_m3
        moles = mass_kg / (molar_mass * 10**-3)  # kg / (g/mol) -> moles
        return int(moles * const.Avogadro)


    def initGrid(self):
        # determine number of atoms per side
        n_side = int(np.ceil(self.npart ** (1 / 3)))
        # determine spacing between atom slices
        spacing = self.side_length / n_side
        # determine the range of the lattice
        lattice_range = (self.side_length - spacing) / 2
        # create a grid of atoms
        grid = [np.linspace(-lattice_range, lattice_range, n_side) for _ in range(self.ndim)]
        mesh = np.meshgrid(*grid, indexing='ij')
        # Flatten the grid to fill coords
        return np.vstack([m.flatten() for m in mesh]).T[:self.npart]

    def initVel(self) -> np.ndarray:
        print("Initializing velocities")

        # determine direction of velocities
        velocities_directions = np.random.normal(0, 1, size=(self.npart, self.ndim))
        velocities_norms = np.linalg.norm(velocities_directions, axis=1)
        velocities_directions = velocities_directions / velocities_norms[:, np.newaxis]

        # determine magnitude of velocities
        mag = np.sqrt(self.kb * self.initial_temp / self.mass)
        velocities_magnitude = np.array([sample(Maxwell('max', mag)) for _ in range(self.npart)])

        #velocity array
        velocities = velocities_directions * velocities_magnitude[:, np.newaxis]

        # Remove the linear momentum from the velocities
        velocities -= np.sum(velocities * self.mass, axis=0) / (self.npart * self.mass)

        print(f"Mean velocity: {np.mean(np.abs(velocities))} m/s")
        print(f"Velocities initialized with a temperature of {self.initial_temp} K")
        return velocities * 10**-5  # velocities in A/fs

    def pbc(self, delta: np.ndarray) -> np.ndarray:
        return (delta + self.side_length / 2) % self.side_length - self.side_length / 2

    def LJ_forces(self) -> np.ndarray:
        forces = np.zeros(self._particles.shape)
        for (i, pos_i) in enumerate(self._particles):
            delta = self.pbc(self._particles - pos_i)
            d_sq = np.sum(delta ** 2, axis=1)
            d_sq[d_sq > self.rcut ** 2] = np.inf
            d_sq[i] = np.inf
            d = np.sqrt(d_sq)
            d7 = d ** 7
            d13 = d ** 13
            lj_force_mag = 24 * self.epsilon * (self.sigma6 / d7 - 2 * self.sigma12 / d13)[:, np.newaxis]
            forces[i] = np.sum(lj_force_mag * delta/d[:, np.newaxis] * 10**-10, axis=0)
        return forces   # Force in #kg A/fs^2

    @property
    def kineticEnergy(self) -> float:
        """
        Calculate the kinetic energy of the system in J
        """
        return np.sum(0.5 * self.mass *(np.linalg.norm(self._velocities * 1e5, axis=1)) ** 2)

    @property
    def temperature(self) -> float:
        """
        :return: the temperature of the system in Kelvin
        """
        vmag = np.linalg.norm(self._velocities, axis=1) * 10**5 # m/s
        return self.mass * np.mean(vmag**2)/(self.ndim * self.kb)

    @property
    def potentialEnergyPressure(self) -> Tuple[float, float]:
        energy = 0.0
        virial = 0.0
        sigma = self.sigma * 10**-10
        sigma6 = sigma ** 6
        sigma12 = sigma6 ** 2
        for (i, pos_i) in enumerate(self._particles):
            delta = self.pbc(self._particles[i + 1:, :] - pos_i) * 10**-10
            d_sq = np.sum((delta ** 2), axis=1)
            # d_sq[d_sq > self.rcut**2] = np.inf
            d_sq[d_sq < (0.95*sigma) ** 2] = (0.95*sigma) ** 2
            d_sq = d_sq[d_sq < (self.rcut*10**-10)**2]
            d6 = d_sq ** 3
            d12 = d6 * d6
            energy += 4 * self.epsilon * np.sum(sigma12 / d12 - sigma6 / d6)
            virial += 24 * self.epsilon * np.sum(sigma6 / d6 - 2 * sigma12 / d12)
        pressure = self.rho_num * self.kb * self.temperature - virial / (6 * (self.side_length*10**-10)**3)
        return energy, pressure  # energy in J, pressure in pa

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
                file.write('%i %i %.4f %.4f %.4f %.6e %.6e %.6e %.4e %.4e %.4e\n' % (part + 1, 1, *temp[part, :]))

    def velocityVerlet(self) -> None:
        forces = self._forces

        # Update positions
        new_positions = self._particles + self._velocities * self.dt + 0.5 * (forces/self.mass) * self.dt**2

        # Update velocities
        v_t2 = self._velocities + 0.5 * forces/self.mass * self.dt

        # Compute new forces
        new_forces = self.LJ_forces()

        # compute new velocities
        new_velocities = v_t2 + 0.5 * new_forces/self.mass * self.dt

        # Update class variables
        self._particles = new_positions
        self._forces = new_forces
        self._velocities = new_velocities
        self._particles = self.pbc(self._particles)

    def velocityVerletThermostat(self) -> None:
        forces = self._forces
        energy = self.kineticEnergy
        temp = self.temperature

        # Update positions
        self._particles += self._velocities * self.dt + 0.5 * (forces / self.mass - self.zeta * self._velocities) * (
                    self.dt ** 2)

        # Intermediate thermostat variable
        zeta_t2 = self.zeta + 0.5 * self.dt / self.Q * (energy - (3 * self.npart + 1) / 2 * self.kb * temp)

        # Intermediate velocity update
        velocities_t2 = self._velocities + 0.5 * (forces / self.mass - zeta_t2 * self._velocities) * self.dt

        # Compute new forces
        new_forces = self.LJ_forces()

        # Update thermostat variable again
        self.zeta = zeta_t2 + 0.5 * self.dt / self.Q * (energy - (3*self.npart + 1)/2 * self.kb * temp)

        # Final velocity update
        self._velocities = (velocities_t2 + 0.5 * self.dt * (new_forces / self.mass)) / (1 + 0.5 * self.dt * self.zeta)

        # Update forces
        self._forces = new_forces

        self._particles = self.pbc(self._particles)

    def log(self, log_name, step):
        potentialEnergy, pressure = self.potentialEnergyPressure
        with open(log_name, 'a') as file:
            file.write(
                f"{step} {self.temperature:.6f} {pressure:.6f} {self.kineticEnergy * self.NA / 4184:.6f} {potentialEnergy * self.NA / 4184:.6f} {(self.kineticEnergy + potentialEnergy) / 4184 * self.NA:.6f}\n")

    def run(self, filename=None, logname=None):
        print("Running simulation \n")
        if os.path.exists(filename):
            os.remove(filename)
            print(f"trajectory_deleted")
        if os.path.exists(logname):
            os.remove(logname)
            print(f"log_deleted")
        for t in range(self.t_end):
            if t % 100 == 0:
                print(
                    f"\rtimestep: {t} | Temperature: {self.temperature:.3f} Kelvin | KE: {self.kineticEnergy * self.NA / 4184:.4e} kcal/mol | Mean U: [{np.mean(np.abs(self._velocities)):.3e}] Å/fs | Mean f: [{np.mean(np.abs(self._forces)) * 10 ** 10 / 4184 * self.NA:.3e}] kcal/(mol Å)",
                    flush=True, end='')
                if logname:
                    self.log(logname, t)
                if filename:
                    self.write_frame(filename, t)
            self.velocityVerlet()
        print("...")
        print("Simulation completed")

    def runThermostat(self, filename: str| None = None, logname: str | None = None):
        print("Running thermostat simulation \n")
        if os.path.exists(filename):
            os.remove(filename)
            print(f"trajectory_deleted")
        if os.path.exists(logname):
            os.remove(logname)
            print(f"log_deleted")
        for t in range(self.t_end):
            if t % 100 == 0:
                print(
                    f"\rtimestep: {t} | Temperature: {self.temperature} Kelvin | KE: {self.kineticEnergy * self.NA / 4184:.4f} kcal/mol | Mean U: [{np.mean(np.abs(self._velocities)):.3e}] Å/fs | Mean f: [{np.mean(np.abs(self._forces)) * 10 ** 10 / 4184 * self.NA:.3e}] kcal/(mol Å)",
                    flush=True, end='')
                if logname:
                    self.log(logname, t)
                if filename:
                    self.write_frame(filename, t)
            self.velocityVerletThermostat()
        print('...')
        print("Simulation completed")


if __name__ == "__main__":
    initial_temp = 150
    npart = 363
    rcut = 1000
    t_end = 10000
    dt = 1
    Q=1.4135e-22
    correct_path = 'MolecularDynamicsAss/Outputs/correct_weird5.lammps'
    correct_log = 'MolecularDynamicsAss/Outputs/correct_weird_log5.txt'
    # MD_comp_thermostat = MolecularDynamicsSimulation(Q=Q, t_end=t_end, dt=dt)
    # MD_comp_thermostat.runThermostat(correct_path, correct_log)


    traj_path = 'MolecularDynamicsAss/Outputs/final_standard_trajectory_05_20.lammps'
    log_path = 'MolecularDynamicsAss/Outputs/final_standard_log2_05_20.txt'
    # MD_comp_thermostat.run(traj_path, log_path)
    def side_length_calc(molar_mass: float, npart: int, rho: float) -> float:
        return (molar_mass / 1000 * npart / (rho * const.N_A)) ** (1 / 3)


    def PT_diagrams():
        print('PT_diagrams')
        temperatures = [100, 200, 300, 400]
        gas_side_length = side_length_calc(16.04, 362, 1.6)*10**10
        liquid_side_length = side_length_calc(16.04, 362, 358.4)*10**10
        print(gas_side_length)
        print(liquid_side_length)
        for temp in temperatures:
            MD_liquid = MolecularDynamicsSimulation(rho=358.4, Q=1.4135e-22, side_length=30, rcut = 10, initial_temp=temp, t_end=10000, dt=1)
            MD_liquid.runThermostat('MolecularDynamicsAss/Outputs/Question22/Fluid/' + str(temp) + '_trajectory.lammps',
                   'MolecularDynamicsAss/Outputs/Question22/Fluid/' + str(temp) + '_log.txt')

            MD_gas = MolecularDynamicsSimulation(rho=1.6, Q=1.4135e-22, side_length=gas_side_length, rcut=50, initial_temp=temp, t_end=10000, dt=1)
            MD_gas.runThermostat('MolecularDynamicsAss/Outputs/Question22/Gas/' + str(temp) + '_trajectory.lammps',
                     'MolecularDynamicsAss/Outputs/Question22/Gas/' + str(temp) + '_log.txt')

    PT_diagrams()

    def tune_Q2():
        print('tuning_Q2')
        start = time.time()
        tune_durations = []
        Q_values = np.logspace(-25, 0, num=25, base=10)
        # Q_values = np.linspace(1e-21, 1e20, num=25)
        for Q in Q_values:
            start_tune = time.time()
            Q_str = f"{Q:.4e}"
            # print(Q_str)
            MD = MolecularDynamicsSimulation(Q=Q, t_end=3000, dt=1)
            MD.runThermostat('MolecularDynamicsAss/Outputs/tuningQ3/' + Q_str + '_pbc_trajectory.lammps',
                   'MolecularDynamicsAss/Outputs/tuningQ3/' + Q_str + '_pbc_log.txt')
            end_tune = time.time()
            tune_durations.append(end_tune - start_tune)
            print(f"Q = {Q:.4e} completed in {end_tune - start_tune:.2f} seconds\n")

        tune_durations = np.array(tune_durations)
        print(f"Total duration: {time.time() - start:.2f} seconds")
        print(f"Average duration: {np.mean(tune_durations):.2f} seconds")
    # tune_Q2()

