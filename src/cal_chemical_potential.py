import os, time, copy
from multiprocessing import Pool
import numpy as np
import IO_util, lib_material

class CP_solver():
    def __init__(self, setup, gap, V, Ef, Temp):
        self.setup = setup
        self.mat = setup['Material']
        self.gap = gap
        self.V = V
        self.Ef = Ef*self.mat.q*1e-3
        self.Temp = self.mat.kB*Temp
        self.kx_mesh = np.arange(float(setup['kx0']),float(setup['kxn'])+float(setup['dkx']),float(setup['dkx']))
        self.ky_mesh = self.kx_mesh
        mesh_size = len(self.kx_mesh)
        '''
        calculate carrier concentration @ 0K
        '''
        with Pool(processes=int(setup['CPU_threads'])) as mp:
            ND = mp.map(self.cal_concentration, range(mesh_size))
        nD = sum(ND)/(4*np.pi**2)
        print('Carrier concentration:',nD,' 1/cm2')
        '''
        find chemical potential
        '''
        self.mu = self.Ef
        counter = 0
        while True:
            with Pool(processes=int(setup['CPU_threads'])) as mp:
                NCV = mp.map(self.find_chemical_potential, range(mesh_size))
            NC = sum(np.array(NCV)[:,0])
            NV = sum(np.array(NCV)[:,1])
            print('Try mu=',self.mu/self.mat.q*1e3,' (meV) --> difference = ',round(np.real(nD-NC+NV),6))
            if float(nD-NC+NV) <= 1 or counter >= 100:
                break
            else:
                self.mu -= 0.01*self.mat.q*1e-3
                counter += 1
    def cal_concentration(self, k_idx):
        H_parser = lib_material.Hamiltonian(self.setup)
        e_count = 0
        for ky in self.ky_mesh:
            H = H_parser.FZ_bulk(self.gap, self.V, self.kx_mesh[k_idx], ky)
            E, _ = np.linalg.eig(H)
            E = np.sort(E)[4]       # get first conduction band energy
            if E <= self.Ef:
                e_count += 1
        else:
            return e_count
    def find_chemical_potential(self, k_idx):
        H_parser = lib_material.Hamiltonian(self.setup)
        nc_count = 0
        nv_count = 0
        for ky in self.ky_mesh:
            H = H_parser.FZ_bulk(self.gap, self.V, self.kx_mesh[k_idx], ky)
            E, _ = np.linalg.eig(H)
            E = np.sort(E)[4]       # get first conduction band energy
            nc_count += 1/(1+np.exp((E-self.mu)/(self.Temp)))
            nv_count += (1-1/(1+np.exp((E-self.mu)/(self.Temp))))
        else:
            return nc_count, nv_count

if __name__ == '__main__':
    setup_file = '../input/setup_CP.csv'
    job_file = '../input/job_2DCT.csv'
    setup, jobs = IO_util.load_setup(setup_file, job_file)
    gap = 20
    V = 0
    Ef = 22
    Temp = 300
    CP_solver(setup, gap, V, Ef, Temp)