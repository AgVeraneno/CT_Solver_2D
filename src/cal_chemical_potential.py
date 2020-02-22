import sys, os, time, copy
from multiprocessing import Pool
import numpy as np
import IO_util, lib_material

class CP_solver():
    def __init__(self, setup, gap, V):
        self.setup = setup
        self.mat = setup['Material']
        self.gap = gap
        self.V = V
        self.Ef = float(setup['Ef'])*self.mat.q*1e-3
        self.Temp = self.mat.kB*float(setup['Temp'])
        self.kx_mesh = np.arange(float(setup['kx0']),float(setup['kxn'])+float(setup['dkx']),float(setup['dkx']))
        self.ky_mesh = self.kx_mesh
        mesh_size = len(self.kx_mesh)
        '''
        calculate carrier concentration @ 0K
        '''
        t0 = time.time()
        with Pool(processes=int(setup['CPU_threads'])) as mp:
            ND = mp.map(self.cal_concentration, range(mesh_size))
        nD = sum(ND)/(4*np.pi**2)
        t1 = time.time()
        print('Carrier concentration:',np.format_float_scientific(nD*1e4),' 1/cm2; calculated time=',round(t1-t0,4),' (sec)')
        '''
        find chemical potential
        '''
        self.mu = self.Ef
        pre_mu = 0
        up_mu = self.mu
        down_mu = pre_mu
        pre_dn = 0
        counter = 0
        while counter <= 100:
            with Pool(processes=int(setup['CPU_threads'])) as mp:
                NCV = mp.map(self.find_chemical_potential, range(mesh_size))
            NC = np.real(sum(np.array(NCV)[:,0])/(4*np.pi**2))
            NV = np.real(sum(np.array(NCV)[:,1])/(4*np.pi**2))
            print('Try mu=',round(self.mu/self.mat.q*1e3,9),' (meV) --> difference = ',round(nD-NC+NV,6),'(NC=',NC,';NV=',NV,')')
            pre_mu = copy.deepcopy(self.mu)
            if abs(nD-NC+NV) <= 1 or round(abs(nD-NC+NV)-pre_dn,6) == 0:
                break
            elif np.real(nD-NC+NV) < 0:
                self.mu = (up_mu+down_mu)/2
                up_mu = self.mu
            elif np.real(nD-NC+NV) > 0:
                self.mu = (up_mu+down_mu)/2
                down_mu = self.mu
            pre_dn = abs(nD-NC+NV)
            counter += 1
    def cal_concentration(self, k_idx):
        H_parser = lib_material.Hamiltonian(self.setup)
        e_count = 0
        for ky in self.ky_mesh:
            H = H_parser.FZ_bulk(self.gap, self.V, self.kx_mesh[k_idx], ky)
            E, _ = np.linalg.eig(H)
            E_Kp = np.sort(E[0:4])[2]   # get first conduction band energy
            E_Kn = np.sort(E[4:8])[2]   # get first conduction band energy
            if E_Kp <= self.Ef:
                e_count += 3
            else:
                e_count += 2
            if E_Kn <= self.Ef:
                e_count += 3
            else:
                e_count += 2
        else:
            return e_count
    def find_chemical_potential(self, k_idx):
        H_parser = lib_material.Hamiltonian(self.setup)
        nc_count = 0
        nv_count = 0
        for ky in self.ky_mesh:
            H = H_parser.FZ_bulk(self.gap, self.V, self.kx_mesh[k_idx], ky)
            E, _ = np.linalg.eig(H)
            for i in range(4):
                nc_count += 1/(1+np.exp((np.sort(E[0:4])[i]-self.mu)/(self.Temp)))
                nv_count += (1-1/(1+np.exp((np.sort(E[0:4])[i]-self.mu)/(self.Temp))))
                nc_count += 1/(1+np.exp((np.sort(E[4:8])[i]-self.mu)/(self.Temp)))
                nv_count += (1-1/(1+np.exp((np.sort(E[4:8])[i]-self.mu)/(self.Temp))))
        else:
            return nc_count, nv_count

if __name__ == '__main__':
    setup_file = '../input/setup_CP.csv'
    job_file = '../input/job_2DCT.csv'
    setup, jobs = IO_util.load_setup(setup_file, job_file)
    gap = 20
    V = 0
    CP_solver(setup, gap, V)
