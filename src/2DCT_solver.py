import os, time, copy
from multiprocessing import Pool
import numpy as np
import IO_util, lib_material, band_solver

class TwoDCT():
    def __init__(self, setup, jobs):
        self.__mesh__(setup, jobs)
        self.setup = setup
        self.jobs = jobs
        self.lattice = setup['Lattice']
        self.m_type = setup['Direction']
        self.H_type = setup['H_type']
    def __mesh__(self, setup, jobs):
        ## construct mesh
        # sweep parameters
        E_start = float(setup['E0'])
        E_stop = float(setup['En'])
        E_step = float(setup['dE'])
        self.E_sweep = np.arange(E_start,E_stop,E_step)
        kx_start = float(setup['kx0'])
        kx_stop = float(setup['kxn'])
        kx_step = float(setup['dkx'])
        if kx_step == 0:
            self.kx_sweep = [kx_start]
        else:
            self.kx_sweep = np.arange(kx_start,kx_stop,kx_step)
        V = float(setup['V2'])-float(setup['V1'])
        # jobs
        self.job_sweep = {}
        for job_name, job in jobs.items():
            self.job_sweep[job_name] = {'gap':[], 'length':[], 'V':[]}
            mesh = [float(m) for m in job['mesh']]
            dV = V/(sum(mesh)+1)
            for idx, mesh in enumerate(job['mesh']):
                for i in range(int(mesh)):
                    self.job_sweep[job_name]['gap'].append(float(job['gap'][idx]))
                    self.job_sweep[job_name]['length'].append(float(job['length'][idx])/int(mesh))
            V_list = [dV for i in range(len(self.job_sweep[job_name]['gap']))]
            self.job_sweep[job_name]['V'] = np.cumsum(V_list)
    def calTransmission(self, kx, job_name):
        self.current_job = job_name
        self.current_kx = kx
        band_parser = band_solver.band_structure(self.setup, kx, job_name)
        band_parser.genBand(self.E_sweep, self.job_sweep)
            
    
if __name__ == '__main__':
    '''
    load input files
    '''
    setup_file = '../input/setup_2DCT.csv'
    job_file = '../input/job_2DCT.csv'
    setup, jobs = IO_util.load_setup(setup_file, job_file)
    solver = TwoDCT(setup, jobs)
    ## calculate jobs
    for job_name, job in solver.job_sweep.items():
        '''
        calculate transmission
        '''
        for kx in solver.kx_sweep:
            solver.calTransmission(kx, job_name)
        