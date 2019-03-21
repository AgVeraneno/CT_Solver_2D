import os, time, copy
from multiprocessing import Pool
from itertools import product
import numpy as np
import IO_util, lib_material

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
        with Pool(int(setup['CPU_threads'])) as mp:
            val_list = mp.map(self.__sweepE__, self.E_sweep)
        ## plot band structure
        for zone in val_list[0]['zone']:
            file_name = job_name+'_kx='+str(kx)+'_z'+str(zone)
            eigVal = []
            eigVec = []
            for idx in range(len(self.E_sweep)):
                eigVal.append(val_list[idx]['val'][zone])
                eigVec.append(val_list[idx]['vec'][0])
            else:
                eigVal, eigVec = self.__sort__(eigVal, eigVec)
                IO_util.saveAsFigure(file_name, eigVal, self.E_sweep, band=True)
        IO_util.saveAsCSV('band.csv', np.block(val_list))
    def __sweepE__(self, E):
        job_name = self.current_job
        kx = self.current_kx
        val_list = {'E':E,'zone':[],'val':[], 'vec':[]}
        ## calculate complex band
        H_parser = lib_material.Hamiltonian(self.setup)
        for idx, gap in enumerate(self.job_sweep[job_name]['gap']):
            val_list['zone'].append(idx)
            length = self.job_sweep[job_name]['length'][idx]
            V = self.job_sweep[job_name]['V'][idx]
            if self.H_type == 'linearized':
                if self.m_type == 'Zigzag':
                    Hi, Hp = H_parser.linearized(gap, E, V, kx)
                    val, vec = np.linalg.eig(-np.dot(np.linalg.inv(Hp), Hi))
                    val_list['val'].append(val)
                    if idx == 0:        # save incident states
                        val_list['vec'].append(vec)
                elif self.m_type == 'Armchair':
                    Hi, Hp = H_parser.linearized(gap, E, V, kx)
                    val, vec = np.linalg.eig(-np.dot(np.linalg.inv(Hp), Hi))
                    val_list['val'].append(val)
                    if idx == 0:        # save incident states
                        val_list['vec'].append(vec)
        return val_list
    def __sort__(self, val, vec):
        if self.H_type == 'linearized':
            '''
            sort first energy eigenstate
            the pattern will be: R1, T1, R2, T2
            '''
            ## K valley
            for idx in range(len(self.E_sweep)):
                val_sorted = np.sort(val[idx][0:4])
                
                new_val = copy.deepcopy(val)
                new_vec = copy.deepcopy(vec)
                for v_idx, v in enumerate(val):
                    for v_idx2, v2 in enumerate(val_sorted):
                        if v == v2:
                            new_val[v_idx] = val[v_idx2]/self.setup['Material'].K_norm
                            new_vec[v_idx,:] = vec[v_idx2,:]
                            continue
                else:
                    return new_val, new_vec
            
    
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
        