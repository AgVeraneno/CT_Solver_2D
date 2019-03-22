import os, time, copy
from multiprocessing import Pool
import numpy as np
import IO_util, lib_material

class band_structure():
    def __init__(self, setup, kx, job_name):
        self.setup = setup
        self.current_job = job_name
        self.current_kx = kx
        self.lattice = setup['Lattice']
        self.m_type = setup['Direction']
        self.H_type = setup['H_type']
        if not os.path.exists('../output/'):
            os.mkdir('../output/')
    def genBand(self, E_sweep, job_sweep):
        self.E_sweep = E_sweep
        self.job_sweep = job_sweep
        job_name = self.current_job
        kx = self.current_kx
        with Pool(int(self.setup['CPU_threads'])) as mp:
            val_list = mp.map(self.__sweepE__, E_sweep)
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
                job_dir = '../output/'+job_name+'/'
                if not os.path.exists(job_dir):
                    os.mkdir(job_dir)
                IO_util.saveAsFigure(job_dir+file_name, eigVal, E_sweep, figure_type='band')
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
            1. place -Re, -Re, Re, Re
            2. if identical, place -Im, -Im, Im, Im
            '''
            K = self.setup['Material'].K_norm
            val_list = {'+K':[],'-K':[]}
            vec_list = {'+K':[],'-K':[]}
            for idx in range(len(self.E_sweep)):
                ## K valley
                this_val = val[idx][0:4]/K
                new_val, new_vec = self.__sort_rule__(idx, this_val, vec)
                val_list['+K'].append(new_val)
                vec_list['+K'].append(new_vec)
                ## K' valley
                this_val = val[idx][4:8]/K
                new_val, new_vec = self.__sort_rule__(idx, this_val, vec)
                val_list['-K'].append(new_val)
                vec_list['-K'].append(new_vec)
            return val_list, vec_list
    def __sort_rule__(self, idx, this_val, vec):
        new_val = copy.deepcopy(this_val)*0
        new_vec = copy.deepcopy(vec[idx])*0
        for i, ky in enumerate(this_val):
            isReZero = np.isclose(np.real(ky), 0)
            isImZero = np.isclose(np.imag(ky), 0)
            isRePos = np.real(ky) > 0
            isImPos = np.imag(ky) > 0
            if isReZero:    # decay state
                if isImPos:     # transmission decay
                    if new_val[1] == 0:
                        new_val[1] = ky
                        new_vec[:,1] = vec[idx][:,i]
                    elif np.imag(new_val[1]) < np.imag(ky):
                        new_val[3] = copy.deepcopy(new_val[1])
                        new_vec[:,3] = copy.deepcopy(new_vec[:,1])
                        new_val[1] = ky
                        new_vec[:,1] = vec[idx][:,i]
                    else:
                        new_val[3] = ky
                        new_vec[:,3] = vec[idx][:,i]
                else:       # reflection decay
                    if new_val[2] == 0:
                        new_val[2] = ky
                        new_vec[:,2] = vec[idx][:,i]
                    elif np.imag(new_val[2]) > np.imag(ky):
                        new_val[0] = copy.deepcopy(new_val[2])
                        new_vec[:,0] = copy.deepcopy(new_vec[:,2])
                        new_val[2] = ky
                        new_vec[:,2] = vec[idx][:,i]
                    else:
                        new_val[0] = ky
                        new_vec[:,0] = vec[idx][:,i]
            elif isRePos:       # transmission state or W shape reflection
                if isImZero:
                    if new_val[3] == 0:
                        new_val[3] = ky
                        new_vec[:,3] = vec[idx][:,i]
                    elif np.real(new_val[3]) < np.real(ky):
                        new_val[2] = copy.deepcopy(new_val[3])
                        new_vec[:,2] = copy.deepcopy(new_vec[:,3])
                        new_val[3] = ky
                        new_vec[:,3] = vec[idx][:,i]
                    else:
                        new_val[2] = ky
                        new_vec[:,2] = vec[idx][:,i]
                elif isImPos:     # transmission decay
                    if new_val[3] == 0:
                        new_val[3] = ky
                        new_vec[:,3] = vec[idx][:,i]
                    else:
                        raise ValueError('collision')
                else:       # reflection decay
                    if new_val[2] == 0:
                        new_val[2] = ky
                        new_vec[:,2] = vec[idx][:,i]
                    else:
                        raise ValueError('collision')
            else:       # reflection state or W shape transmission
                if isImZero:
                    if new_val[0] == 0:
                        new_val[0] = ky
                        new_vec[:,0] = vec[idx][:,i]
                    elif np.real(new_val[0]) > np.real(ky):
                        new_val[1] = copy.deepcopy(new_val[0])
                        new_vec[:,1] = copy.deepcopy(new_vec[:,0])
                        new_val[0] = ky
                        new_vec[:,0] = vec[idx][:,i]
                    else:
                        new_val[1] = ky
                        new_vec[:,1] = vec[idx][:,i]
                elif isImPos:
                    if new_val[1] == 0:
                        new_val[1] = ky
                        new_vec[:,1] = vec[idx][:,i]
                    else:
                        raise ValueError('collision')
                else:
                    if new_val[0] == 0:
                        new_val[0] = ky
                        new_vec[:,0] = vec[idx][:,i]
                    else:
                        raise ValueError('collision')
        return new_val, new_vec