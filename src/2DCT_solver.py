import os, time
import numpy as np
import IO_util

class TwoDCT():
    def __init__(self, setup):
        self.setup = setup
        self.E_sweep = setup['E']
        self.kx_sweep = setup['kx']
        self.lattice = setup['Lattice']
        
    def calComplexBandStructure(self, kx_idx):
        pass
    
if __name__ == '__main__':
    '''
    load input files
    '''
    setup_file = '../input/setup_2DCT.csv'
    job_file = '../input/job_2DCT.csv'
    setup, jobs = IO_util.load_setup(setup_file, job_file)
    ## construct mesh
    E_start = float(setup['E0'])
    E_stop = float(setup['En'])
    E_step = float(setup['dE'])
    E_sweep = np.arange(E_start,E_stop,E_step)
    kx_start = float(setup['kx0'])
    kx_stop = float(setup['kxn'])
    kx_step = float(setup['dkx'])
    if kx_step == 0:
        kx_sweep = [kx_start]
    else:
        kx_sweep = np.arange(kx_start,kx_stop,kx_step)
    print(E_sweep)