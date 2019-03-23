import os, time, copy
from multiprocessing import Pool
import numpy as np
import IO_util, lib_material, band_solver, current_solver

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
    def calBand(self, kx, job_name):
        band_parser = band_solver.band_structure(self.setup, kx, job_name)
        return band_parser.genBand(self.E_sweep, self.job_sweep)
    def calTransmission(self, job, job_name, kx, val, vec, vec_conj):
        current_parser = current_solver.current(self.setup)
        return current_parser.calTransmission(kx, job, job_name, self.E_sweep, val, vec, vec_conj)
            
        
if __name__ == '__main__':
    '''
    load input files
    '''
    setup_file = '../input/setup_2DCT.csv'
    job_file = '../input/job_2DCT.csv'
    setup, jobs = IO_util.load_setup(setup_file, job_file)
    '''
    start solver
    '''
    solver = TwoDCT(setup, jobs)
    ## calculate jobs
    t0 = time.time()
    for job_name, job in solver.job_sweep.items():
        ## build directory
        if not os.path.exists('../output/'):
            os.mkdir('../output/')
        if not os.path.exists('../output/'+job_name):
            os.mkdir('../output/'+job_name)
        ## start sweeping
        for kx in solver.kx_sweep:
            print('Current job:',job_name,'@ kx=',kx)
            '''
            calculate band structure
            '''
            t_start = time.time()
            eigVal, eigVec, eigVecConj, zone_list = solver.calBand(kx, job_name)
            print('Process: band diagram ->',time.time()-t_start, '(sec)')
            '''
            job_dir = '../output/'+job_name+'/band/'
            if not os.path.exists(job_dir):
                os.mkdir(job_dir)
            for zone in zone_list:
                file_name = job_name+'_kx='+str(kx)+'_z'+str(zone)
                #IO_util.saveAsFigure(job_dir+file_name, eigVal[zone], solver.E_sweep, figure_type='band')
                csv_table = np.zeros((len(x),16))
                csv_table[:,0] = sover.E_sweep
                for i in range(1,5):
                    csv_table[:,1] = sover.E_sweep
                IO_util.saveAsCSV(job_dir+file_name+'.csv',)
            '''
            '''
            calculate transmission
            '''
            t_start = time.time()
            T_list = solver.calTransmission(job, job_name, kx, eigVal, eigVec, eigVecConj)
            print('Process: transmission ->',time.time()-t_start, '(sec)')
            '''
            plot output
            '''

            job_dir = '../output/'+job_name+'/PTR/'
            if not os.path.exists(job_dir):
                os.mkdir(job_dir)
            file_name = job_name+'_kx='+str(kx)
            #IO_util.saveAsFigure(job_dir+file_name, solver.E_sweep, T_list, figure_type='PTR')
            x = solver.E_sweep
            y = T_list
            csv_array = np.zeros((len(x),4))
            csv_array[:,0] = x
            csv_array[:,1:3] = np.real(y)
            Py = copy.deepcopy(x)
            for i in range(len(x)):
                if y[i][0] + y[i][1] != 0:
                    Py[i] = np.real((y[i][0] - y[i][1])/(y[i][0] + y[i][1]))
                else:
                    Py[i] = None
            csv_array[:,3] = Py
            saveAsCSV(file_name+'.csv', csv_array)
            
    print('Calculation complete. Total time ->',time.time()-t0, '(sec)')