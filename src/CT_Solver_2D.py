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
        self.dkx = float(setup['dk_amp'])*np.cos(float(setup['dk_ang'])*np.pi/180)
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
            mesh = [int(m) for m in job['mesh']]
            for idx, m in enumerate(mesh):
                for i in range(int(m)):
                    self.job_sweep[job_name]['gap'].append(float(job['gap'][idx]))
                    self.job_sweep[job_name]['length'].append(float(job['length'][idx])/int(m))
            if setup['isLeadInclude']:
                job_mesh = copy.deepcopy(job['mesh'])
                i_lead = int(job_mesh.pop(0))
                Vi = [0 for i in range(i_lead)]
                o_lead = int(job_mesh.pop(-1))
                Vo = [0 for i in range(o_lead)]
                dV = [V/(sum([float(j) for j in job_mesh])+1) for i in range(len(self.job_sweep[job_name]['gap'])-i_lead-o_lead)]
                Vi.extend(dV)
                Vi.extend(Vo)
                V_list = Vi
            else:
                V_list = [V/(sum(mesh)+1) for i in range(len(self.job_sweep[job_name]['gap']))]
            self.job_sweep[job_name]['V'] = np.cumsum(V_list)
    def calBand(self, job_name):
        band_parser = band_solver.band_structure(self.setup, job_name)
        return band_parser.genBand(self.E_sweep, self.job_sweep)
    def calTransmission(self, job, kx, val, vec, vec_conj, job_name):
        current_parser = current_solver.current(self.setup, job_name)
        return current_parser.calTransmission(kx, job, self.E_sweep, val, vec, vec_conj)
    def calTotalCurrent(self, T_list, val, vel, job_name):
        current_parser = current_solver.current(self.setup, job_name)
        return current_parser.calTotalCurrent(self.E_sweep, self.kx_sweep, T_list, val, vel, self.job_sweep[job_name], self.job_sweep)
if __name__ == '__main__':
    '''
    load input files
    '''
    setup_file = '../input/setup_2DCT.csv'
    job_file = '../input/job_2DCT.csv'
    setup, jobs = IO_util.load_setup(setup_file, job_file)
    if not setup['isWarp']:
        setup['Material'].r3 = 0
        setup['Material'].vF3 = 0
    '''
    start solver
    '''
    solver = TwoDCT(setup, jobs)
    ## calculate jobs
    t0 = time.time()
    for job_name, job in solver.job_sweep.items():
        dir_name = job_name+'V='+str(float(setup['V2'])-float(setup['V1']))
        ## build directory
        if not os.path.exists('../output/'):
            os.mkdir('../output/')
        if not os.path.exists('../output/'+dir_name):
            os.mkdir('../output/'+dir_name)
        '''
        calculate band structure
        '''
        kx = float(setup['dk_amp'])*np.cos(float(setup['dk_ang'])*np.pi/180)
        print('Current job:',job_name, '@kx=',kx)
        t_start = time.time()
        eigVal, eigVec, eigVecConj, zone_list, vel = solver.calBand(job_name)
        print('Process: band diagram ->',time.time()-t_start, '(sec)')
        job_dir = '../output/'+dir_name+'/band/'
        if not os.path.exists(job_dir):
            os.mkdir(job_dir)
        for zone in zone_list:
            file_name = job_name+'_kx='+str(kx)+'_z'+str(zone)
            #IO_util.saveAsFigure(job_dir+file_name, eigVal[zone], solver.E_sweep, figure_type='band')
            csv_table = np.zeros((len(solver.E_sweep)+1,25), dtype=object)
            csv_table[0,:] = ['E',"KR1","KR2","KR3","KR4","KI1","KI2","KI3","KI4",
                              "K'R1","K'R2","K'R3","K'R4","K'I1","K'I2","K'I3","K'I4",
                              "K1vel", "K2vel", "K3vel", "K4vel", "K'1vel", "K'2vel", "K'3vel", "K'4vel"]
            csv_table[1:,0] = solver.E_sweep
            for i in range(4):
                val = np.array(eigVal[zone]['+K'])[:,i]
                csv_table[1:,i+1] = np.real(val)
                csv_table[1:,i+5] = np.imag(val)
                val = np.array(eigVal[zone]['-K'])[:,i]
                csv_table[1:,i+9] = np.real(val)
                csv_table[1:,i+13] = np.imag(val)
            else:
                csv_table[1:,17:21] = np.real(np.array(vel['+K']))
                csv_table[1:,21:25] = np.real(np.array(vel['-K']))
            IO_util.saveAsCSV(job_dir+file_name+'.csv',csv_table)
        '''
        calculate transmission coefficient
        '''
        t_start = time.time()
        T_list = solver.calTransmission(job, kx, eigVal, eigVec, eigVecConj, job_name)
        print('Process: transmission ->',time.time()-t_start, '(sec)')
        '''
        plot output
        '''
        job_dir = '../output/'+dir_name+'/PTR/'
        if not os.path.exists(job_dir):
            os.mkdir(job_dir)
        file_name = job_name+'_kx='+str(kx)
        #IO_util.saveAsFigure(job_dir+file_name, solver.E_sweep, T_list, figure_type='PTR')
        x = solver.E_sweep
        y = T_list
        csv_array = np.zeros((len(x),6))
        csv_array[:,0] = x
        csv_array[:,1:5] = np.real(y)
        Py = copy.deepcopy(x)
        for i in range(len(x)):
            if y[i][0] + y[i][1] != 0:
                Py[i] = np.real((y[i][0] - y[i][1])/(y[i][0] + y[i][1]))
            else:
                Py[i] = None
        csv_array[:,5] = Py
        IO_util.saveAsCSV(job_dir+file_name+'.csv', csv_array)
        '''
        calculate total current transmission
        '''
        t_start = time.time()
        JKp, JKn, P = solver.calTotalCurrent(T_list, eigVal, vel, job_name)
        file_name = job_name+'_kx='+str(kx)+'_Total'
        IO_util.saveAsCSV(job_dir+file_name+'.csv', [['K',"K'","PT"],[JKp, JKn, P]])
    print('Calculation complete. Total time ->',time.time()-t0, '(sec)')