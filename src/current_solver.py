import os, time, copy
from multiprocessing import Pool
import numpy as np
import IO_util, lib_material, band_solver

class current():
    def __init__(self, setup, job_name):
        self.setup = setup
        self.mat = setup['Material']
        self.lattice = setup['Lattice']
        self.m_type = setup['Direction']
        self.H_type = setup['H_type']
        if not os.path.exists('../output/'):
            os.mkdir('../output/')
        if self.H_type == 'linearize':
            self.I0 = 1/self.mat.h_bar
        else:
            self.I0 = 3j*self.mat.q*self.mat.acc/(2*self.mat.h_bar)
        self.H_parser = lib_material.Hamiltonian(setup)
        self.band_parser = band_solver.band_structure(setup, job_name)
    def calTransmission(self, kx, job, E_sweep, val, vec, vec_conj, isSingleThread=False):
        self.currentJob = job
        self.currentkx = kx
        self.val = val
        self.vec = vec
        self.vec_conj = vec_conj
        if isSingleThread:
            for E_idx in range(len(E_sweep)):
                T_list = self.__sweepE__(E_idx)
        else:
            with Pool(int(self.setup['CPU_threads'])) as mp:
                T_list = mp.map(self.__sweepE__, range(len(E_sweep)))
        return T_list
    def calTotalCurrent(self, E_sweep, kx_list, T_list, val, vel, job, job_sweep):
        self.job_sweep = job_sweep
        self.job = job
        self.gap = job['gap'][0]
        self.V = job['V'][0]
        self.kx_sweep = kx_list
        self.E_sweep = E_sweep
        with Pool(int(self.setup['CPU_threads'])) as mp:
            J_list = mp.map(self.__sweepE_current__, range(len(E_sweep)))
        JKp = np.array(J_list)[:,0]
        JKn = np.array(J_list)[:,1]
        P = (sum(JKp)-sum(JKn))/(sum(JKp)+sum(JKn))
        return sum(JKp), sum(JKn), P
    def __sweepE_current__(self, E_idx):
        # get incident states
        Jtp = 0
        Jtn = 0
        for kx in self.kx_sweep:
            E = self.E_sweep[E_idx]
            ## calculate band
            val, vec, vec_conj, zone_list, vel = self.band_parser.genBand([E],self.job_sweep, isSingleThread=True)
            i_state, isKpW, isKnW = self.getIncidentState(val)
            ## calculate transmission
            T = self.calTransmission(kx, self.job, [E], val, vec, vec_conj, isSingleThread=True)
            Kp_vel = abs(vel['+K'][0][3])
            Kn_vel = abs(vel['-K'][0][3])
            KpT = T[0]
            KnT = T[1]
            if isKpW:
                f = self.getFermiDist(self.gap, E, self.V, kx, val[0]['+K'][0][3], '+K')
                Jtp += abs(f*Kp_vel*KpT)
            else:
                f = self.getFermiDist(self.gap, E, self.V, kx, val[0]['+K'][0][3], '+K')
                Jtp += abs(f*Kp_vel*KpT)
            if isKnW:
                f = self.getFermiDist(self.gap, E, self.V, kx, val[0]['-K'][0][3], '-K')
                Jtn += abs(f*Kn_vel*KnT)
            else:
                f = self.getFermiDist(self.gap, E, self.V, kx, val[0]['-K'][0][3], '-K')
                Jtn += abs(f*Kn_vel*KnT)
        else:
            return Jtp, Jtn
        
    def __sweepE__(self, E_idx):
        '''
        build transfer matrix
        '''
        kx = self.currentkx
        ## calculate local current
        # incident zone
        val = np.block([self.val[0]['+K'][E_idx], self.val[0]['-K'][E_idx]])
        vec = np.block([self.vec[0]['+K'][E_idx][:,:], self.vec[0]['-K'][E_idx][:,:]])
        vec_conj = np.block([self.vec_conj[0]['+K'][E_idx][:,:], self.vec_conj[0]['-K'][E_idx][:,:]])
        J0 = self.genLocalCurrent(kx,val,vec,1000)
        # get incident states
        i_state, isKpW, isKnW = self.getIncidentState(val)
        # output zone
        val = np.block([self.val[-1]['+K'][E_idx], self.val[-1]['-K'][E_idx]])
        vec = np.block([self.vec[-1]['+K'][E_idx][:,:], self.vec[-1]['-K'][E_idx][:,:]])
        vec_conj = np.block([self.vec_conj[-1]['+K'][E_idx][:,:], self.vec_conj[-1]['-K'][E_idx][:,:]])
        JN = self.genLocalCurrent(kx,val,vec,1000)
        ## calculate interface current
        z_len_list = np.cumsum(self.currentJob['length'])
        z_len_list = z_len_list[:-1]
        for z_idx, z_len in enumerate(z_len_list):
            val = np.block([self.val[z_idx]['+K'][E_idx], self.val[z_idx]['-K'][E_idx]])
            vec = np.block([self.vec[z_idx]['+K'][E_idx][:,:], self.vec[z_idx]['-K'][E_idx][:,:]])
            vec_conj = np.block([self.vec_conj[z_idx]['+K'][E_idx][:,:], self.vec_conj[z_idx]['-K'][E_idx][:,:]])
            z1 = {'val':copy.deepcopy(val),
                  'vec':copy.deepcopy(vec),
                  'vec_conj':copy.deepcopy(vec_conj)}
            val = np.block([self.val[z_idx+1]['+K'][E_idx], self.val[z_idx+1]['-K'][E_idx]])
            vec = np.block([self.vec[z_idx+1]['+K'][E_idx][:,:], self.vec[z_idx+1]['-K'][E_idx][:,:]])
            vec_conj = np.block([self.vec_conj[z_idx+1]['+K'][E_idx][:,:], self.vec_conj[z_idx+1]['-K'][E_idx][:,:]])
            z2 = {'val':copy.deepcopy(val),
                  'vec':copy.deepcopy(vec),
                  'vec_conj':copy.deepcopy(vec_conj)}
            Ji, Jo = self.genInterCurrent(kx, z1, z2, z_len)
            if z_idx == 0:
                Jinc = copy.deepcopy(Ji)
                Tmat = Jo
            else:
                Tmat = np.dot(Tmat, Jo)
        '''
        calculate transmission and reflection
        '''
        if self.m_type == 'Zigzag':
            # K valley transmission
            if sum(i_state[0:4]) == 0:
                TKp = RKp = 0
            else:
                TKp, RKp = self.calCurrent(i_state[0:4], Tmat[0:4,0:4], Jinc[0:4,0:4], JN[0:4], J0[0:4])
            # K' valley transmission
            if sum(i_state[4:8]) == 0:
                TKn = RKn = 0
            else:
                TKn, RKn = self.calCurrent(i_state[4:8], Tmat[4:8,4:8], Jinc[4:8,4:8], JN[4:8], J0[4:8])
        return TKp, TKn, RKp, RKn
    def genLocalCurrent(self, kx, val, vec, z_len):
        if self.m_type == 'Zigzag':
            m_size = len(val)
            J_local = np.zeros(m_size, dtype=np.complex128)
            for i in range(m_size):
                ky = val[i]
                psi = vec[:,i]
                J = self.H_parser.J_op(kx, ky, ky, isLocal=True)
                J_prob = np.vdot(psi, np.dot(J, psi))
                if np.imag(ky-np.conj(ky)) >= 0:
                    phase = np.exp(1j*(ky-np.conj(ky))*self.mat.K_norm*z_len*1e-9)
                else:
                    phase = np.exp(-1j*(ky-np.conj(ky))*self.mat.K_norm*z_len*1e-9)
                J_local[i] = self.I0*phase*J_prob
            else:
                return J_local
    def genInterCurrent(self, kx, z1, z2, z_len):
        if self.m_type == 'Zigzag':
            m_size = len(z1['val'])
            Ji = np.zeros((m_size,m_size), dtype=np.complex128)
            Jo = np.zeros((m_size,m_size), dtype=np.complex128)
            for i in range(m_size):
                ky1 = z1['val'][i]
                psi1 = z1['vec_conj'][:,i]
                for j in range(m_size):
                    ## incident state
                    ky2 = z1['val'][j]
                    psi2 = z1['vec'][:,j]
                    J = self.H_parser.J_op(kx, ky1, ky2, isLocal=False)
                    J_prob = np.dot(psi1, np.dot(J, psi2))
                    phase = np.exp(1j*(ky2-ky1)*self.mat.K_norm*z_len*1e-9)
                    Ji[i,j] = self.I0*phase*J_prob
                    ## output state
                    ky2 = z2['val'][j]
                    psi2 = z2['vec'][:,j]
                    J = self.H_parser.J_op(kx, ky1, ky2, isLocal=False)
                    J_prob = np.dot(psi1, np.dot(J, psi2))
                    phase = np.exp(1j*(ky2-ky1)*self.mat.K_norm*z_len*1e-9)
                    Jo[i,j] = self.I0*phase*J_prob
            else:
                Jo = np.dot(np.linalg.inv(Ji), Jo)
                return Ji, Jo
    def getIncidentState(self, val):
        val_im = np.imag(val)
        i_state = np.zeros(len(val))
        for i, v in enumerate(val_im):
            if np.isclose(v, 0) and i%2 == 1:   # record transmission states
                i_state[i] = 1
        else:
            if sum(i_state[0:4]) == 2:
                WKp = True
            else:
                WKp = False
            if sum(i_state[4:8]) == 2:
                WKn = True
            else:
                WKn = False
            return i_state, WKp, WKn
    def getFermiDist(self, gap, E, V, kx, ky, valley):
        T = float(self.setup['Temp'])
        Ef = float(self.setup['Ef'])
        dkx = float(self.setup['dk_amp'])*np.cos(float(self.setup['dk_ang'])*np.pi/180)
        dky = float(self.setup['dk_amp'])*np.sin(float(self.setup['dk_ang'])*np.pi/180)
        if valley == '+K':
            E_eig, _ = np.linalg.eig(self.H_parser.FZ_bulk(gap, V, kx-dkx, ky-dky))
            thisE = sorted(E_eig[0:4])[2]
        elif valley == '-K':
            E_eig, _ = np.linalg.eig(self.H_parser.FZ_bulk(gap, V, kx-dkx, ky-dky))
            thisE = sorted(E_eig[4:8])[2]
        if T > 10:
            f = 1/(1+np.exp((thisE-(Ef+V)*1e-3*self.mat.q)/(self.mat.kB*T)))
        else:
            if thisE <= (Ef+V)*1e-3*self.mat.q:
                f = 1
            else:
                f = 0
        return f
    def calCurrent(self, i_state, T, Jinc, JT, JR):
        t_state = [0,1]*int(len(i_state)/2)
        r_state = [1,0]*int(len(i_state)/2)
        '''
        calculate interface coefficient
        '''
        for i in range(len(t_state)):
            for j in range(len(t_state)):
                if i == j and t_state[j] == 0:
                    T[i][j] = -1
                else:
                    T[i][j] = T[i][j]*t_state[j]
        if sum(i_state) == 2:
            Jt = 0
            Jr = 0
            Ji = 0
            for i in [3]:
                i_state = [0,0,0,0]
                i_state[i] = 1
                c_vec = np.linalg.solve(T, i_state)
                c_abs = [np.conj(c)*c for c in c_vec]
                '''
                calculate current
                '''
                Ji += np.abs(np.real(np.dot(i_state, np.dot(Jinc, i_state))))
                Jt += sum([t*c_abs[i]*np.abs(np.real(JT[i])) for i, t in enumerate(t_state)])
                Jr += sum([r*c_abs[i]*np.abs(np.real(JR[i])) for i, r in enumerate(r_state)])
            else:
                return Jt/Ji, Jr/Ji
        else:
            c_vec = np.linalg.solve(T, i_state)
            c_abs = [np.conj(c)*c for c in c_vec]
            '''
            calculate current
            '''
            Ji = np.dot(i_state, np.dot(Jinc, i_state))
            Jt = sum([t*c_abs[i]*np.abs(np.real(JT[i]/Ji)) for i, t in enumerate(t_state)])
            Jr = sum([r*c_abs[i]*np.abs(np.real(JR[i]/Ji)) for i, r in enumerate(r_state)])
            return Jt, Jr
