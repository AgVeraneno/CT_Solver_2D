import sys, copy
sys.path.append('../lib/')
import numpy as np
import numpy.linalg as LA
import lib_Hamiltonian as lH
import lib_material as lm

class InterfaceCurrent():
    def __init__(self, Temp=300):
        '''
        input
        '''
        self.Temp = Temp
        '''
        output
        '''
        self.Jt = {'+K':[],'-K':[]}      # transmission current
        self.Jr = {'+K':[],'-K':[]}      # reflection current
        self.Ji = {'+K':[],'-K':[]}      # incident interface current matrix
        self.Jo = {'+K':[],'-K':[]}      # output interface current matrix
        '''
        library
        '''
        self.H = lH.Graphene()
        self.mat = lm.Graphene()
        '''
        constant
        '''
        self.I0 = 3j*self.mat.q*self.mat.acc/(2*self.mat.h_bar)
    def calLocalCurrent(self, zone, E_idx, kx, v):
        Jt_v = {'+K':[],'-K':[]}
        Jr_v = {'+K':[],'-K':[]}
        Jt = np.zeros((4,4), dtype=np.complex128)      # transmission current
        Jr = np.zeros((4,4), dtype=np.complex128)      # reflection current
        if v == '+K':
            kx = 1+kx
        elif v == '-K':
            kx = -1+kx
        for idx1 in range(4):
            ky1 = zone.ky_list[v][E_idx][idx1]
            psi1 = zone.eig_list[v][E_idx][:, idx1]
            psi1_c = [np.conj(p) for p in psi1]
            for idx2 in range(4):
                ky2 = zone.ky_list[v][E_idx][idx2]
                psi2 = zone.eig_list[v][E_idx][:, idx2]
                k = {'x':kx, 'pre y':ky1, 'y':ky2}
                H = self.H.current(k, typ='local')
                prob = np.dot(psi1_c, np.dot(H, psi2))
                ## transmission current
                if np.imag(ky2-np.conj(ky1)) >= 0:
                    phase = np.exp(1j*(ky2-np.conj(ky1))*self.mat.K_norm*1000*1e-9)
                else:
                    phase = np.exp(1j*(ky2-np.conj(ky1))*self.mat.K_norm*-1000*1e-9)
                Jt[idx1][idx2] = round(self.I0*phase*prob,20)
                ## reflection current
                if np.imag(ky2-np.conj(ky1)) >= 0:
                    phase = np.exp(1j*(ky2-np.conj(ky1))*self.mat.K_norm*1000*1e-9)
                else:
                    phase = np.exp(1j*(ky2-np.conj(ky1))*self.mat.K_norm*-1000*1e-9)
                Jr[idx1][idx2] = round(self.I0*phase*prob,20)
        Jt_v[v] = copy.deepcopy(Jt)
        Jr_v[v] = copy.deepcopy(Jr)
        return Jt_v, Jr_v
    def calInterCurrent(self, z1, z2, E_idx, kx, y, v):
        Ji = np.zeros((4,4), dtype=np.complex128)      # transmission current
        Jo = np.zeros((4,4), dtype=np.complex128)      # reflection current
        if v == '+K':
            kx = 1+kx
        elif v == '-K':
            kx = -1+kx
        for idx1 in range(4):
            ky1 = z1.ky_list[v][E_idx][idx1]
            psi1_c = z1.eigc_list[v][E_idx][idx1]
            for idx2 in range(4):
                ## incident current
                ky2_z1 = z1.ky_list[v][E_idx][idx2]
                psi2_z1 = z1.eig_list[v][E_idx][:, idx2]
                k = {'x':kx, 'pre y':ky1, 'y':ky2_z1}
                H = self.H.current(k, typ='inter')
                prob = np.dot(psi1_c, np.dot(H, psi2_z1))
                phase = np.exp(1j*(ky2_z1-ky1)*y*1e-9*self.mat.K_norm)
                Ji[idx1][idx2] = self.I0*phase*prob
                ## output current
                ky2_z2 = z2.ky_list[v][E_idx][idx2]
                psi2_z2 = z2.eig_list[v][E_idx][:, idx2]
                k = {'x':kx, 'pre y':ky1, 'y':ky2_z2}
                H = self.H.current(k, typ='inter')
                prob = np.dot(psi1_c, np.dot(H, psi2_z2))
                phase = np.exp(1j*(ky2_z2-ky1)*y*1e-9*self.mat.K_norm)
                Jo[idx1][idx2] = self.I0*phase*prob
        Jo = np.dot(LA.inv(Ji), Jo)
        return Ji, Jo
    def calState(self, ky1, ky2):
        r_state = [0,0,0,0]
        t_state = [0,0,0,0]
        i_state = [0,0,0,0]
        r_name = ['','','','']
        t_name = ['','','','']
        isW = False
        for i in range(4):
            dy = ky2[i]-ky1[i]
            if np.imag(dy) == 0:    # traveling state
                if np.real(dy) > 0:     # transmission state
                    t_state[i] = 1
                    t_name[i] = 'T'
                    i_state[i] = 1
                    if sum(i_state) > 1:    # W shape case. drop -ky case
                        for j in range(4):
                            if np.real(ky1[j]) < 0:
                                i_state[j] = 0
                                t_name[j] = 'Tw'
                                isW = True
                            else:
                                pass
                elif np.real(dy) < 0:   # reflection state
                    r_state[i] = 1
                    r_name[i] = 'R'
                    if isW:
                        for j in range(4):
                            if np.real(ky1[j]) < 0:
                                r_name[j] = 'Rw'
                                isW = True
                            else:
                                pass
                else:
                    print("encounter identical state")
            elif np.imag(ky1[i]) > 0:      # transmission decay state
                t_state[i] = 1
                if np.real(ky1[i]) > 0:
                    t_name[i] = 't+'
                elif np.real(ky1[i]) < 0:
                    t_name[i] = 't-'
                else:
                    t_name[i] = 't0'
            elif np.imag(ky1[i]) < 0:      # reflection decay state
                r_state[i] = 1
                if np.real(ky1[i]) > 0:
                    r_name[i] = 'r+'
                elif np.real(ky1[i]) < 0:
                    r_name[i] = 'r-'
                else:
                    r_name[i] = 'r0'
        return [r_state, r_name], [t_state, t_name], i_state
    def calInterfaceCoeff(self, Jt, t, i):
        T = copy.deepcopy(Jt)
        for col in range(4):
            for row in range(4):
                T[row][col] = T[row][col]*t[col]
                if t[col] == 0:
                    T[col][col] = -1
        c_vec = LA.solve(T, i)
        c_abs = [np.conj(c)*c for c in c_vec]       # coefficient magnitude
        return c_abs
    def calVelocity(self, zone, E_idx, kx_idx):
        thiskx = zone.kx_sweep[kx_idx]
        vel = {'+K':[0,0,0,0],'-K':[0,0,0,0]}
        for v in ['+K', '-K']:
            if v == '+K':
                kx = 1+thiskx
            elif v == '-K':
                kx = -1+thiskx
            for idx in range(4):
                ky1 = zone.ky_list[v][E_idx][idx]
                psi1 = zone.eig_list[v][E_idx][:, idx]
                psi1_c = [np.conj(p) for p in psi1]
                ky2 = zone.ky_list[v][E_idx][idx]
                psi2 = zone.eig_list[v][E_idx][:, idx]
                k = {'x':kx, 'pre y':ky1, 'y':ky2}
                H = self.H.velocity(k)
                vel[v][idx] = np.dot(psi1_c, np.dot(H, psi2))
        return vel
    def calShiftFermi(self, zone, kx_idx, ky, input_list):
        ## inputs
        dkx = input_list['dk']['Amp']*np.cos(input_list['dk']['Ang'])
        dky = input_list['dk']['Amp']*np.sin(input_list['dk']['Ang'])
        Ef = input_list['Ef']
        V = input_list['V']['In']
        gap = input_list['node']['gap']['In']
        ## outputs
        f = {'+K':0,'-K':0}
        kx = zone.kx_sweep[kx_idx]
        for v in ['+K', '-K']:
            if v == '+K':
                thiskx = 1+kx
            elif v == '-K':
                thiskx = -1+kx
            k = {'x':thiskx-dkx, 'y':ky[v]-dky}
            E, vec = LA.eig(self.H.bulk(gap, V, k))
            ## find real E
            sorted_E = sorted(E)
            thisE = sorted_E[2]
            #print(v+","+str(np.real(thisE)/self.mat.q))
            #print(thisE/self.mat.q- Ef*1e-3*self.mat.q)
            if self.Temp > 10:
                f[v] = 1/(1+np.exp((thisE-(Ef+V)*1e-3*self.mat.q)/(self.mat.kB*self.Temp)))
            else:
                if thisE <= (Ef+V)*1e-3*self.mat.q:
                    f[v] = 1
                else:
                    f[v] = 0
        return f
