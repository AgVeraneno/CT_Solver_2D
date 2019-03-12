import sys, copy
sys.path.append('../lib/')
sys.path.append('../src/')
import numpy as np
import lib_Hamiltonian as lH
import lib_material as lm
import lib_sort as ls

class BandStructure():
    def __init__(self, input_list):
        '''
        inputs
        '''
        self.E_sweep = input_list['E']               # energy mesh
        self.kx_sweep = input_list['kx']             # kx mesh
        self.zone_type = input_list['type']         # 2D material direction
        ## create zone profile
        self.zone_gap = []      # zone gap list
        self.zone_len = []      # zone length list.
        self.zone_V = []        # zone V list
        z_mesh = input_list['zone_mesh']
        z_count = len(input_list['zone_gap'])
        V_step = (input_list['zone_V']['O']-input_list['zone_V']['I'])/(z_count*z_mesh-1)
        for z_idx in range(z_count):
            for m in range(z_mesh):
                self.zone_gap.append(input_list['zone_gap'][z_idx])
                self.zone_len.append(input_list['zone_len'][z_idx]/z_mesh)
                self.zone_V.append(input_list['zone_V']['I']+(z_idx*z_mesh+m)*V_step)
        self.zone_count = len(self.zone_len)
        '''
        outputs
        '''
        self.v_ky = {'+K':[],'-K':[]}       # velocity ky
        self.v_lambda = {'+K':[],'-K':[]}       # velocity ky
        self.v_vec = {'+K':[],'-K':[]}      # velocity eigenstate
        self.v_vec_c = {'+K':[],'-K':[]}    # conjugated velocity eigenstate
        self.v_CB = {'+K':[],'-K':[]}       # velocity conduction band edge
        '''
        library
        '''
        self.H = lH.Graphene()
        self.mat = lm.Graphene()
    def __refresh__(self):
        self.v_ky = {'+K':[],'-K':[]}       # velocity ky
        self.v_lambda = {'+K':[],'-K':[]}       # velocity ky
        self.v_vec = {'+K':[],'-K':[]}      # velocity eigenstate
        self.v_vec_c = {'+K':[],'-K':[]}    # conjugated velocity eigenstate
        self.v_CB = {'+K':[],'-K':[]}       # velocity conduction band edge
    def calComplexBand(self, z_idx, kx_idx):
        gap = self.zone_gap[z_idx]*1e-3*self.mat.q
        V = self.zone_V[z_idx]
        kx = self.kx_sweep[kx_idx]
        v_kx = [kx+1, kx-1]
        if self.zone_type == 'zigzag':
            for v_idx, v in enumerate(['+K', '-K']):
                for E_idx, E in enumerate(self.E_sweep):
                    k = {'x': v_kx[v_idx]}
                    eigVal, eigVec = self.H.bandgap(gap, V, k, E)
                    v_ky = self.__eig2ky__(eigVal)
                    ## normalize eigenstate
                    self.v_lambda[v].append(eigVal)
                    for i in range(4):
                        eigVec[:,i] = eigVec[:,i]/np.dot(np.conj(eigVec[:,i]),eigVec[:,i])**0.5
                    self.v_ky[v].append(v_ky)
                    self.v_vec[v].append(eigVec)
            '''
            Sort ky to sequence
            '''
            sorter = ls.eigenstate(self.v_ky, self.v_vec)
            sorter.sortEigenvalue()
            self.v_ky = sorter.sorted['value']
            self.v_vec = sorter.sorted['state']
            self.v_vec_c = sorter.sorted['conj_state']
                
            self.calBandEdge()
    def calBandEdge(self):
        for v in ['+K', '-K']:
            test_queue = self.v_ky[v]
            for E_idx, E in enumerate(self.E_sweep):
                for ky_idx in range(4):
                    if round(abs(np.imag(test_queue[E_idx][ky_idx])),5) == 0:
                        self.v_CB[v].append(E)
                        isCB = True
                        break
                    else:
                        isCB = False
                if isCB:
                    break
            if not isCB:
                self.v_CB[v].append(None)
    def extractKy(self):
        X_ky = {'+K':[],'-K':[]}
        for v in ['+K', '-K']:
            for kx_idx in range(len(self.kx_sweep)):
                X_ky[v].append([])
                for E_idx in range(len(self.E_sweep)):
                    X_ky[v][kx_idx].append(self.v_ky[v][E_idx][kx_idx])
        return X_ky
    def __eig2ky__(self, eigVal):
        if self.zone_type == 'zigzag':
            a = np.real(eigVal)
            b = np.imag(eigVal)
            return (2/(3*self.mat.acc)*np.arctan(b/a) - 1j/(3*self.mat.acc)*np.log(a**2+b**2))/self.mat.K_norm