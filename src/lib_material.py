import numpy as np
import copy

class Graphene():
    def __init__(self):
        '''
        Physics const.
        '''
        self.q = 1.6e-19                      # C. electron charge
        self.me = 9.11e-31                    # Kg. electron rest mass
        self.h_bar = 1.05457e-34              # J*s. Planck's const. divided by 2*pi
        self.kB = 1.38e-23                    # J/K. Boltzmann const
        '''
        Graphene const.
        '''
        self.a = 2.46e-10                     # m. same atom's nearest neighbor distance
        self.acc = self.a/3**0.5              # m. carbon to carbon distance
        self.K_norm = 4/3*np.pi/self.acc      # m-1. normalized K vector
        self.vF = 8e5                         # m/s. Fermi velocity for graphene
        ### BLG const.
        self.r0 = 2.8*self.q                  # J. A1-B1 hopping energy
        self.r1 = 0.39*self.q                 # J. A2-B1 hopping energy
        self.r3 = 0.315*self.q                # J. A1-B2 hopping energy
        
    def effectiveMass(self, delta):
        # delta (J)
        return delta/self.vF**2
class Material():
    def __init__(self, mat):
        self.mat = mat
        '''
        Physics constant
        '''
        self.q = 1.6e-19                      # C. electron charge
        self.me = 9.11e-31                    # Kg. electron rest mass
        self.h_bar = 1.05457e-34              # J*s. Planck's const. divided by 2*pi
        self.kB = 1.38e-23                    # J/K. Boltzmann const
        '''
        Material
        '''
        self.__material__(mat)
    def __material__(self, mat):
        if mat == 'Graphene':
            self.a = 2.46e-10                     # m. same atom's nearest neighbor distance
            self.acc = self.a/3**0.5              # m. carbon to carbon distance
            self.K_norm = 4/3*np.pi/self.acc      # m-1. normalized K vector
            ## Hopping energy and Fermi velocity
            self.r0 = 2.8*self.q                  # J. A1-B1 hopping energy
            self.r1 = 0.39*self.q                 # J. A2-B1 hopping energy
            self.r3 = 0.315*self.q                # J. A1-B2 hopping energy
            self.vF0 = 1.5*self.r0*self.acc       # m/s. Fermi velocity for graphene
            self.vF3 = 1.5*self.r3*self.acc       # m/s. Fermi velocity for graphene
class Hamiltonian():
    def __init__(self, setup):
        self.mat = setup['Material']
        self.lattice = setup['Lattice']
        self.m_type = setup['Direction']
    def linearized(self, gap, E, V, kx, ky=0):
        ## variables
        E = E*1e-3*self.mat.q
        V = V*1e-3*self.mat.q
        gap = gap*1e-3*self.mat.q
        ## matrix
        empty_matrix = np.zeros((4,4),dtype=np.complex128)
        H0Kp = copy.deepcopy(empty_matrix)
        H0Kn = copy.deepcopy(empty_matrix)
        H1Kp = copy.deepcopy(empty_matrix)
        H1Kn = copy.deepcopy(empty_matrix)
        if self.m_type == 'Zigzag':
            # ky independent terms (K valley)
            H0Kp[0,1] = self.mat.vF0*(1+kx)*self.mat.K_norm
            H0Kp[0,3] = self.mat.vF3*(1+kx)*self.mat.K_norm
            H0Kp[1,2] = self.mat.r1
            H0Kp[2,3] = self.mat.vF0*(1+kx)*self.mat.K_norm
            H0Kp += np.transpose(np.conj(H0Kp))
            H0Kp[0,0] = gap+V-E
            H0Kp[1,1] = gap+V-E
            H0Kp[2,2] = -gap+V-E
            H0Kp[3,3] = -gap+V-E
            # ky independent terms (K- valley)
            H0Kn[0,1] = -self.mat.vF0*(-1+kx)*self.mat.K_norm
            H0Kn[0,3] = -self.mat.vF3*(-1+kx)*self.mat.K_norm
            H0Kn[1,2] = self.mat.r1
            H0Kn[2,3] = -self.mat.vF0*(-1+kx)*self.mat.K_norm
            H0Kn += np.transpose(np.conj(H0Kn))
            H0Kn[0,0] = gap+V-E
            H0Kn[1,1] = gap+V-E
            H0Kn[2,2] = -gap+V-E
            H0Kn[3,3] = -gap+V-E
            # ky dependent terms (K valley)
            H1Kp[0,1] = -1j*self.mat.vF0
            H1Kp[0,3] = 1j*self.mat.vF3
            H1Kp[2,3] = -1j*self.mat.vF0
            H1Kp += np.transpose(np.conj(H1Kp))
            # ky dependent terms (K' valley)
            H1Kn[0,1] = -1j*self.mat.vF0
            H1Kn[0,3] = 1j*self.mat.vF3
            H1Kn[2,3] = -1j*self.mat.vF0
            H1Kn += np.transpose(np.conj(H1Kn))
            Hi = [[H0Kp, empty_matrix],
                  [empty_matrix, H0Kn]]
            Hi = np.block(Hi)
            Hp = [[H1Kp, empty_matrix],
                  [empty_matrix, H1Kn]]
            Hp = np.block(Hp)
            return Hi, Hp
        elif self.m_type == 'Armchair':
            # ky independent terms (K valley)
            H0Kp[0,1] = 1j*self.mat.r0*kx
            H0Kp[0,3] = -1j*self.mat.r3*kx
            H0Kp[1,2] = self.mat.r1
            H0Kp[2,3] = 1j*self.mat.r0*kx
            H0Kp += np.transpose(np.conj(H0Kp))
            H0Kp[0,0] = gap+V-E
            H0Kp[1,1] = gap+V-E
            H0Kp[2,2] = -gap+V-E
            H0Kp[3,3] = -gap+V-E
            # ky independent terms (K- valley)
            H0Kn[0,1] = 1j*self.mat.r0*kx
            H0Kn[0,3] = -1j*self.mat.r3*kx
            H0Kn[1,2] = self.mat.r1
            H0Kn[2,3] = 1j*self.mat.r0*kx
            H0Kn += np.transpose(np.conj(H0Kn))
            H0Kn[0,0] = gap+V-E
            H0Kn[1,1] = gap+V-E
            H0Kn[2,2] = -gap+V-E
            H0Kn[3,3] = -gap+V-E
            # ky dependent terms (K valley)
            H1Kp[0,1] = -self.mat.r0
            H1Kp[0,3] = -self.mat.r3
            H1Kp[2,3] = -self.mat.r0
            H1Kp += np.transpose(np.conj(H1Kp))
            # ky dependent terms (K' valley)
            H1Kn[0,1] = -self.mat.r0
            H1Kn[0,3] = -self.mat.r3
            H1Kn[2,3] = -self.mat.r0
            H1Kn += np.transpose(np.conj(H1Kn))
            Hi = [[H0Kp, empty_matrix],
                  [empty_matrix, H0Kn]]
            Hi = np.block(Hi)
            Hp = [[H1Kp, empty_matrix],
                  [empty_matrix, H1Kn]]
            Hp = np.block(Hp)
            return Hi, Hp
        else:
            raise ValueError("Bad input:",self.m_type)