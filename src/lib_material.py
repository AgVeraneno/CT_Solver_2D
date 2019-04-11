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
        self.K_norm = 4/3*np.pi/self.a      # m-1. normalized K vector
        self.vF = 8e5                         # m/s. Fermi velocity for graphene
        ### BLG const.
        self.r0 = 2.8*self.q                  # J. A1-B1 hopping energy
        self.r1 = 0.39*self.q                 # J. A2-B1 hopping energy
        self.r3 = 0.315*self.q                # J. A1-B2 hopping energy
        self.vF0 = 1.5*self.r0*self.acc       # m/s. Fermi velocity for graphene
        self.vF3 = 1.5*self.r3*self.acc       # m/s. Fermi velocity for graphene
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
            self.K_norm = 4/3*np.pi/self.a        # m-1. normalized K vector
            ## Hopping energy and Fermi velocity
            self.r0 = 2.8*self.q                  # J. A1-B1 hopping energy
            self.r1 = 0.39*self.q                 # J. A2-B1 hopping energy
            self.r3 = 0.315*self.q*0              # J. A1-B2 hopping energy
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
            H0Kp[0,1] = self.mat.vF0*kx*self.mat.K_norm
            H0Kp[0,3] = self.mat.r1
            H0Kp[1,2] = self.mat.vF3*kx*self.mat.K_norm
            H0Kp[2,3] = self.mat.vF0*kx*self.mat.K_norm
            H0Kp += np.transpose(np.conj(H0Kp))
            H0Kp[0,0] = gap+V-E
            H0Kp[1,1] = gap+V-E
            H0Kp[2,2] = -gap+V-E
            H0Kp[3,3] = -gap+V-E
            # ky independent terms (K- valley)
            H0Kn[0,1] = -self.mat.vF0*kx*self.mat.K_norm
            H0Kn[0,3] = self.mat.r1
            H0Kn[1,2] = -self.mat.vF3*kx*self.mat.K_norm
            H0Kn[2,3] = -self.mat.vF0*kx*self.mat.K_norm
            H0Kn += np.transpose(np.conj(H0Kn))
            H0Kn[0,0] = gap+V-E
            H0Kn[1,1] = gap+V-E
            H0Kn[2,2] = -gap+V-E
            H0Kn[3,3] = -gap+V-E
            # ky dependent terms (K valley)
            H1Kp[0,1] = 1j*self.mat.vF0
            H1Kp[0,3] = -3j*self.mat.r1*self.mat.acc
            H1Kp[1,2] = 1j*self.mat.vF3
            H1Kp[2,3] = 1j*self.mat.vF0
            H1Kp += np.transpose(np.conj(H1Kp))
            # ky dependent terms (K' valley)
            H1Kn[0,1] = 1j*self.mat.vF0
            H1Kn[0,3] = -3j*self.mat.r1*self.mat.acc
            H1Kn[1,2] = 1j*self.mat.vF3
            H1Kn[2,3] = 1j*self.mat.vF0
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
    def linearize_velocity(self):
        ## matrix
        empty_matrix = np.zeros((4,4),dtype=np.complex128)
        H0Kp = copy.deepcopy(empty_matrix)
        H0Kn = copy.deepcopy(empty_matrix)
        if self.m_type == 'Zigzag':
            # ky independent terms (K valley)
            H0Kp[0,1] = -1j*self.mat.vF0
            H0Kp[0,3] = -3j*self.mat.r1*self.mat.acc
            H0Kp[1,2] = -1j*self.mat.vF3
            H0Kp[2,3] = -1j*self.mat.vF0
            H0Kp += np.transpose(np.conj(H0Kp))
            # ky independent terms (K- valley)
            H0Kn[0,1] = -1j*self.mat.vF0
            H0Kn[0,3] = -3j*self.mat.r1*self.mat.acc
            H0Kn[1,2] = -1j*self.mat.vF3
            H0Kn[2,3] = -1j*self.mat.vF0
            H0Kn += np.transpose(np.conj(H0Kn))
            Hi = [[H0Kp, empty_matrix],
                  [empty_matrix, H0Kn]]
            Hi = np.block(Hi)
            return Hi
    def TB_bulk(self, gap, E, V, kx, ky=0):
        ## variables
        E = E*1e-3*self.mat.q
        V = V*1e-3*self.mat.q
        gap = gap*1e-3*self.mat.q
        ## matrix
        empty_matrix = np.zeros((4,4),dtype=np.complex128)
        H0Kp = copy.deepcopy(empty_matrix)
        H0Kn = copy.deepcopy(empty_matrix)
        if self.m_type == 'Zigzag':
            kx_term = np.cos(kx*self.mat.K_norm*self.mat.acc*3**0.5/2)
            ky_term = np.exp(1.5j*ky*self.mat.acc)
            # ky independent terms (K valley)
            H0Kp[0,1] = -self.mat.r0*(1+2*np.conj(ky_term)*kx_term)
            H0Kp[0,3] = self.mat.r1*np.conj(ky_term**2)
            H0Kp[1,2] = -self.mat.r3*(1+2*np.conj(ky_term)*kx_term)
            H0Kp[2,3] = -self.mat.r0*(1+2*np.conj(ky_term)*kx_term)
            H0Kp += np.transpose(np.conj(H0Kp))
            H0Kp[0,0] = gap+V-E
            H0Kp[1,1] = gap+V-E
            H0Kp[2,2] = -gap+V-E
            H0Kp[3,3] = -gap+V-E
            # ky independent terms (K- valley)
            H0Kn[0,1] = -self.mat.r0*(1+2*np.conj(ky_term)*kx_term)
            H0Kn[0,3] = self.mat.r1*np.conj(ky_term**2)
            H0Kn[1,2] = -self.mat.r3*(1+2*np.conj(ky_term)*kx_term)
            H0Kn[2,3] = -self.mat.r0*(1+2*np.conj(ky_term)*kx_term)
            H0Kn += np.transpose(np.conj(H0Kn))
            H0Kn[0,0] = gap+V-E
            H0Kn[1,1] = gap+V-E
            H0Kn[2,2] = -gap+V-E
            H0Kn[3,3] = -gap+V-E
            Hi = [[H0Kp, empty_matrix],
                  [empty_matrix, H0Kn]]
            return np.block(Hi)
    def TB_band(self, gap, E, V, kx, ky=0):
        ## variables
        E = E*1e-3*self.mat.q
        V = V*1e-3*self.mat.q
        gap = gap*1e-3*self.mat.q
        gapA = -gap+V
        gapB = gap+V
        if self.m_type == 'Zigzag':
            kx_term = np.cos(kx*self.mat.K_norm*self.mat.acc*3**0.5/2)
            ## 4 by 4 matrix
            H0 = np.zeros((4,4), dtype=np.complex128)
            H0[0][0] = gapA - E
            H0[0][1] = -self.mat.r0
            H0[1][0] = -2*self.mat.r0*kx_term
            H0[2][1] = -2*self.mat.r3*kx_term
            H0[3][0] = self.mat.r1
            H1 = np.zeros((4,4), dtype=np.complex128)
            H1[0][1] = -2*self.mat.r0*kx_term
            H1[1][0] = -self.mat.r0
            H1[1][1] = gapA - E
            H1[1][0] = -self.mat.r3
            H1[2][1] = -self.mat.r3
            H1[2][2] = gapB - E
            H1[2][3] = -self.mat.r0
            H1[3][2] = -2*self.mat.r0*kx_term
            H2 = np.zeros((4,4), dtype=np.complex128)
            H2[0][3] = self.mat.r1
            H2[1][2] = -2*self.mat.r3*kx_term
            H2[2][3] = -2*self.mat.r0*kx_term
            H2[3][2] = -self.mat.r0
            H2[3][3] = gapB - E
            '''
            travel along armchair edge
            interface with zigzag edge
            '''
            multiper = self.mat.r3/(-2*self.mat.r0*kx_term)
            Ha0 = np.zeros((2,2), dtype=np.complex128)
            Ha0[0][0] = gapA - E
            Ha0[0][1] = -self.mat.r0
            Ha0[1][0] = -2*self.mat.r0*kx_term+multiper*self.mat.r1
            Ha0[1][1] = 0
            Ha1 = np.zeros((2,2), dtype=np.complex128)
            Ha1[0][0] = 0
            Ha1[0][1] = -2*self.mat.r0*kx_term
            Ha1[1][0] = -self.mat.r0
            Ha1[1][1] = gapA - E
            Ha2 = np.zeros((2,2), dtype=np.complex128)
            Ha2[0][0] = 0
            Ha2[0][1] = self.mat.r1
            Ha2[1][0] = -2*self.mat.r3*kx_term+multiper*(-self.mat.r0)
            Ha2[1][1] = (gapB - E)*multiper
            Hb0 = np.zeros((2,2), dtype=np.complex128)
            Hb0[0][0] = (gapA - E)*multiper
            Hb0[0][1] = -2*self.mat.r3*kx_term+multiper*(-self.mat.r0)
            Hb0[1][0] = self.mat.r1
            Hb0[1][1] = 0
            Hb1 = np.zeros((2,2), dtype=np.complex128)
            Hb1[0][0] = gapB - E
            Hb1[0][1] = -self.mat.r0
            Hb1[1][0] = -2*self.mat.r0*kx_term
            Hb1[1][1] = 0
            Hb2 = np.zeros((2,2), dtype=np.complex128)
            Hb2[0][0] = 0
            Hb2[0][1] = -2*self.mat.r0*kx_term+multiper*self.mat.r1
            Hb2[1][0] = -self.mat.r0
            Hb2[1][1] = gapB - E
            ## calculate psi 1
            Hc0 = -np.dot(Hb1, np.dot(np.linalg.inv(Ha2), Ha0))
            Hc1 = Hb0 - np.dot(Hb1, np.dot(np.linalg.inv(Ha2), Ha1)) - np.dot(Hb2, np.dot(np.linalg.inv(Ha2), Ha0))
            Hc2 = -np.dot(Hb2, np.dot(np.linalg.inv(Ha2), Ha1))
            
            H = np.zeros((4,4), dtype=np.complex128)
            H[0][2] = 1
            H[1][3] = 1
            H[2:4, 0:2] = np.dot(np.linalg.inv(-Hc0), Hc2)
            H[2:4, 2:4] = np.dot(np.linalg.inv(-Hc0), Hc1)
            eigVal, eigVec = np.linalg.eig(H)
            eigVec2=np.zeros((4,4), dtype=np.complex128)
            for l_idx, l in enumerate(eigVal):
                eigVec2[0:2,l_idx] = np.dot(np.linalg.inv(-Ha2), np.dot(Ha0*l**2 + Ha1*l, eigVec[0:2,l_idx]))
            newVec = np.zeros((4,4), dtype=np.complex128)
            newVec[0:2,:] = eigVec[0:2, :]
            newVec[2:4,:] = eigVec2[0:2, :]
            ## convert lambda to ky
            a = np.real(eigVal)
            b = np.imag(eigVal)
            ky = (2/(3*self.mat.acc)*np.arctan(b/a) - 1j/(3*self.mat.acc)*np.log(a**2+b**2))/self.mat.K_norm
            return ky, newVec
    def J_op(self, kx, ky1, ky2, isLocal=True):
        if isLocal:
            ky_from = np.conj(ky1)*self.mat.K_norm
            ky_to = ky2*self.mat.K_norm
        else:
            ky_from = ky1*self.mat.K_norm
            ky_to = ky2*self.mat.K_norm
        ## matrix
        empty_matrix = np.zeros((4,4),dtype=np.complex128)
        H0Kp = copy.deepcopy(empty_matrix)
        H0Kn = copy.deepcopy(empty_matrix)
        if self.m_type == 'Zigzag':
            ky_term = np.exp(-1.5j*ky_from*self.mat.acc)+np.exp(-1.5j*ky_to*self.mat.acc)
            ky_term2 = np.exp(-3j*ky_from*self.mat.acc)+np.exp(-3j*ky_to*self.mat.acc)
            # ky independent terms (K valley)
            kxValley = (1+kx)*self.mat.K_norm
            kx_term = np.cos(kxValley*self.mat.acc*3**0.5/2)
            H0Kp[0,1] = -2*self.mat.r0*ky_term*kx_term
            H0Kp[0,3] = 2*self.mat.r1*ky_term2
            H0Kp[1,0] = 2*self.mat.r0*np.conj(ky_term)*kx_term
            H0Kp[1,2] = -2*self.mat.r3*ky_term*kx_term
            H0Kp[2,1] = 2*self.mat.r3*np.conj(ky_term)*kx_term
            H0Kp[2,3] = -2*self.mat.r0*ky_term*kx_term
            H0Kp[3,0] = -2*self.mat.r1*np.conj(ky_term2)
            H0Kp[3,2] = 2*self.mat.r0*np.conj(ky_term)*kx_term
            # ky independent terms (K- valley)
            kxValley = (-1+kx)*self.mat.K_norm
            kx_term = np.cos(kxValley*self.mat.acc*3**0.5/2)
            H0Kn[0,1] = -2*self.mat.r0*ky_term*kx_term
            H0Kn[0,3] = 2*self.mat.r1*ky_term2
            H0Kn[1,0] = 2*self.mat.r0*np.conj(ky_term)*kx_term
            H0Kn[1,2] = -2*self.mat.r3*ky_term*kx_term
            H0Kn[2,1] = 2*self.mat.r3*np.conj(ky_term)*kx_term
            H0Kn[2,3] = -2*self.mat.r0*ky_term*kx_term
            H0Kn[3,0] = -2*self.mat.r1*np.conj(ky_term2)
            H0Kn[3,2] = 2*self.mat.r0*np.conj(ky_term)*kx_term
            Hi = [[H0Kp, empty_matrix],
                  [empty_matrix, H0Kn]]
        return np.block(Hi)