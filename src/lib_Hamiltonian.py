import sys, copy
sys.path.append('../lib/')
import numpy as np
import obj_Graphene
from numpy import linalg as LA

class Graphene():
    def __init__(self):
        self.mat = obj_Graphene.Graphene()          # material parameters
    def __eig2ky__(self, eigVal):
        a = np.real(eigVal)
        b = np.imag(eigVal)
        return (2/(3*self.mat.acc)*np.arctan(b/a) - 1j/(3*self.mat.acc)*np.log(a**2+b**2))/self.mat.K_norm
    def bulk(self, gap, V, k, l, E=0, isZigzag=True, isMLG=False):
        '''
        k is in form of {'x': x, 'y':y}
        unit is |K|
        '''
        E = E*1e-3*self.mat.q
        V = V*1e-3*self.mat.q
        gap = gap*1e-3*self.mat.q
        kx = k['x']*self.mat.K_norm
        ky = k['y']*self.mat.K_norm
        if isZigzag:
            '''
            travel along armchair edge
            interface with zigzag edge
            '''
            const_term = np.cos(kx*self.mat.acc*3**0.5/2)
            #k_lambda3 = np.exp(3j*ky*self.mat.acc/2)
            #k_lambda3c = np.exp(-3j*ky*self.mat.acc/2)
            #k_lambda6 = np.exp(3j*ky*self.mat.acc)
            #k_lambda6c = np.exp(-3j*ky*self.mat.acc)
            k_lambda3 = l
            k_lambda3c = np.conj(l)
            k_lambda6 = l**2
            k_lambda6c = np.conj(l**2)
            # constant term
            H0 = np.zeros((4,4), dtype=np.complex128)
            H0[0][0] = -gap+V -E
            H0[0][1] = -self.mat.r0*(1+2*k_lambda3c*const_term)
            H0[0][3] = self.mat.r1*k_lambda6c
            H0[1][0] = -self.mat.r0*(1+2*k_lambda3*const_term)
            H0[1][1] = -gap+V -E
            H0[1][2] = -self.mat.r3*(1+2*k_lambda3c*const_term)
            H0[2][1] = -self.mat.r3*(1+2*k_lambda3*const_term)
            H0[2][2] = gap+V -E
            H0[2][3] = -self.mat.r0*(1+2*k_lambda3c*const_term)
            H0[3][0] = self.mat.r1*k_lambda6
            H0[3][2] = -self.mat.r0*(1+2*k_lambda3*const_term)
            H0[3][3] = gap+V -E
        return H0
    def bandgap(self, gap, V, k, E, typ='zigzag'):
        '''
        k is in form of {'x': x, 'y':y}
        unit is |K|
        unit of E is meV
        '''
        kx = k['x']*self.mat.K_norm
        E = E*1e-3*self.mat.q
        V = V*1e-3*self.mat.q
        gapA = -gap+V
        gapB = gap+V
        if typ == 'zigzag':
            const_term = np.cos(kx*self.mat.acc*3**0.5/2)
            ## 4 by 4 matrix
            H0 = np.zeros((4,4), dtype=np.complex128)
            H0[0][0] = gapA - E
            H0[0][1] = -self.mat.r0
            H0[1][0] = -2*self.mat.r0*const_term
            H0[2][1] = -2*self.mat.r3*const_term
            H0[3][0] = self.mat.r1
            H1 = np.zeros((4,4), dtype=np.complex128)
            H1[0][1] = -2*self.mat.r0*const_term
            H1[1][0] = -self.mat.r0
            H1[1][1] = gapA - E
            H1[1][0] = -self.mat.r3
            H1[2][1] = -self.mat.r3
            H1[2][2] = gapB - E
            H1[2][3] = -self.mat.r0
            H1[3][2] = -2*self.mat.r0*const_term
            H2 = np.zeros((4,4), dtype=np.complex128)
            H2[0][3] = self.mat.r1
            H2[1][2] = -2*self.mat.r3*const_term
            H2[2][3] = -2*self.mat.r0*const_term
            H2[3][2] = -self.mat.r0
            H2[3][3] = gapB - E
            '''
            travel along armchair edge
            interface with zigzag edge
            '''
            multiper = self.mat.r3/(-2*self.mat.r0*const_term)
            Ha0 = np.zeros((2,2), dtype=np.complex128)
            Ha0[0][0] = gapA - E
            Ha0[0][1] = -self.mat.r0
            Ha0[1][0] = -2*self.mat.r0*const_term+multiper*self.mat.r1
            Ha0[1][1] = 0
            Ha1 = np.zeros((2,2), dtype=np.complex128)
            Ha1[0][0] = 0
            Ha1[0][1] = -2*self.mat.r0*const_term
            Ha1[1][0] = -self.mat.r0
            Ha1[1][1] = gapA - E
            Ha2 = np.zeros((2,2), dtype=np.complex128)
            Ha2[0][0] = 0
            Ha2[0][1] = self.mat.r1
            Ha2[1][0] = -2*self.mat.r3*const_term+multiper*(-self.mat.r0)
            Ha2[1][1] = (gapB - E)*multiper
            Hb0 = np.zeros((2,2), dtype=np.complex128)
            Hb0[0][0] = (gapA - E)*multiper
            Hb0[0][1] = -2*self.mat.r3*const_term+multiper*(-self.mat.r0)
            Hb0[1][0] = self.mat.r1
            Hb0[1][1] = 0
            Hb1 = np.zeros((2,2), dtype=np.complex128)
            Hb1[0][0] = gapB - E
            Hb1[0][1] = -self.mat.r0
            Hb1[1][0] = -2*self.mat.r0*const_term
            Hb1[1][1] = 0
            Hb2 = np.zeros((2,2), dtype=np.complex128)
            Hb2[0][0] = 0
            Hb2[0][1] = -2*self.mat.r0*const_term+multiper*self.mat.r1
            Hb2[1][0] = -self.mat.r0
            Hb2[1][1] = gapB - E
            ## calculate psi 1
            Hc0 = -np.dot(Hb1, np.dot(LA.inv(Ha2), Ha0))
            Hc1 = Hb0 - np.dot(Hb1, np.dot(LA.inv(Ha2), Ha1)) - np.dot(Hb2, np.dot(LA.inv(Ha2), Ha0))
            Hc2 = -np.dot(Hb2, np.dot(LA.inv(Ha2), Ha1))
            
            H = np.zeros((4,4), dtype=np.complex128)
            H[0][2] = 1
            H[1][3] = 1
            H[2:4, 0:2] = np.dot(LA.inv(-Hc0), Hc2)
            H[2:4, 2:4] = np.dot(LA.inv(-Hc0), Hc1)
            eigVal, eigVec = LA.eig(H)
            eigVec2=np.zeros((4,4), dtype=np.complex128)
            for l_idx, l in enumerate(eigVal):
                eigVec2[0:2,l_idx] = np.dot(LA.inv(-Ha2), np.dot(Ha0*l**2 + Ha1*l, eigVec[0:2,l_idx]))
            newVec = np.zeros((4,4), dtype=np.complex128)
            newVec[0:2,:] = eigVec[0:2, :]
            newVec[2:4,:] = eigVec2[0:2, :]
            
            return eigVal, newVec
    def current(self, k, typ='local', isZigzag=True):
        '''
        k is in form of {'x': x, 'y':y}
        unit is |K|
        unit of E is meV
        '''
        kx = k['x']*self.mat.K_norm
        ky = k['y']*self.mat.K_norm
        if typ == 'local':
            ky0 = np.conj(k['pre y']*self.mat.K_norm)
        elif typ == 'interface' or typ == 'inter':
            ky0 = k['pre y']*self.mat.K_norm
        if isZigzag:
            '''
            travel along armchair edge
            interface with zigzag edge
            '''
            const_term = np.cos(kx*self.mat.acc*3**0.5/2)
            phase_term = np.exp(-1j*ky0*1.5*self.mat.acc)+np.exp(-1j*ky*1.5*self.mat.acc)
            phase_term_c = np.exp(1j*ky0*1.5*self.mat.acc)+np.exp(1j*ky*1.5*self.mat.acc)
            # constant term
            H0 = np.zeros((4,4), dtype=np.complex128)
            H0[0][1] = -2*self.mat.r0*phase_term*const_term
            H0[0][3] = 2*self.mat.r1*(np.exp(-1j*ky0*3*self.mat.acc)+np.exp(-1j*ky*3*self.mat.acc))
            H0[1][0] = 2*self.mat.r0*phase_term_c*const_term
            H0[1][2] = -2*self.mat.r3*phase_term*const_term
            H0[2][1] = 2*self.mat.r3*phase_term_c*const_term
            H0[2][3] = -2*self.mat.r0*phase_term*const_term
            H0[3][0] = -2*self.mat.r1*(np.exp(1j*ky0*3*self.mat.acc)+np.exp(1j*ky*3*self.mat.acc))
            H0[3][2] = 2*self.mat.r0*phase_term_c*const_term
            return H0
    def velocity(self, k):
        kx = k['x']*self.mat.K_norm
        ky = k['y']*self.mat.K_norm
        v1 = -1j*1.5*self.mat.acc*np.exp(-1j*ky*1.5*self.mat.acc)*np.cos(3**0.5/2*kx*self.mat.acc)
        v2 = -3j*self.mat.acc*np.exp(-1j*ky*3*self.mat.acc)
        H0 = np.zeros((4,4), dtype=np.complex128)
        H0[0][1] = -2*self.mat.r0*v1
        H0[0][3] = self.mat.r1*v2
        H0[1][0] = -2*self.mat.r0*np.conj(v1)
        H0[1][2] = -2*self.mat.r3*v1
        H0[2][1] = -2*self.mat.r3*np.conj(v1)
        H0[2][3] = -2*self.mat.r0*v1
        H0[3][0] = self.mat.r1*np.conj(v2)
        H0[3][2] = -2*self.mat.r0*np.conj(v1)
        return H0
class Hamiltonian():
    def __init__(self):
        self.mat = obj_Graphene.Graphene()          # material parameters
    def bulk(self, FET, k, E=0, isZigzag=True, isMLG=False):
        '''
        k is in form of {'x': x, 'y':y}
        unit is |K|
        '''
        E = E*1e-3*self.mat.q
        kx = k['x']*self.mat.K_norm
        ky = k['y']*self.mat.K_norm
        if isZigzag:
            '''
            travel along armchair edge
            interface with zigzag edge
            '''
            const_term = np.cos(kx*self.mat.acc*3**0.5/2)
            k_lambda3 = np.exp(3j*ky*self.mat.acc/2)
            k_lambda3c = np.exp(-3j*ky*self.mat.acc/2)
            k_lambda6 = np.exp(3j*ky*self.mat.acc)
            k_lambda6c = np.exp(-3j*ky*self.mat.acc)
            # constant term
            H0 = np.zeros((4,4), dtype=np.complex128)
            H0[0][0] = -FET.delta+FET.V_level -E
            H0[0][1] = -self.mat.r0*(1+2*k_lambda3c*const_term)
            H0[0][3] = self.mat.r1*k_lambda6c
            H0[1][0] = -self.mat.r0*(1+2*k_lambda3*const_term)
            H0[1][1] = -FET.delta+FET.V_level -E
            H0[1][2] = -self.mat.r3*(1+2*k_lambda3c*const_term)
            H0[2][1] = -self.mat.r3*(1+2*k_lambda3*const_term)
            H0[2][2] = FET.delta+FET.V_level -E
            H0[2][3] = -self.mat.r0*(1+2*k_lambda3c*const_term)
            H0[3][0] = self.mat.r1*k_lambda6
            H0[3][2] = -self.mat.r0*(1+2*k_lambda3*const_term)
            H0[3][3] = FET.delta+FET.V_level -E
        return H0
    def bandgap(self, FET, k, E, isZigzag=True):
        '''
        k is in form of {'x': x, 'y':y}
        unit is |K|
        unit of E is meV
        '''
        kx = k['x']*self.mat.K_norm
        E = E*1e-3*self.mat.q
        gapA = -FET.delta + FET.V_level
        gapB = FET.delta + FET.V_level
        if isZigzag:
            const_term = np.cos(kx*self.mat.acc*3**0.5/2)
            ## 4 by 4 matrix
            H0 = np.zeros((4,4), dtype=np.complex128)
            H0[0][0] = gapA - E
            H0[0][1] = -self.mat.r0
            H0[1][0] = -2*self.mat.r0*const_term
            H0[2][1] = -2*self.mat.r3*const_term
            H0[3][0] = self.mat.r1
            H1 = np.zeros((4,4), dtype=np.complex128)
            H1[0][1] = -2*self.mat.r0*const_term
            H1[1][0] = -self.mat.r0
            H1[1][1] = gapA - E
            H1[1][0] = -self.mat.r3
            H1[2][1] = -self.mat.r3
            H1[2][2] = gapB - E
            H1[2][3] = -self.mat.r0
            H1[3][2] = -2*self.mat.r0*const_term
            H2 = np.zeros((4,4), dtype=np.complex128)
            H2[0][3] = self.mat.r1
            H2[1][2] = -2*self.mat.r3*const_term
            H2[2][3] = -2*self.mat.r0*const_term
            H2[3][2] = -self.mat.r0
            H2[3][3] = gapB - E
            '''
            travel along armchair edge
            interface with zigzag edge
            '''
            multiper = self.mat.r3/(-2*self.mat.r0*const_term)
            Ha0 = np.zeros((2,2), dtype=np.complex128)
            Ha0[0][0] = gapA - E
            Ha0[0][1] = -self.mat.r0
            Ha0[1][0] = -2*self.mat.r0*const_term+multiper*self.mat.r1
            Ha0[1][1] = 0
            Ha1 = np.zeros((2,2), dtype=np.complex128)
            Ha1[0][0] = 0
            Ha1[0][1] = -2*self.mat.r0*const_term
            Ha1[1][0] = -self.mat.r0
            Ha1[1][1] = gapA - E
            Ha2 = np.zeros((2,2), dtype=np.complex128)
            Ha2[0][0] = 0
            Ha2[0][1] = self.mat.r1
            Ha2[1][0] = -2*self.mat.r3*const_term+multiper*(-self.mat.r0)
            Ha2[1][1] = (gapB - E)*multiper
            Hb0 = np.zeros((2,2), dtype=np.complex128)
            Hb0[0][0] = (gapA - E)*multiper
            Hb0[0][1] = -2*self.mat.r3*const_term+multiper*(-self.mat.r0)
            Hb0[1][0] = self.mat.r1
            Hb0[1][1] = 0
            Hb1 = np.zeros((2,2), dtype=np.complex128)
            Hb1[0][0] = gapB - E
            Hb1[0][1] = -self.mat.r0
            Hb1[1][0] = -2*self.mat.r0*const_term
            Hb1[1][1] = 0
            Hb2 = np.zeros((2,2), dtype=np.complex128)
            Hb2[0][0] = 0
            Hb2[0][1] = -2*self.mat.r0*const_term+multiper*self.mat.r1
            Hb2[1][0] = -self.mat.r0
            Hb2[1][1] = gapB - E
            ## calculate psi 1
            Hc0 = -np.dot(Hb1, np.dot(LA.inv(Ha2), Ha0))
            Hc1 = Hb0 - np.dot(Hb1, np.dot(LA.inv(Ha2), Ha1)) - np.dot(Hb2, np.dot(LA.inv(Ha2), Ha0))
            Hc2 = -np.dot(Hb2, np.dot(LA.inv(Ha2), Ha1))
            
            H = np.zeros((4,4), dtype=np.complex128)
            H[0][2] = 1
            H[1][3] = 1
            H[2:4, 0:2] = np.dot(LA.inv(-Hc0), Hc2)
            H[2:4, 2:4] = np.dot(LA.inv(-Hc0), Hc1)
            eigVal, eigVec = LA.eig(H)
            eigVec2=np.zeros((4,4), dtype=np.complex128)
            for l_idx, l in enumerate(eigVal):
                eigVec2[0:2,l_idx] = np.dot(LA.inv(-Ha2), np.dot(Ha0*l**2 + Ha1*l, eigVec[0:2,l_idx]))
            newVec = np.zeros((4,4), dtype=np.complex128)
            newVec[0:2,:] = eigVec[0:2, :]
            newVec[2:4,:] = eigVec2[0:2, :]
            
            return eigVal, newVec
    def current(self, k, typ='local', isZigzag=True):
        '''
        k is in form of {'x': x, 'y':y}
        unit is |K|
        unit of E is meV
        '''
        kx = k['x']*self.mat.K_norm
        ky = k['y']*self.mat.K_norm
        if typ == 'local':
            ky0 = np.conj(k['pre y']*self.mat.K_norm)
        elif typ == 'interface' or typ == 'inter':
            ky0 = k['pre y']*self.mat.K_norm
        if isZigzag:
            '''
            travel along armchair edge
            interface with zigzag edge
            '''
            const_term = np.cos(kx*self.mat.acc*3**0.5/2)/self.mat.q
            phase_term = np.exp(-1j*ky0*1.5*self.mat.acc)+np.exp(-1j*ky*1.5*self.mat.acc)
            phase_term_c = np.exp(1j*ky0*1.5*self.mat.acc)+np.exp(1j*ky*1.5*self.mat.acc)
            # constant term
            H0 = np.zeros((4,4), dtype=np.complex128)
            H0[0][1] = -2*self.mat.r0*phase_term*const_term
            H0[0][3] = 2*self.mat.r1*(np.exp(-1j*ky0*3*self.mat.acc)+np.exp(-1j*ky*3*self.mat.acc))/self.mat.q
            H0[1][0] = 2*self.mat.r0*phase_term_c*const_term
            H0[1][2] = -2*self.mat.r3*phase_term*const_term
            H0[2][1] = 2*self.mat.r3*phase_term_c*const_term
            H0[2][3] = -2*self.mat.r0*phase_term*const_term
            H0[3][0] = -2*self.mat.r1*(np.exp(1j*ky0*3*self.mat.acc)+np.exp(1j*ky*3*self.mat.acc))
            H0[3][2] = 2*self.mat.r0*phase_term_c*const_term
            '''
            # variable term
            const_term = -2j*3**0.5*np.sin(kx*self.mat.acc*3**0.5/2)
            phase_term = np.exp(1j*ky0*1.5*self.mat.acc)+np.exp(-1j*ky*1.5*self.mat.acc)
            phase_term_c = np.exp(-1j*ky0*1.5*self.mat.acc)+np.exp(1j*ky*1.5*self.mat.acc)
            H1 = np.zeros((4,4), dtype=np.complex128)
            H1[0][1] = const_term*self.mat.r0*phase_term
            H1[1][0] = const_term*self.mat.r0*phase_term_c
            H1[1][2] = const_term*self.mat.r3*phase_term
            H1[2][1] = const_term*self.mat.r3*phase_term_c
            H1[2][3] = const_term*self.mat.r0*phase_term
            H1[3][2] = const_term*self.mat.r0*phase_term_c
            '''
            return H0
