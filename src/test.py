import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as mplot

class black_phosphorus():
    def __init__(self, mesh=101):
        ## single layer parameters
        self.a1 = 2.22*1e-10            # A1-B1 lattice constant (m)
        self.a2 = 2.24*1e-10            # B1-A2 lattice constant (m)
        self.ang1 = 96.5                # honeycomb A1 angle (degree)
        self.ang2 = 101.9               # honeycomb B1 angle (degree)
        self.ang3 = 72                  # honeycomb A2-B1 angle (degree)
        '''
        ux: unit cell boundary x
        uy: unit cell boundary y
        uz: sublayer offset in z
        '''
        self.ux = self.a1*np.sin(self.ang1/2 * np.pi/180)
        self.uy1 = self.a1*np.cos(self.ang1/2 * np.pi/180)
        self.uy2 = self.a2*np.sin(self.ang3/2 * np.pi/180)
        self.uz = self.a2*np.cos(self.ang3/2 * np.pi/180)
        ## multi layer parameters (defined between nearest layers)
        self.dx = self.ux               # shift in ZZ direction (m)
        self.dy = self.uy1+self.uy2     # shift in AC direction (m)
        self.dz = 5.345*1e-10           # distance between layers (m)
        ## Band disaapersion range
        self.kx = np.linspace(-np.pi/(2*self.ux), np.pi/(2*self.ux), mesh)
        self.ky = np.linspace(-np.pi/(2*self.ux), np.pi/(2*self.ux), mesh)
        self.kz = np.linspace(-np.pi/(2*self.ux), np.pi/(2*self.ux), mesh)
        self.k0 = np.zeros(mesh)
        ## hopping energy
        self.t1 = -1.486*sc.electron_volt
        self.t2 = 3.729*sc.electron_volt
        self.t3 = -0.252*sc.electron_volt
        self.t4 = -0.071*sc.electron_volt
        self.t5 = -0.019*sc.electron_volt
        self.t6 = 0.186*sc.electron_volt
        self.t7 = -0.063*sc.electron_volt
        self.t8 = 0.101*sc.electron_volt
        self.t9 = -0.042*sc.electron_volt
        self.t10 = 0.073*sc.electron_volt
        self.t1V = 0.524*sc.electron_volt
        self.t2V = 0.18*sc.electron_volt
        self.t3V = -0.123*sc.electron_volt
        self.t4V = -0.168*sc.electron_volt
    def build(self, BDtype='TB', plane='xy'):
        if BDtype == 'TB':
            '''
            Build lattice block for A1-B1-A2-B2
            '''
            ## hopping length
            d1 = np.array([self.ux, self.uy1, 0])
            d2 = np.array([0, -self.uy2, -self.uz])
            d3 = np.array([2*self.ux, 0, 0])
            d4 = np.array([self.ux, -(self.uy1 + 2*self.uy2), 0])
            d5 = np.array([self.ux, self.uy1 + self.uy2, self.uz])
            d6 = np.array([0, 2*self.uy1 + self.uy2, -self.uz])
            d7 = np.array([0, 2*self.uy1 + 2*self.uy2, 0])
            d8 = np.array([4*self.ux, self.uy1, 0])
            d9 = np.array([2*self.ux, 2*self.uy1 + self.uy2, self.uz])
            d10 = np.array([2*self.ux, 2*self.uy1 + 2*self.uy2, 0])
            d1V = np.array([self.ux, self.uy1, self.dz-self.uz])
            d2V = np.array([0, self.uy1+self.uy2, self.dz-self.uz])
            d3V = np.array([2*self.ux, self.uy1+self.uy2, self.dz-self.uz])
            d4V = np.array([self.ux, 2*self.uy1 + self.uy2, self.dz-self.uz])
            '''
            build matrix according to plane direction
            HA1A1 | HA1B1 | HA1A2 | HA1B2
            HB1A1 | HB1B1 | HB1A2 | HB1B2
            HA2A1 | HA2B1 | HA2A2 | HA2B2
            HB2A1 | HB2B1 | HB2A2 | HB2B2
            '''
            if plane == 'xy':
                k_vector = np.block([[self.kx],[self.ky],[self.k0]])
            elif plane == 'xz':
                k_vector = np.block([[self.kx],[self.k0],[self.kz]])
            elif plane == 'yz':
                k_vector = np.block([[self.k0],[self.ky],[self.kz]])
            t1_1 = np.exp(1j*np.dot(d1,k_vector))
            t1_2 = np.exp(1j*np.dot(d1*[-1,1,1],k_vector))
            t2 = np.exp(1j*np.dot(d2,k_vector))
            t3 = np.exp(1j*np.dot(d3,k_vector))
            t4_1 = np.exp(1j*np.dot(d4,k_vector))
            t4_2 = np.exp(1j*np.dot(d4*[-1,1,1],k_vector))
            t5_1 = np.exp(1j*np.dot(d5,k_vector))
            t5_2 = np.exp(1j*np.dot(d5*[-1,1,1],k_vector))
            t5_3 = np.exp(1j*np.dot(d5*[-1,-1,1],k_vector))
            t5_4 = np.exp(1j*np.dot(d5*[1,-1,1],k_vector))
            t6 = np.exp(1j*np.dot(d6,k_vector))
            t7 = np.exp(1j*np.dot(d7,k_vector))
            t8_1 = np.exp(1j*np.dot(d8,k_vector))
            t8_2 = np.exp(1j*np.dot(d8*[-1,1,1],k_vector))
            t9_1 = np.exp(1j*np.dot(d9,k_vector))
            t9_2 = np.exp(1j*np.dot(d9*[-1,1,1],k_vector))
            t10_1 = np.exp(1j*np.dot(d10,k_vector))
            t10_2 = np.exp(1j*np.dot(d10*[-1,1,1],k_vector))
            t10_3 = np.exp(1j*np.dot(d10*[-1,-1,1],k_vector))
            t10_4 = np.exp(1j*np.dot(d10*[1,-1,1],k_vector))
            t1V = np.exp(1j*np.dot(d1V,k_vector))
            t2V = np.exp(1j*np.dot(d2V,k_vector))
            t3V_1 = np.exp(1j*np.dot(d3V,k_vector))
            t3V_2 = np.exp(1j*np.dot(d3V*[-1,1,1],k_vector))
            t4V_1 = np.exp(1j*np.dot(d4V,k_vector))
            t4V_1 = np.exp(1j*np.dot(d4V*[-1,1,1],k_vector))
            ## build Hamiltonian
            eig_val = np.zeros((len(self.k0), 4),dtype=np.complex128)
            h11 = t3 + np.conj(t3) + t7 + np.conj(t7) + t10_1 + t10_2 + t10_3 + t10_4
            h12 = t1_1 + t1_2 + t4_1 + t4_2 + t8_1 + t8_2
            h13 = t5_1 + t5_2 + t5_3 + t5_4
            h14 = t2 + t6 + t9_1 + t9_2
            h21 = np.conj(h12)
            h22 = h11
            h23 = np.conj(h14)
            h24 = np.conj(h13)
            h31 = np.conj(h13)
            h32 = np.conj(h23)
            h33 = h11
            h34 = h12
            h41 = np.conj(h14)
            h42 = np.conj(h24)
            h43 = np.conj(h34)
            h44 = h11
            for i in range(len(self.k0)):
                H = np.array([[h11[i],h12[i],h13[i],h14[i]],
                              [h21[i],h22[i],h23[i],h24[i]],
                              [h31[i],h32[i],h33[i],h34[i]],
                              [h41[i],h42[i],h43[i],h44[i]]])
                val, _ = np.linalg.eig(H)
                eig_val[i,:] = val
            else:
                for i in range(4):
                    mplot.scatter(self.kx, np.real(eig_val[:,i]))
                mplot.show()

        elif BDtype == 'ab-initio':
            '''
            ab initio matrix elements
            '''
            H11 = np.array([[0.0, self.t1, self.t5, self.t6],
                            [self.t1, 0.0, self.t2, self.t5],
                            [self.t5, self.t2, 0.0, self.t1],
                            [self.t6, self.t5, self.t1, 0.0]])
            H12 = np.array([[self.t3, self.t8, 0.0, self.t9],
                            [self.t1, self.t3, 0.0, self.t5],
                            [self.t5, 0.0, self.t3, self.t1],
                            [self.t9, 0.0, self.t8, self.t3]])
            H13 = np.array([[0.0, 0.0, 0.0, 0.0],
                            [self.t8, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, self.t8],
                            [0.0, 0.0, 0.0, 0.0]])
            P11 = np.array([[self.t7, 0.0, 0.0, 0.0],
                            [self.t4, self.t7, 0.0, 0.0],
                            [self.t5, self.t6, self.t7, 0.0],
                            [self.t1, self.t5, self.t4, self.t7]])
            P12 = np.array([[0.0, 0.0, 0.0, 0.0],
                            [self.t4, self.t10, 0.0, 0.0],
                            [self.t5, self.t9, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])
            P21 = np.array([[self.t10, 0.0, 0.0, 0.0],
                            [0.0, self.t10, 0.0, 0.0],
                            [0.0, self.t9, self.t10, 0.0],
                            [0.0, self.t5, self.t4, self.t10]])
            Z11 = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [self.t2V, self.t1V, 0.0, 0.0],
                            [self.t4V, self.t3V, 0.0, 0.0]])
            Z12 = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [self.t3V, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])
            Z21 = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [self.t3V, self.t1V, 0.0, 0.0],
                            [self.t4V, self.t2V, 0.0, 0.0]])
            Z31 = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, self.t3V]])
            L11 = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [self.t2V, self.t4V, 0.0, 0.0],
                            [self.t1V, self.t3V, 0.0, 0.0]])
            L12 = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [self.t3V, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])
            L21 = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [self.t3V, self.t4V, 0.0, 0.0],
                            [self.t1V, self.t2V, 0.0, 0.0]])
            L31 = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, self.t3V, 0.0, 0.0]])

'''
Debug entrance
'''
if __name__ == '__main__':
    BP = black_phosphorus()
    BP.build()
