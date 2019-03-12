import numpy as np


class Graphene():
    def __init__(self):
        '''
        Physics const.
        '''
        self.q = 1.6e-19                      # C. electron charge
        self.me = 9.11e-31                    # Kg. electron rest mass
        self.h_bar = 1.05457e-34              # J*s. Planck's const. divided by 2*pi
        self.kB = 1.38e-23                    # J/K. Boltzmann const.
        '''
        Graphene const.
        '''
        self.a = 2.46e-10                     # m. same atom's nearest neighbor distance
        self.acc = self.a/3**0.5              # m. carbon to carbon distance
        self.K_norm = 4/3*np.pi/self.a        # m-1. normalized K vector
        self.vF = 8e5                         # m/s. Fermi velocity for graphene
        ### BLG const.
        self.r0 = 2.8*self.q                  # J. A1-B1 hopping energy
        self.r1 = 0.39*self.q                 # J. A2-B1 hopping energy
        self.r3 = 0.315*self.q                # J. A1-B2 hopping energy
        
    def effectiveMass(self, delta):
        # delta (J)
        return delta/self.vF**2