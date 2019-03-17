import os
from matplotlib import pyplot as mplot
import numpy as np
import calTransCurrent as cT
import csv, copy

class zone():
    def __init__(self, input_list, cal_list):
        '''
        Band diagram
        '''
        self.kx_sweep = input_list['kx']
        self.E_sweep = input_list['E']
        self.ky_list = cal_list.v_ky
        self.eigVal_list = cal_list.v_lambda
        self.eig_list = cal_list.v_vec
        self.eigc_list = cal_list.v_vec_c
        self.CB_list = cal_list.v_CB
        '''
        Current
        '''
        self.vel = {'+K':[], '-K':[]}
        self.f = {'+K':[], '-K':[]}
        self.T = {'+K':[], '-K':[]}
        self.R = {'+K':[], '-K':[]}
        self.PT = []
        self.PR = []
        self.JT = {'+K':[], '-K':[]}
        self.JR = {'+K':[], '-K':[]}
        '''
        library
        '''
        self.currentTool = cT.InterfaceCurrent()
    def plotBand(self, title):
        X_ky = self.extractKy()
        f, axes = mplot.subplots(2, 2)
        for idx in range(4):
            axes[0,0].grid()
            axes[0,0].plot(np.real(np.asarray(X_ky['+K'])[:,idx]), self.E_sweep)
            axes[0,0].set_title('ky(Re)')
            #axes[0,0].set_xlabel('ky (|K|)')
            axes[0,0].set_ylabel("K Valley \n E (meV)")
            axes[0,1].grid()
            axes[0,1].plot(np.imag(np.asarray(X_ky['+K'])[:,idx]), self.E_sweep)
            axes[0,1].set_title('ky(Im)')
            #axes[0,1].set_xlabel('ky (|K|)')
            #axes[0,1].set_ylabel('E (meV)')
            axes[0,1].set_xlim((-0.0001, 0.025))
            axes[1,0].grid()
            axes[1,0].plot(np.real(np.asarray(X_ky['-K'])[:,idx]), self.E_sweep)
            #axes[1,0].set_title('+K valley Re')
            axes[1,0].set_xlabel('ky (|K|)')
            axes[1,0].set_ylabel("K' Valley \n E (meV)")
            axes[1,1].grid()
            axes[1,1].plot(np.imag(np.asarray(X_ky['-K'])[:,idx]), self.E_sweep)
            #axes[1,1].set_title('+K valley Im')
            axes[1,1].set_xlabel('ky (|K|)')
            #axes[1,1].set_ylabel('E (meV)')
            axes[1,1].set_xlim((-0.0001, 0.025))
            mplot.suptitle('Band diagram @'+title)
        if not os.path.exists('../figures/'):
            os.mkdir('../figures/')
        mplot.savefig('../figures/Band_diagram_'+str(title)+'.png')
    def plotTR(self):
        for kx_idx, kx in enumerate(self.kx_sweep):
            f, (axes1, axes2) = mplot.subplots(2, 1, sharey=True)
            axes1.grid()
            axes1.plot(self.E_sweep, self.T['-K'][kx_idx])
            axes1.plot(self.E_sweep, self.T['+K'][kx_idx])
            axes1.set_title('transmission')
            #axes1.set_xlabel('E (meV)')
            axes1.set_ylabel("T")
            axes1.set_xlim((self.E_sweep[0], self.E_sweep[-1]))
            axes1.set_ylim((-0.01, 1.01))
            axes2.grid()
            axes2.plot(self.E_sweep, self.R['-K'][kx_idx])
            axes2.plot(self.E_sweep, self.R['+K'][kx_idx])
            axes2.set_title('reflection')
            axes2.set_xlabel('E (meV)')
            axes2.set_ylabel("R")
            axes2.set_xlim((self.E_sweep[0], self.E_sweep[-1]))
            axes2.set_ylim((-0.01, 1.01))
            mplot.suptitle('Transmission & Reflection. dkx='+str(kx)+'|K|')
            mplot.savefig('../figures/TR_dkx='+str(kx)+'K.png')
        #mplot.show()
        with open('TR_dkx='+str(kx)+'K.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel', delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['E','kx','T(K)','R(K)',"T(K')","R(K')"])
            for E_idx, E in enumerate(self.E_sweep):
                for kx_idx, kx in enumerate(self.kx_sweep):
                    output = []
                    output.append(E)
                    output.append(kx)
                    for v in ['+K', '-K']:
                        output.append(np.real(self.T[v][kx_idx][E_idx]))
                        output.append(np.real(self.R[v][kx_idx][E_idx]))
                    PT = (np.real(self.T['+K'][kx_idx][E_idx])-np.real(self.T['-K'][kx_idx][E_idx]))/(np.real(self.T['+K'][kx_idx][E_idx])+np.real(self.T['-K'][kx_idx][E_idx]))
                    PR = (np.real(self.R['+K'][kx_idx][E_idx])-np.real(self.R['-K'][kx_idx][E_idx]))/(np.real(self.R['+K'][kx_idx][E_idx])+np.real(self.R['-K'][kx_idx][E_idx]))
                    output.append(PT)
                    output.append(PR)
                    spamwriter.writerow(output)
    def plotPTR(self, dk):
        f, (axes1, axes2) = mplot.subplots(2, 1, sharey=True)
        axes1.grid()
        axes1.plot(self.E_sweep, self.PT)
        axes1.set_title('transmission polarization')
        axes1.set_xlabel('E (meV)')
        axes1.set_ylabel("(JK-JK')/(JK+JK')")
        axes1.set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        axes1.set_ylim((-1.01, 1.01))
        axes2.grid()
        axes2.plot(self.E_sweep, self.PR)
        axes2.set_title('reflection polarization')
        axes2.set_xlabel('E (meV)')
        axes2.set_ylabel("(JK-JK')/(JK+JK')")
        axes2.set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        axes2.set_ylim((-1.01, 1.01))
        mplot.suptitle('Polarization. k_shift='+str(dk)+'|K|')
        mplot.savefig('../figures/PTR_dk=('+str(dk['x'])+','+str(dk['y'])+').png')
        f, axes = mplot.subplots(2, 2)
        axes[0,0].grid()
        axes[0,0].plot(self.E_sweep, self.JT['-K'])
        axes[0,0].set_title('-K valley transmission current')
        axes[0,0].set_xlabel('E (meV)')
        axes[0,0].set_ylabel('A')
        axes[0,0].set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        axes[1,0].grid()
        axes[1,0].plot(self.E_sweep, self.JR['-K'])
        axes[1,0].set_title('-K valley reflection current')
        axes[1,0].set_xlabel('E (meV)')
        axes[1,0].set_ylabel('A')
        axes[1,0].set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        axes[0,1].grid()
        axes[0,1].plot(self.E_sweep, self.JT['+K'])
        axes[0,1].set_title('+K valley transmission current')
        axes[0,1].set_xlabel('E (meV)')
        axes[0,1].set_ylabel('A')
        axes[0,1].set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        axes[1,1].grid()
        axes[1,1].plot(self.E_sweep, self.JR['+K'])
        axes[1,1].set_title('+K valley reflection current')
        axes[1,1].set_xlabel('E (meV)')
        axes[1,1].set_ylabel('A')
        axes[1,1].set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        mplot.suptitle('Valley current. k_shift='+str(dk)+'|K|')
        mplot.savefig('../figures/Current_dk=('+str(dk['x'])+','+str(dk['y'])+').png')
        f, (axes1, axes2) = mplot.subplots(2, 1, sharey=True)
        axes1.grid()
        axes1.plot(self.E_sweep, self.vel['+K'])
        axes1.set_title('K valley velocity')
        axes1.set_xlabel('E (meV)')
        axes1.set_ylabel("v")
        axes1.set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        axes2.grid()
        axes2.plot(self.E_sweep, self.vel['-K'])
        axes2.set_title("K' valley velocity")
        axes2.set_xlabel('E (meV)')
        axes2.set_ylabel("v")
        axes2.set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        mplot.suptitle('Velocity. k_shift='+str(dk)+'|K|')
        mplot.savefig('../figures/Velocity_dk=('+str(dk['x'])+','+str(dk['y'])+').png')
        ## Fermi distribution
        f, (axes1, axes2) = mplot.subplots(2, 1, sharey=True)
        axes1.grid()
        axes1.plot(self.E_sweep, self.f['+K'])
        axes1.set_title('K valley Fermi distribution')
        axes1.set_xlabel('E (meV)')
        axes1.set_ylabel("f")
        axes1.set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        axes2.grid()
        axes2.plot(self.E_sweep, self.f['-K'])
        axes2.set_title("K' valley Fermi distribution")
        axes2.set_xlabel('E (meV)')
        axes2.set_ylabel("f")
        axes2.set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        mplot.suptitle('Velocity. k_shift='+str(dk)+'|K|')
        mplot.savefig('../figures/Fermi_dk=('+str(dk['x'])+','+str(dk['y'])+').png')
        #mplot.show()
        ## save as csv
        with open('Polarization_dk=('+str(dk['x'])+','+str(dk['y'])+').csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel', delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['E','PT','PR','JT(K)','JR(K)',"JT(K')","JR(K')", "f(K)", "f(K')"])
            for E_idx, E in enumerate(self.E_sweep):
                output = []
                output.append(E)
                output.append(self.PT[E_idx])
                output.append(self.PR[E_idx])
                for v in ['+K', '-K']:
                    output.append(self.JT[v][E_idx])
                    output.append(self.JR[v][E_idx])
                for v in ['+K', '-K']:
                    output.append(self.f[v][E_idx])
                spamwriter.writerow(output)
    def extractKy(self):
        X_ky = {'+K':[],'-K':[]}
        for v in ['+K', '-K']:
            for E_idx in range(len(self.E_sweep)):
                X_ky[v].append(self.ky_list[v][E_idx])
        return X_ky
    def extractTransKy(self, kx_idx, E_idx):
        X_ky = {'+K':0,'-K':0}
        i_state_out = {'+K':0,'-K':0}
        for v in ['+K', '-K']:
            ## incident zone
            ky1 = self.ky_list[v][E_idx][kx_idx]
            ky2 = self.ky_list[v][E_idx+1][kx_idx]
            r_state, t_state, i_state = self.currentTool.calState(ky1, ky2)
            X_ky[v] = np.dot(ky1,i_state)
            i_state_out[v] = copy.deepcopy(i_state)
        return X_ky, i_state_out
    def export2csv(self, filename):
        ## save as csv
        with open('../zone_state/'+filename+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel', delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['E'])
            for E_idx, E in enumerate(self.E_sweep):
                output = []
                output.append(E)
                for v in ['+K', '-K']:
                    for ky in self.ky_list[v][E_idx]:
                        output.append(np.real(ky))
                        output.append(np.imag(ky))
                    for psi1 in self.eig_list[v][E_idx]:
                        for psi2 in psi1:
                            output.append(np.real(psi2))
                            output.append(np.imag(psi2))
                spamwriter.writerow(output)
