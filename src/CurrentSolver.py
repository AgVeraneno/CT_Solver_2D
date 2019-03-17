import copy, sys, os, csv
sys.path.append('../lib/')
sys.path.append('../src/')
import matplotlib.pyplot as mplot
import numpy as np
import numpy.linalg as LA
import lib_Hamiltonian as lH
import lib_material as lm
import calTransCurrent

class CurrentSolver():
    def __init__(self, input_list, worker):
        self.worker = calTransCurrent.InterfaceCurrent(input_list['T'])
        self.kx_sweep = input_list['kx']
        self.E_sweep = input_list['E']
        self.Temp = input_list['T']
        self.zone_len = copy.deepcopy(worker.zone_len)
        self.zone_len.pop()
        ## output
        self.T = {'+K':[], '-K':[]}
        self.R = {'+K':[], '-K':[]}
        self.T_r = {'+K':[], '-K':[]}
        self.R_r = {'+K':[], '-K':[]}
        self.f = {'+K':[], '-K':[]}
        self.f_r = {'+K':[], '-K':[]}
        self.PT = []
        self.PR = []
        self.vel = {'+K':[], '-K':[]}
        self.vel_r = {'+K':[], '-K':[]}
        self.JTf = {'+K':[], '-K':[]}
        self.JTr = {'+K':[], '-K':[]}
        self.JRf = {'+K':[], '-K':[]}
        self.JRr = {'+K':[], '-K':[]}
        self.JPT = []
        self.JPR = []
        ## Transfer matrix
        self.Jr_f = {'+K':[], '-K':[]}
        self.Jt_f = {'+K':[], '-K':[]}
        self.Ji_f = {'+K':[], '-K':[]}
        self.Jo_f = {'+K':[], '-K':[]}
        self.Jr_r = {'+K':[], '-K':[]}
        self.Jt_r = {'+K':[], '-K':[]}
        self.Ji_r = {'+K':[], '-K':[]}
        self.Jo_r = {'+K':[], '-K':[]}
        '''
        library
        '''
        self.H = lH.Graphene()
        self.mat = lm.Graphene()
    def calTranferMatrix(self, zone_list, kx, E_idx, v, mode='f'):
        '''
        Calculate interface current matrix
        '''
        ## derive local current for terminal zone
        Jt, Jr = self.worker.calLocalCurrent(zone_list[0], E_idx, kx, v)
        self.Jr_f[v] = copy.deepcopy(Jr[v])
        self.Jt_r[v] = copy.deepcopy(Jt[v])
        Jt, Jr = self.worker.calLocalCurrent(zone_list[-1], E_idx, kx, v)
        self.Jt_f[v] = copy.deepcopy(Jt[v])
        self.Jr_r[v] = copy.deepcopy(Jr[v])
        if mode == 'f':
            ## treat barrier (forward)
            Jo_pre_f = None       # previous output current matrix
            this_y = 0
            for z_idx, y in enumerate(self.zone_len):
                this_y += y
                Ji, Jo = self.worker.calInterCurrent(zone_list[z_idx], zone_list[z_idx+1], E_idx, kx, this_y, v)
                if z_idx == 0:      # first zone. save incident current matrix
                    self.Ji_f[v] = copy.deepcopy(Ji)
                    Jo_pre_f = Jo
                else:
                    Jo_pre_f = np.dot(Jo_pre_f, Jo)
            self.Jo_f[v] = copy.deepcopy(Jo_pre_f)
        elif mode == 'r':
            ## treat barrier (reverse)
            Jo_pre_r = None       # previous output current matrix
            this_y = 0
            for z_idx, y in enumerate(self.zone_len):
                z_idx_r = len(self.zone_len)-z_idx-1
                this_y += self.zone_len[z_idx_r]
                Ji, Jo = self.worker.calInterCurrent(zone_list[z_idx_r], zone_list[z_idx_r-1], E_idx, kx, this_y, v)
                if z_idx == 0:      # first zone. save incident current matrix
                    self.Ji_r[v] = copy.deepcopy(Ji)
                    Jo_pre_r = Jo
                else:
                    Jo_pre_r = np.dot(Jo_pre_r, Jo)
            self.Jo_r[v] = copy.deepcopy(Jo_pre_r)
    def findIncidentState(self, ky):
        if np.imag(round(ky[3], 10)) == 0:
            return [0,0,0,1]
        elif np.imag(round(ky[1], 10)) == 0:
            return [0,1,0,0]
        else:
            return [0,0,0,0]
    def calTransmission(self, i_state, E_idx, kx_idx, vel, v, mode='f'):
        t_state = [0,1,0,1]
        r_state = [1,0,1,0]
        ## velocity
        #Tmatrix = np.dot(LA.inv(self.Ji_f[v]), self.Jo_f[v])
        if mode == 'f':
            Tmatrix = self.Jo_f[v]
            c_abs = self.worker.calInterfaceCoeff(Tmatrix, t_state, i_state[v])
            Ji_v = np.dot(i_state[v], np.dot(self.Ji_f[v], i_state[v]))
            Jt = sum([t*c_abs[t_idx]*np.abs(np.real(self.Jt_f[v][t_idx, t_idx]/Ji_v)) for t_idx, t in enumerate(t_state)])
            Jr = sum([r*c_abs[r_idx]*np.abs(np.real(self.Jr_f[v][r_idx, r_idx]/Ji_v)) for r_idx, r in enumerate(r_state)])
            self.T[v][kx_idx].append(copy.deepcopy(Jt))
            self.R[v][kx_idx].append(copy.deepcopy(Jr))
            ## get velocity
            self.vel[v][kx_idx].append(np.real(np.dot(vel[v], i_state[v])))
        elif mode == 'r':
            Tmatrix = self.Jo_r[v]
            c_abs = self.worker.calInterfaceCoeff(Tmatrix, t_state, i_state[v])
            Ji_v = np.dot(i_state[v], np.dot(self.Ji_r[v], i_state[v]))
            Jt = sum([t*c_abs[t_idx]*np.abs(np.real(self.Jt_r[v][t_idx, t_idx]/Ji_v)) for t_idx, t in enumerate(t_state)])
            Jr = sum([r*c_abs[r_idx]*np.abs(np.real(self.Jr_r[v][r_idx, r_idx]/Ji_v)) for r_idx, r in enumerate(r_state)])
            self.T_r[v][kx_idx].append(copy.deepcopy(Jt))
            self.R_r[v][kx_idx].append(copy.deepcopy(Jr))
            ## get velocity
            self.vel_r[v][kx_idx].append(np.real(np.dot(vel[v], i_state[v])))
    def calPolarization(self, E_idx, kx_idx):
        ## calculate Pt
        Tp = self.T['+K'][kx_idx][E_idx]
        Tn = self.T['-K'][kx_idx][E_idx]
        if (Tp == None or Tp == 0) and (Tn == None or Tn == 0):
            self.PT[kx_idx].append(None)
        elif Tp == None:
            self.PT[kx_idx].append(-1)
        elif Tn == None:
            self.PT[kx_idx].append(1)
        else:
            self.PT[kx_idx].append((Tp-Tn)/(Tp+Tn))
        ## calculate Pr
        Rp = self.R['+K'][kx_idx][E_idx]
        Rn = self.R['-K'][kx_idx][E_idx]
        if (Rp == None or Rp == 0) and (Rn == None or Rn == 0):
            self.PR[kx_idx].append(None)
        elif Rp == None:
            self.PR[kx_idx].append(-1)
        elif Rn == None:
            self.PR[kx_idx].append(1)
        else:
            self.PR[kx_idx].append((Rp-Rn)/(Rp+Rn))
    def __eig2ky__(self, eigVal):
        a = np.real(eigVal)
        b = np.imag(eigVal)
        return (2/(3*self.mat.acc)*np.arctan(b/a) - 1j/(3*self.mat.acc)*np.log(a**2+b**2))/self.mat.K_norm
    def calFermiDirac(self, input_list, zone_V, zone_gap, kx, ky, l):
        ## inputs
        dkx = input_list['dk']['Amp']*np.cos(input_list['dk']['Ang']*np.pi/180)
        dky = input_list['dk']['Amp']*np.sin(input_list['dk']['Ang']*np.pi/180)
        Ef = input_list['Ef']
        ## outputs
        k = {'x':kx-dkx, 'y':ky-dky}
        flg = self.__eig2ky__(l)
        for ff, f in enumerate(flg):
            if f == ky:
                lamb = l[ff]
                break
        E, vec = LA.eig(self.H.bulk(zone_gap, zone_V, k, lamb))
        ## find real E
        sorted_E = sorted(E)
        thisE = sorted_E[2]
        #print(np.real(thisE)*1e3/1.6e-19)
        if self.Temp > 10:
            f = 1/(1+np.exp((thisE-(Ef+zone_V)*1e-3*self.mat.q)/(self.mat.kB*self.Temp)))
        else:
            if thisE <= (Ef+zone_V)*1e-3*self.mat.q:
                f = 1
            else:
                f = 0
        return f
    def calTotalCurrent(self):
        # calculate polarization from total current
        Tp = sum(self.JTf['+K']-self.JTr['+K'])
        Tn = sum(self.JTf['-K']-self.JTr['-K'])
        if (Tp == None or Tp == 0) and (Tn == None or Tn == 0):
            self.JPT = None
        elif Tp == None:
            self.JPT = -1
        elif Tn == None:
            self.JPT = 1
        else:
            self.JPT = ((Tp-Tn)/(Tp+Tn))
        Rp = sum(self.JRf['+K'])
        Rn = sum(self.JRf['-K'])
        if (Rp == None or Rp == 0) and (Rn == None or Rn == 0):
            self.JPR = None
        elif Rp == None:
            self.JPR = -1
        elif Rn == None:
            self.JPR = 1
        else:
            self.JPR = ((Rp-Rn)/(Rp+Rn))
    def plotTR(self, title, kx_idx):
        f, (axes1, axes2) = mplot.subplots(2, 1, sharey=True)
        axes1.grid()
        axes1.plot(self.E_sweep, self.T['-K'][kx_idx])
        axes1.plot(self.E_sweep, self.T['+K'][kx_idx])
        axes1.set_ylabel("T")
        axes1.set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        axes1.set_ylim((-0.01, 1.01))
        axes1.legend(['-K', '+K'])
        axes2.grid()
        axes2.plot(self.E_sweep, self.R['-K'][kx_idx])
        axes2.plot(self.E_sweep, self.R['+K'][kx_idx])
        axes2.set_xlabel('E (meV)')
        axes2.set_ylabel("R")
        axes2.set_xlim((self.E_sweep[0], self.E_sweep[-1]))
        axes2.set_ylim((-0.01, 1.01))
        mplot.suptitle('Transmission & Reflection.'+title)
        mplot.savefig('../figures/TR_'+title+'.png')
    def export2csv(self, filename):
        if not os.path.exists('../zone_state/'):
            os.mkdir('../zone_state/')
        with open('../zone_state/TR_'+filename+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel', delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['E','kx','T(K)',"T(K')",'PT','R(K)',"R(K')",'PR','v(K)',"v(K')",'f(K)',"f(K')",'Tr(K)',"Tr(K')",'Rr(K)',"Rr(K')",'vr(K)',"vr(K')",'fr(K)',"fr(K')"])
            for kx_idx, kx in enumerate(self.kx_sweep):
                for E_idx, E in enumerate(self.E_sweep):
                    output = []
                    output.append(E)
                    output.append(kx)
                    output.append(np.real(self.T['+K'][kx_idx][E_idx]))
                    output.append(np.real(self.T['-K'][kx_idx][E_idx]))
                    output.append(np.real(self.PT[kx_idx][E_idx]))
                    output.append(np.real(self.R['+K'][kx_idx][E_idx]))
                    output.append(np.real(self.R['-K'][kx_idx][E_idx]))
                    output.append(np.real(self.PR[kx_idx][E_idx]))
                    output.append(np.real(self.vel['+K'][kx_idx][E_idx]))
                    output.append(np.real(self.vel['-K'][kx_idx][E_idx]))
                    output.append(np.real(self.f['+K'][kx_idx][E_idx]))
                    output.append(np.real(self.f['-K'][kx_idx][E_idx]))
                    output.append(np.real(self.T_r['+K'][kx_idx][E_idx]))
                    output.append(np.real(self.T_r['-K'][kx_idx][E_idx]))
                    output.append(np.real(self.R_r['+K'][kx_idx][E_idx]))
                    output.append(np.real(self.R_r['-K'][kx_idx][E_idx]))
                    output.append(np.real(self.vel_r['+K'][kx_idx][E_idx]))
                    output.append(np.real(self.vel_r['-K'][kx_idx][E_idx]))
                    output.append(np.real(self.f_r['+K'][kx_idx][E_idx]))
                    output.append(np.real(self.f_r['-K'][kx_idx][E_idx]))
                    spamwriter.writerow(output)
        with open('../zone_state/PTR_'+filename+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel', delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['E','JTf(K)',"JTf(K')",'JTr(K)',"JTr(K')",'JRf(K)',"JRf(K')",'JRr(K)',"JRr(K')",'PT','PR'])
            for E_idx, E in enumerate(self.E_sweep):
                output = []
                output.append(E)
                output.append(np.real(self.JTf['+K'][E_idx]))
                output.append(np.real(self.JTf['-K'][E_idx]))
                output.append(np.real(self.JTr['+K'][E_idx]))
                output.append(np.real(self.JTr['-K'][E_idx]))
                output.append(np.real(self.JRf['+K'][E_idx]))
                output.append(np.real(self.JRf['-K'][E_idx]))
                output.append(np.real(self.JRr['+K'][E_idx]))
                output.append(np.real(self.JRr['-K'][E_idx]))
                if E_idx == 0:
                    output.append(np.real(self.JPT))
                    output.append(np.real(self.JPR))
                spamwriter.writerow(output)
if __name__ == '__main__':
    pass