import copy
import numpy as np

class eigenstate():
    def __init__(self, eigVal, eigVec):
        # import tool box
        self.toolbox = toolbox()
        # initialize variable
        self.rawdata = {'value':eigVal,'state':eigVec}
        self.sorted = {'value':eigVal,'state':eigVec,'conj_state':None}
        self.valley_key = ['+K','-K']
        # get array size
        self.E_len = len(eigVal['+K'])
        self.ky_len = len(eigVal['+K'][0])
    def conjEigenstate(self, val, vec):
        vecConj = copy.deepcopy(vec)
        for valley in self.valley_key:
            for E_idx in range(self.E_len):
                thisVal = val[valley][E_idx]
                thisVec = vec[valley][E_idx]
                newVec = np.zeros((self.ky_len,self.ky_len),dtype=np.complex128)
                # match eigenvalue with conjugate one
                for ky_idx, ky in enumerate(thisVal):
                    if np.imag(round(ky, 10)) == 0:
                        newVec[ky_idx,:] = np.conj(thisVec[:,ky_idx])
                    else:
                        newVec[3-ky_idx,:] = np.conj(thisVec[:,ky_idx])
                # attach newVec to sorted data
                vecConj[valley][E_idx] = copy.deepcopy(newVec)
        self.sorted['conj_state'] = vecConj
    def sortEigenvalue(self):
        val = self.rawdata['value']
        vec = self.rawdata['state']
        for valley in self.valley_key:
            for E_idx in range(self.E_len):
                thisVal = val[valley][E_idx]
                thisVec = vec[valley][E_idx]
                newVal = np.zeros(self.ky_len,dtype=np.complex128)
                newVec = np.zeros((self.ky_len,self.ky_len),dtype=np.complex128)
                ## create sorted ky
                ky_list = list(thisVal)
                ky_list = sorted(ky_list)
                for ky_idx, ky in enumerate(thisVal):
                    if np.imag(round(ky_list[0], 10)) == 0 and np.imag(round(ky_list[1], 10)) == 0 and np.imag(round(ky_list[2], 10)) == 0:
                        isW = True
                    else:
                        isW = False  
                    if np.imag(round(ky, 10)) == 0:     # real state
                        if isW:     # W shape
                            for ky_idx2, ky2 in enumerate(ky_list):
                                if ky2 == ky:
                                    newVal[ky_idx2] = copy.deepcopy(ky)
                                    newVec[:, ky_idx2] = copy.deepcopy(thisVec[:, ky_idx])
                                    break
                        else:       # non W shape
                            if np.real(round(ky, 10)) < 0:  # reflection
                                newVal[0] = copy.deepcopy(ky)
                                newVec[:, 0] = copy.deepcopy(thisVec[:, ky_idx])
                            elif np.real(round(ky, 10)) > 0:  # transmission
                                newVal[3] = copy.deepcopy(ky)
                                newVec[:, 3] = copy.deepcopy(thisVec[:, ky_idx])
                    elif np.imag(round(ky,10)) > 0:     # transmission decay
                        if np.real(round(ky,10)) > 0:
                            newVal[3] = copy.deepcopy(ky)
                            newVec[:, 3] = copy.deepcopy(thisVec[:, ky_idx])
                        elif np.real(round(ky,10)) < 0:
                            newVal[1] = copy.deepcopy(ky)
                            newVec[:, 1] = copy.deepcopy(thisVec[:, ky_idx])
                        else:
                            if newVal[1] == 0:
                                newVal[1] = copy.deepcopy(ky)
                                newVec[:, 1] = copy.deepcopy(thisVec[:, ky_idx])
                            else:
                                newVal[3] = copy.deepcopy(ky)
                                newVec[:, 3] = copy.deepcopy(thisVec[:, ky_idx])
                    elif np.imag(round(ky,10)) < 0:     # reflection decay
                        if np.real(round(ky,10)) > 0:
                            newVal[2] = copy.deepcopy(ky)
                            newVec[:, 2] = copy.deepcopy(thisVec[:, ky_idx])
                        elif np.real(round(ky,10)) < 0:
                            newVal[0] = copy.deepcopy(ky)
                            newVec[:, 0] = copy.deepcopy(thisVec[:, ky_idx])
                        else:
                            if newVal[2] == 0:
                                newVal[2] = copy.deepcopy(ky)
                                newVec[:, 2] = copy.deepcopy(thisVec[:, ky_idx])
                            else:
                                newVal[0] = copy.deepcopy(ky)
                                newVec[:, 0] = copy.deepcopy(thisVec[:, ky_idx])
                    else:
                        raise ValueError("zero state,"+str(ky))
                # attach newVec to sorted data
                self.sorted['value'][valley][E_idx] = newVal
                self.sorted['state'][valley][E_idx] = newVec
        self.conjEigenstate(self.sorted['value'], self.sorted['state'])
    def identifyState(self):
        val = self.rawdata['value']
        vec = self.rawdata['state']
        vecConj = self.sorted['conj_state']
        state = {'+K':[], '-K':[]}
        for valley in self.valley_key:
            for kx_idx in range(self.kx_len):
                state[valley].append([])
                for E_idx in range(self.E_len):
                    state[valley][kx_idx].append(['','','',''])
                    if E_idx == 0:
                        preky = val[valley][E_idx][kx_idx]
                    else:
                        thisky = val[valley][E_idx][kx_idx]
                        kyVec = thisky - preky
                        for idx, dk in enumerate(kyVec):
                            if np.imag(dk) == 0:    # real state
                                if np.real(dk) > 0:     # forward state
                                    state[valley][E_idx][kx_idx][idx] = 'T'
                                elif np.real(dk) < 0:
                                    state[valley][E_idx][kx_idx][idx] = 'R'
                                else:
                                    raise ValueError('encounter zero state @ E=',E_idx)
                            elif np.imag(thisky[idx]) > 0:
                                state[valley][E_idx][kx_idx][idx] = 'Td'
                            elif np.imag(thisky[idx]) < 0:
                                state[valley][E_idx][kx_idx][idx] = 'Rd'
class toolbox():
    def __init__(self):
        pass
    def findVal(self, base, target, precision=7, findFirst=False):
        output = []
        for idx, item in enumerate(base):
            if round(item, precision) == round(target, precision):
                if findFirst:
                    return idx
                else:
                    output.append[idx]
        if findFirst:
            return None
        else:
            return output
    def findMinShift(self, base, target, precision=7):
        testList = [abs(round(b-target, precision)) for b in base]
        return testList.index(min(testList))