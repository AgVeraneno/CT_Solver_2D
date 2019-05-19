import copy, os, time
import numpy as np
import CurrentSolver, calBandStructure
import obj_zone

def BarrierSolver(input_list):
    Jt_tot_f = {'+K':np.zeros(len(input_list['E'])), '-K':np.zeros(len(input_list['E']))}
    Jr_tot_f = {'+K':np.zeros(len(input_list['E'])), '-K':np.zeros(len(input_list['E']))}
    Jt_tot_r = {'+K':np.zeros(len(input_list['E'])), '-K':np.zeros(len(input_list['E']))}
    Jr_tot_r = {'+K':np.zeros(len(input_list['E'])), '-K':np.zeros(len(input_list['E']))}
    grid_size = len(input_list['E'])*len(input_list['kx'])
    global_start_time = time.time()
    for kx_idx, kx in enumerate(input_list['kx']):
        step_start_time = time.time()
        print('Calculating Band diagram @ kx=',kx)
        '''
        calculate band edge
        '''
        zone_list = []
        worker = calBandStructure.BandStructure(input_list)
        for z_idx in range(worker.zone_count):
            worker.__refresh__()
            worker.calComplexBand(z_idx, kx_idx)
            new_zone = obj_zone.zone(input_list, worker)
            zone_list.append(new_zone)
            ## plot band diagram
            if input_list['isPlot']:
                title = 'kx='+str(kx)+'K,'+'dt='+str(worker.zone_gap[z_idx])+'(meV),'+'V='+str(worker.zone_V[z_idx])+'(mV)'
                new_zone.plotBand(title)
            ## output csv
            '''
            filename = 'Zone'+str(z_idx+1)+'/kx'+str(kx)+'K,pts='+str(len(input_list['E']))
            if not os.path.exists('../zone_state/Zone'+str(z_idx+1)):
                os.makedirs('../zone_state/Zone'+str(z_idx+1))
            new_zone.export2csv(filename)
            '''
        '''
        calculate transmission
        '''
        print('Calculating transmission @ kx=',kx)
        if kx_idx == 0:
            worker2 = CurrentSolver.CurrentSolver(input_list, worker)
        worker2.T['+K'].append([])
        worker2.T['-K'].append([])
        worker2.T_r['+K'].append([])
        worker2.T_r['-K'].append([])
        worker2.PT.append([])
        worker2.R['+K'].append([])
        worker2.R['-K'].append([])
        worker2.R_r['+K'].append([])
        worker2.R_r['-K'].append([])
        worker2.PR.append([])
        worker2.vel['+K'].append([])
        worker2.vel['-K'].append([])
        worker2.vel_r['+K'].append([])
        worker2.vel_r['-K'].append([])
        worker2.f['+K'].append([])
        worker2.f['-K'].append([])
        worker2.f_r['+K'].append([])
        worker2.f_r['-K'].append([])
        for E_idx, E in enumerate(input_list['E']):
            i_state_f = {'+K':[], '-K':[]}
            i_state_r = {'+K':[], '-K':[]}
            for v in ['+K','-K']:
                '''
                forward
                '''
                ky_list = zone_list[0].ky_list[v][E_idx]
                i_state_f[v], isW = worker2.findIncidentState(ky_list)
                vel = worker2.worker.calVelocity(zone_list[0], E_idx, kx_idx)
                if i_state_f[v] == [0,0,0,0]:
                    worker2.T[v][kx_idx].append(None)
                    worker2.R[v][kx_idx].append(None)
                    worker2.vel[v][kx_idx].append(None)
                elif isW:
                    ## calculate transfer matrix
                    worker2.calTranferMatrix(zone_list, kx, E_idx, v)
                    ## calculate transmission
                    i_state_f[v] = [0,1,0,0]
                    worker2.calTransmission(i_state_f, E_idx, kx_idx, vel, v)
                    tempT = copy.deepcopy(worker2.T[v][kx_idx][-1])
                    tempR = copy.deepcopy(worker2.R[v][kx_idx][-1])
                    worker2.T[v][kx_idx].pop(-1)
                    worker2.R[v][kx_idx].pop(-1)
                    ## calculate transfer matrix
                    worker2.calTranferMatrix(zone_list, kx, E_idx, v)
                    ## calculate transmission
                    i_state_f[v] = [0,0,0,1]
                    worker2.calTransmission(i_state_f, E_idx, kx_idx, vel, v)
                    worker2.T[v][kx_idx][-1] += tempT
                    worker2.R[v][kx_idx][-1] += tempR
                else:
                    ## calculate transfer matrix
                    worker2.calTranferMatrix(zone_list, kx, E_idx, v)
                    ## calculate transmission
                    worker2.calTransmission(i_state_f, E_idx, kx_idx, vel, v)
                '''
                backward
                '''
                ky_list = zone_list[-1].ky_list[v][E_idx]
                i_state_r[v], isW = worker2.findIncidentState(ky_list)
                vel_r = worker2.worker.calVelocity(zone_list[-1], E_idx, kx_idx)
                if i_state_r[v] == [0,0,0,0]:
                    worker2.T_r[v][kx_idx].append(None)
                    worker2.R_r[v][kx_idx].append(None)
                    worker2.vel_r[v][kx_idx].append(None)
                elif isW:
                    ## calculate transfer matrix
                    worker2.calTranferMatrix(zone_list, kx, E_idx, v, 'r')
                    ## calculate transmission
                    i_state_r[v] = [0,1,0,0]
                    worker2.calTransmission(i_state_r, E_idx, kx_idx, vel, v, 'r')
                    tempT = copy.deepcopy(worker2.T_r[v][kx_idx][-1])
                    tempR = copy.deepcopy(worker2.R_r[v][kx_idx][-1])
                    worker2.T_r[v][kx_idx].pop(-1)
                    worker2.R_r[v][kx_idx].pop(-1)
                    ## calculate transfer matrix
                    worker2.calTranferMatrix(zone_list, kx, E_idx, v, 'r')
                    ## calculate transmission
                    i_state_r[v] = [0,0,0,1]
                    worker2.calTransmission(i_state_r, E_idx, kx_idx, vel, v, 'r')
                    worker2.T_r[v][kx_idx][-1] += tempT
                    worker2.R_r[v][kx_idx][-1] += tempR
                else:
                    ## calculate transfer matrix
                    worker2.calTranferMatrix(zone_list, kx, E_idx, v, 'r')
                    ## calculate transmission
                    worker2.calTransmission(i_state_r, E_idx, kx_idx, vel_r, v, 'r')
            ## calculate single polarization
            worker2.calPolarization(E_idx, kx_idx)
        ## plot transmission
        if input_list['isPlot']:
            title = str(input_list['T'])+'K_V='+str(worker.zone_V[z_idx])+'_kx='+str(kx)+'_'+str(input_list['zone_gap'])+'_'+str(input_list['zone_len'])
            worker2.plotTR(title, kx_idx)
        '''
        Calculate polarization
        '''
        print('Calculating polarization')
        '''
        for E_idx, E in enumerate(input_list['E']):
            for v in ['+K', '-K']:
                if v == '+K':
                    thiskx = 1+kx
                elif v == '-K':
                    thiskx = -1+kx
                ## forward current
                thisVel = worker2.vel[v][kx_idx][E_idx]
                thisLambda = zone_list[0].eigVal_list[v][E_idx]
                ky = zone_list[0].ky_list[v][E_idx]
                i_state, isW = worker2.findIncidentState(ky)
                thisky = np.dot(ky, i_state)
                if i_state != [0,0,0,0]:
                    # calculate Fermi distribution
                    V = worker.zone_V[0]
                    gap = worker.zone_gap[0]
                    f = worker2.calFermiDirac(input_list, V, gap, thiskx, thisky, thisLambda)
                    worker2.f[v][kx_idx].append(copy.deepcopy(f))
                    # calculate total current
                    T = worker2.T[v][kx_idx][E_idx]
                    R = worker2.R[v][kx_idx][E_idx]
                    if T != None:
                        Jt_tot_f[v][E_idx] += abs(f*thisVel*T)/grid_size
                    if R != None:
                        Jr_tot_f[v][E_idx] += abs(f*thisVel*R)/grid_size
                else:
                    worker2.f[v][kx_idx].append(None)
                ## backward current
                thisVel = worker2.vel_r[v][kx_idx][E_idx]
                thisLambda = zone_list[-1].eigVal_list[v][E_idx]
                ky = zone_list[-1].ky_list[v][E_idx]
                i_state = worker2.findIncidentState(ky)
                thisky = np.dot(ky, i_state)
                if i_state != [0,0,0,0]:
                    # calculate Fermi distribution
                    V = worker.zone_V[-1]
                    gap = worker.zone_gap[-1]
                    f = worker2.calFermiDirac(input_list, V, gap, thiskx, thisky, thisLambda)
                    worker2.f_r[v][kx_idx].append(copy.deepcopy(f))
                    # calculate total current
                    T = worker2.T_r[v][kx_idx][E_idx]
                    R = worker2.R_r[v][kx_idx][E_idx]
                    if T != None:
                        Jt_tot_r[v][E_idx] += abs(f*thisVel*T)/grid_size
                    if R != None:
                        Jr_tot_r[v][E_idx] += abs(f*thisVel*R)/grid_size
                else:
                    worker2.f_r[v][kx_idx].append(None)
        '''
        step_elapsed_time = time.time() - step_start_time
        print("\n1 step time:",step_elapsed_time, '(s)')
    worker2.JTf = Jt_tot_f
    worker2.JTr = Jt_tot_r
    worker2.JRf = Jr_tot_f
    worker2.JRr = Jr_tot_r
    worker2.calTotalCurrent()
    # output total current csv
    filename = str(input_list['T'])+'K_V='+str(worker.zone_V[z_idx])+'_kx='+str(kx)+'_'+str(input_list['zone_gap'])+'_'+str(input_list['zone_len'])
    worker2.export2csv(filename)
    print('total calculation time:',time.time() - global_start_time, '(s)')

if __name__ == '__main__':
    '''
    define inputs
    '''
    kx1 = np.arange(-0.02, -0.00025, 1e-5)
    kx2 = np.arange(0.00025, 0.02, 1e-5)
    kx = np.append(kx1, kx2)
    kx = [0.0035]
    #kx = np.arange(0.001, 0.008, 0.001)
    input_list = {
        # Assign plotter flag
        'isPlot': True,
        # Assign temperature (K)
        'T': 0,
        # Assign interface type (armchair or zigzag)
        'type': 'zigzag',
        # Sweep energy (meV). E=(startE,stopE,stepE). stepE <= 0.04 meV (current converge)
        'E': np.arange(10, 100, 0.1),
        # Sweep kx (|K|). kx=(startkx,stopkx,pts). pts >= 5000 (current converge)
        'kx': kx,
        # Assign Fermi-Dirac shift (Amp:|K|; Ang:degree)
        'dk': {'Amp':0.001, 'Ang':30},
        # Assign Fermi energy (meV)
        'Ef': 1000,
        ## Assign zone parameters
        # Zone gap height (meV)
        'zone_gap': [10,60,10],
        # Zone length (nm)
        'zone_len': [10,10,10],
        # Assign voltage drop (mV)
        'zone_V': {'I':0,'O':0},
        # Assign mesh in each zone
        'zone_mesh': 1}
    BarrierSolver(input_list)
