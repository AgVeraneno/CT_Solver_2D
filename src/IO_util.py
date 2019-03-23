import os, csv, copy
import numpy as np
from matplotlib import pyplot
import lib_material


def load_setup(setup_file, job_file):
    setup = {'CPU_threads':1,
             'Material':False,
             'Lattice':False,
             'Direction':False,
             'H_type':False,
             'Temp':1,
             'Ef':None,
             'E0':None,
             'En':None,
             'dE':None,
             'kx0':None,
             'kxn':None,
             'dkx':None,
             'V2':None,
             'V1':None}
    job = {'gap':0,
           'length':0,
           'mesh':0,
           'V':0}
    '''
    import setup
    '''
    with open(setup_file,newline='') as csv_file:
        rows = csv.DictReader(csv_file)
        for row in rows:
            for key in setup.keys():
                if key[0:2] == 'is':
                    if row[key] == '1':
                        setup[key] = True
                    elif row[key] == '0':
                        setup[key] = False
                    else:
                        raise ValueError('Incorrect input in job file:', row[key])
                elif key == 'Material':
                    setup[key] = lib_material.Material(row[key])
                else:
                    setup[key] = row[key]
    '''
    import jobs
    '''
    with open(job_file,newline='') as csv_file:
        rows = csv.DictReader(csv_file)
        job_list = {}
        for row in rows:
            if row['enable'] == 'o':
                if row['job'] in job_list:
                    for key in job.keys():
                        job_list[row['job']][key].append(row[key])
                else:
                    new_job = copy.deepcopy(job)
                    for key in job.keys():
                        new_job[key] = [row[key]]
                    job_list[row['job']] = new_job
            else:
                continue
    return setup, job_list
def saveAsCSV(file_name, table):
    with open(file_name, 'w', newline='') as csv_file:
        csv_parser = csv.writer(csv_file, delimiter=',',quotechar='|')
        for i in range(np.size(np.array(table), 0)):
            try:
                csv_parser.writerow(list(table[i,:]))
            except:
                csv_parser.writerow(table[i])
def saveAsFigure(file_name, x, y, figure_type=None):
    if figure_type == 'band':
        pyplot.figure(1,figsize=(16,9))
        f, axes = pyplot.subplots(2,1)
        ## plot K valley
        color = ['b','g','r','k']
        for x_idx in range(int(np.size(x['+K'],1))):
            axes[0].grid()
            axes[0].set_title('ky(+K)')
            if x_idx < 2:
                axes[0].plot(np.real(np.array(x['+K'])[:,x_idx]),y, color[x_idx])
                axes[0].plot(np.abs(np.imag(np.array(x['+K'])[:,x_idx])),y, color[x_idx])
        ## plot K' valley
        for x_idx in range(int(np.size(x['-K'],1))):
            axes[1].grid()
            axes[1].set_title('ky(-K)')
            if x_idx < 2:
                axes[1].plot(np.real(np.array(x['-K'])[:,x_idx]),y, color[x_idx])
                axes[1].plot(np.abs(np.imag(np.array(x['-K'])[:,x_idx])),y, color[x_idx])
    elif figure_type == 'PTR':
        pyplot.figure(1,figsize=(16,9))
        f, (axes1, axes2) = pyplot.subplots(2, 1, sharex=True)
        axes1.grid()
        axes1.plot(x, y)
        axes1.set_title('Transmission')
        axes1.set_ylabel("T")
        axes1.legend(["+K","-K"])
        ## polarization
        Py = copy.deepcopy(x)
        for i in range(len(x)):
            if y[i][0] + y[i][1] != 0:
                Py[i] = np.real((y[i][0] - y[i][1])/(y[i][0] + y[i][1]))
            else:
                Py[i] = None
        axes2.grid()
        axes2.plot(x, Py)
        axes2.set_title('Polarization')
        axes2.set_xlabel('E (meV)')
        axes2.set_ylabel("(TK-TK')/(TK+TK')")
        csv_array = np.zeros((len(x),4))
        csv_array[:,0] = x
        csv_array[:,1:3] = np.real(y)
        csv_array[:,3] = Py
        saveAsCSV(file_name+'.csv', csv_array)
    else:
        pyplot.plot(x, y)
    pyplot.savefig(file_name+'.png')
    pyplot.close('all')