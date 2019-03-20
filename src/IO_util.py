import os, csv, copy
import numpy as np
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
def saveAsPlot(file_name, x, y):
    pass