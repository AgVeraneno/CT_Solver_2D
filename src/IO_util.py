import os, csv, copy
import lib_material


def load_setup(setup_file, job_file):
    setup = {'Material':False,
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
             'V1':0,
             'V2':0}
    job = {'region':-1,
           'cell_type':1,
           'shift':0,
           'width':0,
           'length':0,
           'Vtop':0,
           'Vbot':0,
           'gap':0}
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
        job_list = []
        for row in rows:
            if row['enable'] == 'o':
                new_job = copy.deepcopy(job)
                for key in job.keys():
                    new_job[key] = row[key]
                job_list.append(new_job)
            else:
                continue
    return setup, job_list