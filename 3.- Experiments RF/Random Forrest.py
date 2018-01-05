# coding: utf-8
from sklearn.ensemble import RandomForestClassifier
import glob, os
import itertools
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm

## Settings
# Technical Setting
limit = 8000 # Maximum amount of Star Per Class
permutation = True # Permute Files
n = 1000
folds = 10

# Files Setting
base_path = '/mnt/nas2/GrimaRepo/claguirre/Dataset/DataRF/'
regular_exp1 = base_path + 'Data/Corot/**/*.npy'
regular_exp2 = base_path + 'Data/**/OGLE-*.npy'
regular_exp3 = base_path + 'Data/VVV/**/*.npy'

## Methods
# subclasses = ['cepDiez', 'cepEfe', 'RRab', 'RRc', 'nonEC', 'EC', 'Mira', 'SRV', 'Osarg'] 
subclasses = ['lpv','cep','rrlyr','ecl']

def get_survey(path):
    if 'Corot' in path:
        return 'Corot'
    elif 'VVV' in path:
        return 'VVV'
    elif 'OGLE' in path:
        return 'OGLE'
    else:
        return 'err'

def get_name(path):
    for subclass in subclasses:
        if subclass in path:
            return subclass
    return 'err'  

def get_name_with_survey(path):
    for subclass in subclasses:
        if subclass in path:
            survey = get_survey(path)
            return survey + '_' + subclass
    return 'err'


def get_files(permutation=False):
    files1 = np.array(list(glob.iglob(regular_exp1, recursive=True)))
    files2 = np.array(list(glob.iglob(regular_exp2, recursive=True)))    
    files3 = np.array(list(glob.iglob(regular_exp3, recursive=True)))
    
    print('[!] Files in Memory')
    
    # Permutations
    if permutation:
        files1 = files1[np.random.permutation(len(files1))]
        files2 = files2[np.random.permutation(len(files2))]
        files3 = files3[np.random.permutation(len(files3))]
        
        print('[!] Permutation applied')
        
    aux_dic = {}
    corot = {}
    vvv = {}
    ogle = {}
    for subclass in subclasses:
        aux_dic[subclass] = []
        corot[subclass] = 0
        vvv[subclass] = 0
        ogle[subclass] = 0

        
    new_files = []
    for idx in tqdm(range(len(files2))):
        foundCorot = False
        foundVista = False
        foundOgle = False
        
        for subclass in subclasses:        
            # Corot
            if not foundCorot and corot[subclass] < limit and idx < len(files1) and subclass in files1[idx]:
                new_files += [files1[idx]]
                corot[subclass] += 1
                foundCorot = True
                    
            # Ogle
            if not foundOgle and ogle[subclass] < limit and subclass in files2[idx]:
                new_files += [files2[idx]]
                ogle[subclass] += 1
                foundOgle = True            
                    
            # VVV           
            if not foundVista and vvv[subclass] < limit and idx < len(files3) and subclass in files3[idx]:
                new_files += [files3[idx]]
                vvv[subclass] += 1
                foundVista = True   
        
    del files1, files2, files3

    print('[!] Loaded Files')
    
    files = np.array(new_files)
    
    # Permutation
    for i in range(10):
        files = files[np.random.permutation(len(files))]
    
    # Class Names
    Y = np.array([get_name(i) for i in files])
    mask = np.where(~(Y == 'err'))
    Y, files = Y[mask], files[mask]
    
    return files, Y

def load_dic(files, Y):
    
    df = pd.DataFrame([])
    new_y = []
    new_subclass = []   
    survey = []
    for idx, i in enumerate(tqdm(files)):
        
        dic = np.load(i, encoding='latin1').item()
        
        ############################
        ### Eliminate Inf Values ###
        ############################
        
        df_aux = pd.DataFrame.from_dict(dic, orient='index')
        if len(np.where(np.isinf(df_aux))[0]) == 0:
            if len(df) == 0:
                df = df_aux.T
            else:
                df = df.append(df_aux.T)
            new_y.append(get_name_with_survey(i))
            new_subclass.append( Y[idx])
            survey.append(get_survey(i))
            
            
    return df, np.array(new_y), np.array(new_subclass), np.array(survey)
        
files, Y = get_files(permutation)
dic, Y, ySubClass, survey = load_dic(files, Y)

########################
## Replace Nan Values ##
########################

dic = dic.apply(lambda x: x.fillna(x.mean()), axis=0)

## Random Forrest
yReal = []
yPred = []
sReal = []

X = dic.values
skf = StratifiedKFold(n_splits=folds, shuffle =True)
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = ySubClass[train_index], ySubClass[test_index]
    s_test = survey[test_index]

    ########################
    #### Random Forrest ####
    ########################
    
    clf = RandomForestClassifier(min_samples_leaf=100, random_state=0)
    clf.fit(X_train, y_train)
    
    yPred = np.append(yPred, clf.predict(X_test))
    yReal = np.append(yReal, y_test)
    sReal = np.append(sReal, s_test)
    # break

directory = './Resultados'
if not os.path.exists(directory):
    print('[+] Creando Directorio \n\t ->', directory)
    os.mkdir(directory)

filename_exp =  directory + '/data'   
print('\n \t\t\t [+] Saving Results in', filename_exp)
np.save(filename_exp, [yReal, yPred, sReal])
