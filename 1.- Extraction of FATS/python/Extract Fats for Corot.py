# coding: utf-8
import os, glob, warnings, FATS, fnmatch
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
import signal
import time
warnings.filterwarnings("ignore", category=DeprecationWarning) 
np.seterr(divide='ignore', invalid='ignore')

### Settings
# - n_cores = la cantidad de cores que vamos a usar para correrlo en paralelo
# - types= Los dos tipos de estrellas de Corot. Solo sirve para poder meterse a los archivos
# - n= Representa la cantidad de tiempos a considerar

n_cores = multiprocessing.cpu_count() - 1
folder_path = './Fats/'
feature_space = FATS.FeatureSpace(Data= ['magnitude','time','error'], featureList=None)

base_path = '/mnt/nas2/GrimaRepo/claguirre/Dataset/Subclasses/'
exp_corot = base_path + 'Corot'     #'*.csv'
exp_ogle = base_path + 'OGLE-III' #'OGLE-*.dat'
exp_vvv = base_path + 'VVV'      #'*.csv'

## Number of Points per Light Curve
n = 10000


#### Metodos
class Timeout():
    """Timeout class using ALARM signal."""
    class Timeout(Exception):
        pass
 
    def __init__(self, sec):
        self.sec = sec
 
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)
 
    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm
 
    def raise_timeout(self, *args):
        raise Timeout.Timeout()

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_files(directory, pattern = "*.csv"):
    matches = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                matches.append(filename)
    return matches

def open_corot(path, n, columns):
    df = pd.read_csv(path, comment='#', sep=',')
    df = df[df.DATEBARTT > 0]

    # 3 Desviaciones Standard
    #df = df[np.abs(df.mjd-df.mjd.mean())<=(3*df.mjd.std())]
    
    time = np.array(df[df.columns[columns[0]]].values, dtype=float)
    magnitude = np.array(df[df.columns[columns[1]]].values, dtype=float)
    error = np.array(df[df.columns[columns[2]]].values, dtype=float)
    
    # Not Nan
    not_nan = np.where(~np.logical_or(np.isnan(time), np.isnan(magnitude)))[0]
    time = time[not_nan]
    magnitude = magnitude[not_nan]
    error = error[not_nan]
    
    if len(time) > n:
        time = time[:n]
        magnitude = magnitude[:n]
        error = error[:n]
    
    return time, magnitude, error

def calculate_one_band_features(path, n, columns):
    time, mag, error = open_corot(path, n, columns)
    preproccesed_data = FATS.Preprocess_LC(mag, time, error)
    lc = np.array(preproccesed_data.Preprocess())
    # feature_space = FATS.FeatureSpace(Data= ['magnitude','time','error'], featureList=None)
    feature_space.calculateFeature(lc)
    return  feature_space.result(method='dict')

def save_file(filename, dictionary):    
    new_df = pd.DataFrame(dictionary, index=['0'])
    new_df.to_csv(filename, float_format='%12.6f', sep=",", index=False)

def run_FATS(path, n):
    print '\t ', path
    fats = [-1]*59
    try:
        with Timeout(TIMEOUT):
            columns = [0, 1, 2]
            if 'EN2_STAR_CHR' in path:
                columns = [0, 4, 8]

            # Save Fats into folder
            directory = os.path.dirname(path) + '/Fats'
            if not os.path.exists(directory):
                print '[+] Creando Directorio \n\t ->', directory
                os.mkdir(directory)    

            directory += '/' + str(n)
            if not os.path.exists(directory):
                print '[+] Creando Directorio \n\t ->', directory
                os.mkdir(directory)        

            name = os.path.basename(path)[:-4] + '.npy'
            new_path = directory + '/' + name

            # Calculate Fats
            if not os.path.exists(new_path):
                fats = calculate_one_band_features(path, n, columns)
                np.save(new_path, fats)
    except: 
        print '\t\t\t Fatal Error'
    return fats

## Corot 
files = get_files(exp_corot)
print '\t\t [+] Total:', len(files)
batchs = list(chunks(files, 10)) 
for batch in batchs:
    result = Parallel(n_jobs=n_cores)(delayed(run_FATS)(i, n) for i in batch)