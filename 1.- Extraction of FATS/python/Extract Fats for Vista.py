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
n_cores = multiprocessing.cpu_count() - 2
folder_path = './Fats/'
feature_space = FATS.FeatureSpace(Data= ['magnitude','time','error'], featureList=None)

# base_path = '/Users/Carlos/Desktop/Redes/Data/'
base_path = '/mnt/nas2/GrimaRepo/claguirre/Redes/Subclasses/'
exp_corot = base_path + 'Corot'     #'*.csv'
exp_ogle = base_path + 'OGLE-III' #'OGLE-*.dat'
exp_vvv = base_path + 'VVV'      #'*.csv'
TIMEOUT = 100 # segundos

#### Methods
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
                
def open_vista(path):
    df = pd.read_csv(path, comment='#', sep=',')
    df = df[df.mjd > 0]
    df = df.sort_values(by=[df.columns[1]])
 
    time = np.array(df[df.columns[1]].values, dtype=float)
    magnitude = np.array(df[df.columns[2]].values, dtype=float)
    error = np.array(df[df.columns[3]].values, dtype=float)
    
    # Not Nan
    not_nan = np.where(~np.logical_or(np.isnan(time), np.isnan(magnitude)))[0]
    time = time[not_nan]
    magnitude = magnitude[not_nan]
    error = error[not_nan]
    
    return time, magnitude, error


def first_fats(path):
    lc = FATS.ReadLC_MACHO(path)
    [mag, time, error] = lc.ReadLC()
    preproccesed_data = FATS.Preprocess_LC(mag, time, error)
    lc = np.array(preproccesed_data.Preprocess())
    b = feature_space.calculateFeature(lc)

def calculate_one_band_features(path):
    time, mag, error = open_vista(path)
    preproccesed_data = FATS.Preprocess_LC(mag, time, error)
    lc = np.array(preproccesed_data.Preprocess())
    # feature_space = FATS.FeatureSpace(Data= ['magnitude','time','error'], featureList=None)
    feature_space.calculateFeature(lc)
    return  feature_space.result(method='dict')
    
def run_FATS(path):
    print '\t ', path
    fats = [-1]*59
    try:
        with Timeout(TIMEOUT):
            # Save Fats into folder
            directory = os.path.dirname(path) + '/Fats'
            name = os.path.basename(path)[:-4] + '.npy'
            if not os.path.exists(directory):
                print '[+] Creando Directorio \n\t ->', directory
                os.mkdir(directory)    
            new_path = directory + '/' + name

            # Calculate Fats
            if not os.path.exists(new_path):
                fats = calculate_one_band_features(path)
                np.save(new_path, fats)
    except: 
        print '\t\t\t Fatal Error'
    return fats

### VVV Fats
files = get_files(exp_vvv)
print '\t\t [+] Total:', len(files)
batchs = list(chunks(files, 10)) 
for batch in batchs:
    result = Parallel(n_jobs=n_cores)(delayed(run_FATS)(i) for i in batch)