import os
import sys
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def shuffle_data(data, labels,global_pl=[]):
  """ Shuffle data and labels.
    Input:
      data: B,N,... numpy array
      label: B,N, numpy array
    Return:
      shuffled data, label and shuffle indices
  """
  idx = np.arange(len(labels))
  np.random.shuffle(idx)
  if global_pl != []:
    return data[idx,:], labels[idx], global_pl[idx,:], idx
  else:
    return data[idx,:], labels[idx],idx


def getDataFiles(list_filename):
  return [line.rstrip() for line in open(list_filename)]

def load_add(h5_filename,names=[]):
  f = h5py.File(h5_filename,'r')
  if len(names) ==0:
    names = list(f.keys())
    print ("Additional distributions: ",names)
    names.remove('data')
    names.remove('pid')

  datasets = {}
  for data in names:
    datasets[data] = f[data][:]

  return datasets

def load_h5(h5_filename,mode='seg',unsup=False,glob=False):
  global_pl = []
  f = h5py.File(h5_filename,'r')
  data = f['data'][:]
  #data = norm_inputs_point_cloud(data)
  if mode == 'class':
    label = f['pid'][:].astype(int)
  elif mode == 'seg':
    label = f['label'][:].astype(int)
  else:
    print('No mode found')
  if glob:
    global_pl = f['global'][:]
    #global_pl = norm_inputs_point_cloud(global_pl,cloud=False)
    
  print("loaded {0} events".format(len(data)))
  return (data, label,global_pl)


def loadDataFile(filename):
  return load_h5(filename)
