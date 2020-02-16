import argparse
import h5py
from math import *
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os, ast
import sys
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.special import softmax
from scipy.special import expit
#import ROOT
from itertools import combinations
#import importlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..', 'utils'))
import provider
import gapnet_QG as model


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--params', default='[10,1,32,128,128,128,2,64,128,128,128,128,256]', help='DNN parameters[[k,H,A,F,F,F,H,A,F,C,F]]')
parser.add_argument('--gpu', type=int, default=0, help='GPUs to use [default: 0]')
parser.add_argument('--model_path', default='train_results/trained_models/epoch_2.ckpt', help='Model checkpoint path')
parser.add_argument('--plot_path', default='../Plots/QG', help='Path to store created plots [default: ../Plots/QG]')
parser.add_argument('--modeln', type=int,default=0, help='Model number')
parser.add_argument('--batch', type=int, default=1024, help='Batch Size  during training [default: 1024]')
parser.add_argument('--num_point', type=int, default=100, help='Point Number [default: 100]')
parser.add_argument('--data_dir', default='data', help='directory with data [default: data]')
parser.add_argument('--nfeat', type=int, default=8, help='Number of features [default: 8]')
parser.add_argument('--ncat', type=int, default=2, help='Number of categories [default: 2]')
parser.add_argument('--name', default="", help='name of the output file')


FLAGS = parser.parse_args()
MODEL_PATH = FLAGS.model_path
params = ast.literal_eval(FLAGS.params)
DATA_DIR = FLAGS.data_dir
H5_DIR = os.path.join(BASE_DIR, DATA_DIR)

# MAIN SCRIPT
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch
NFEATURES = FLAGS.nfeat


NUM_CATEGORIES = FLAGS.ncat
#Only used to get how many parts per category

print('#### Batch Size : {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(NUM_POINT))
print('#### Using GPUs: {0}'.format(FLAGS.gpu))



    
print('### Starting evaluation')


EVALUATE_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'evaluate_files.txt')) 
print("Loading: ",os.path.join(H5_DIR, 'evaluate_files.txt'))

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
    for idx in range(labels.shape[0]):
      label_one_hot[idx, labels[idx]- MIN_LABEL] = 1
    return label_one_hot

  
def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_pl,  labels_pl, global_pl = model.placeholder_inputs(BATCH_SIZE, NUM_POINT,NFEATURES) #REMEMBER I CHANGED THE ORDER
          
            batch = tf.Variable(0, trainable=False)
                        
            is_training_pl = tf.placeholder(tf.bool, shape=())

            pred,coefs, coefs2, adj = model.get_model(pointclouds_pl, is_training=is_training_pl,global_pl = global_pl,params=params,num_class=NUM_CATEGORIES)
            
            
            loss  = model.get_loss(pred,labels_pl,NUM_CATEGORIES)
            
            saver = tf.train.Saver()
          

    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        
        saver.restore(sess,os.path.join(MODEL_PATH,'model_{0}.ckpt'.format(FLAGS.modeln)))
        print('model restored')
        
        

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'global_pl':global_pl,
               'coefs': coefs,
               'coefs2': coefs2,
               'adj': adj,
               'loss': loss,}
            
        eval_one_epoch(sess,ops)

def get_batch(data,label,global_pl,  start_idx, end_idx):
    batch_label = label[start_idx:end_idx]
    batch_global = global_pl[start_idx:end_idx,:]
    batch_data = data[start_idx:end_idx,:,:]

    return batch_data, batch_label, batch_global


        
def eval_one_epoch(sess,ops):
    is_training = False

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    ncorr = 0
    eval_idxs = np.arange(0, len(EVALUATE_FILES))
    y_val = []
                
    for fn in range(len(EVALUATE_FILES)):
        current_file = os.path.join(H5_DIR,EVALUATE_FILES[eval_idxs[fn]])
        current_data, current_label , current_global = provider.load_h5(current_file,'class',glob=True)
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        

        for batch_idx in range(num_batches):
            scores = np.zeros(NUM_POINT)
            true = np.zeros(NUM_POINT)
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            batch_data, batch_label,batch_global = get_batch(current_data, current_label,current_global, start_idx, end_idx)
            
            cur_batch_size = end_idx-start_idx


            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training,
                         ops['global_pl']:batch_global,
            }
            #,beforemax
            loss, pred, coefs, coefs2 = sess.run([ops['loss'], ops['pred'],
                                                  ops['coefs'],ops['coefs2']],feed_dict=feed_dict)
            
            pred_val = np.argmax(pred, 1)            
            correct = np.sum(pred_val == batch_label)

            total_correct += correct
            total_seen += (BATCH_SIZE)
            loss_sum += np.mean(loss)
            idx_batch=0
            if len(y_val)==0:
                y_val=batch_label
                y_coef1 = np.squeeze(np.max(coefs,-1))
                y_coef2 = np.squeeze(np.max(coefs2,-1))
                y_data = batch_data[:,:,:3] 
                y_sc=pred[:,1]
            else:
                y_val=np.concatenate((y_val,batch_label),axis=0)
                y_coef1=np.concatenate((y_coef1,np.squeeze(np.max(coefs,-1))),axis=0)
                y_coef2=np.concatenate((y_coef2,np.squeeze(np.max(coefs2,-1))),axis=0)
                y_data=np.concatenate((y_data,batch_data[:,:,:3]),axis=0)                
                y_sc=np.concatenate((y_sc,pred[:,1]),axis=0)

    pos_label = 1
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_sc, pos_label=pos_label)

    print('AUC: ',metrics.roc_auc_score(y_val, y_sc))

    signal = y_sc[y_val==1]
    background = y_sc[y_val==0]
    n, bins, patches = plt.hist([signal,background], 50, color=['m','g'], alpha=0.75,
                                range=(0,1), 
                                label=['Signal','Background'],histtype='stepfilled')
    plt.grid(True)
    plt.savefig("{0}/output_{1}.pdf".format(FLAGS.plot_path,FLAGS.name),dpi=150)
    print('Saving DNN output histograms at: ',"../Plots/output_{0}.pdf".format(FLAGS.name))

    fig, base = plt.subplots(dpi=150)
    p = base.semilogy(tpr, 1.0/fpr,color = 'm')
    bineff30 = np.argmax(tpr>0.3)
    bineff50 = np.argmax(tpr>0.5)
    print ('1/effB at {0} effS: '.format(tpr[bineff30]),1.0/fpr[bineff30])
    print ('1/effB at {0} effS: '.format(tpr[bineff50]),1.0/fpr[bineff50])
    base.set_xlabel("True Postive Rate")
    base.set_ylabel("1.0/False Postive Rate")   
    plt.grid(True)
    plt.savefig("{0}/ROC_{1}.pdf".format(FLAGS.plot_path,FLAGS.name))

    total_loss = loss_sum*1.0 / float(num_batches)    
    print('The total accuracy is {0}'.format(total_correct / float(total_seen)))
    npyname = 'GapNet_{0}.npy'.format(FLAGS.name)
    np.save(npyname,y_sc)
    
    with h5py.File('{0}.h5'.format(FLAGS.name), "w") as fh5:
        dset = fh5.create_dataset("pid", data=y_val)
        dset = fh5.create_dataset("DNN", data=y_sc)
        dset = fh5.create_dataset("coef1", data=y_coef1)
        dset = fh5.create_dataset("coef2", data=y_coef2)
        dset = fh5.create_dataset("data", data=y_data)




################################################          
    

if __name__=='__main__':
  eval()
