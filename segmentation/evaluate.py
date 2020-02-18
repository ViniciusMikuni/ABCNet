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


np.set_printoptions(threshold=sys.maxsize)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
#from MVA_cfg import *
import provider
import gapnet_PU as MODEL
#import  gapnet_classify_global as model


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--params', default='[50,1,32,64,128,128,2,64,128,128,256,256,256]', help='DNN parameters[[k,H,A,F,F,F,H,A,F,C,F]]')
parser.add_argument('--gpu', type=int, default=0, help='GPUs to use [default: 0]')
parser.add_argument('--model_path', default='../logs/PU/model.ckpt', help='Model checkpoint path')
parser.add_argument('--modeln', type=int,default=0, help='Model number')
parser.add_argument('--batch', type=int, default=64, help='Batch Size  during training [default: 64]')
parser.add_argument('--num_point', type=int, default=500, help='Point Number [default: 500]')
parser.add_argument('--data_dir', default='../data/PU', help='directory with data [default: ../data/PU]')
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


EVALUATE_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'evaluate_files.txt')) # Need to create those
print("Loading: ",os.path.join(H5_DIR, 'evaluate_files.txt'))

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

  
def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_pl,  labels_pl, global_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NFEATURES) #REMEMBER I CHANGED THE ORDER
          
            batch = tf.Variable(0, trainable=False)
                        
            is_training_pl = tf.placeholder(tf.bool, shape=())
            #Model does not wven know about the segmentation, the loss stears it towards the correct direction
            #,beforemax
            pred,coefs, coefs2, adj = MODEL.get_model(pointclouds_pl, is_training=is_training_pl,global_pl = global_pl,params=params,num_class=NUM_CATEGORIES)
            
            
            loss  = MODEL.get_loss(pred,labels_pl)
            
            saver = tf.train.Saver()
          

    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        if FLAGS.modeln >0:
            saver.restore(sess,os.path.join(MODEL_PATH,'model_{0}.ckpt'.format(FLAGS.modeln)))
        else:
            saver.restore(sess,os.path.join(MODEL_PATH,'model.ckpt'))
        print('model restored')
        
        

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'global_pl':global_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'coefs': coefs,
               'coefs2': coefs2,
               'adj': adj,
               'loss': loss,}
            
        eval_one_epoch(sess,ops)

def get_batch(data,label,global_pl, start_idx, end_idx):
    batch_label = label[start_idx:end_idx,:]
    batch_global = global_pl[start_idx:end_idx,:]
    batch_data = data[start_idx:end_idx,:,:]
    return batch_data, batch_label, batch_global

        
def eval_one_epoch(sess,ops):
    is_training = False

    total_correct = total_correct_ones =  total_seen =total_seen_ones= loss_sum =0    
    eval_idxs = np.arange(0, len(EVALUATE_FILES))
    y_val = []
    for fn in range(len(EVALUATE_FILES)):
        current_file = os.path.join(H5_DIR,EVALUATE_FILES[eval_idxs[fn]])
        current_truth = []
        current_mass = []
        current_data, current_label, current_global = provider.load_h5(current_file,'seg',glob=True)
        
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        

        for batch_idx in range(num_batches):
            scores = np.zeros(NUM_POINT)
            true = np.zeros(NUM_POINT)
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            batch_data, batch_label, batch_global = get_batch(current_data, current_label,current_global, start_idx, end_idx)
            
            cur_batch_size = end_idx-start_idx


            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training,
                         ops['global_pl']:batch_global,
            }
            #,beforemax
            loss, pred, coefs, coefs2 = sess.run([ops['loss'], ops['pred'],
                                                  ops['coefs'],ops['coefs2']],feed_dict=feed_dict)
            
            pred_val = np.argmax(pred, 2)

            correct = np.sum(pred_val == batch_label)
            correct_ones = np.sum(pred_val*batch_label)
            total_correct += correct
            total_correct_ones +=correct_ones
            total_seen += (BATCH_SIZE*NUM_POINT)
            total_seen_ones += np.sum(batch_label)
            loss_sum += np.mean(loss)
            if len(y_val)==0:
                y_val=batch_label
                y_data = batch_data[:,:,:] 
                y_glob = batch_global                
                y_sc=pred[:,:,1]
            else:
                y_val=np.concatenate((y_val,batch_label),axis=0)
                y_data=np.concatenate((y_data,batch_data[:,:,:]),axis=0)
                y_glob = np.concatenate((y_glob,batch_global),axis=0)                
                y_sc=np.concatenate((y_sc,pred[:,:,1]),axis=0)

    pos_label = 1
    total_loss = loss_sum*1.0 / float(num_batches)    
    print('The total accuracy is {0}'.format(total_correct / float(total_seen)))
    print('The signal accuracy is {0}'.format(total_correct_ones / float(total_seen_ones)))
    
    with h5py.File('{0}.h5'.format(FLAGS.name), "w") as fh5:
        dset = fh5.create_dataset("pid", data=y_val)
        dset = fh5.create_dataset("DNN", data=y_sc)
        dset = fh5.create_dataset("global", data=y_glob)
        dset = fh5.create_dataset("data", data=y_data)


################################################          
    

if __name__=='__main__':
  eval()
