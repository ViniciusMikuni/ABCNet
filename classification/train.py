import argparse
import math
import subprocess
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os,ast
import sys
import time
from sklearn import metrics
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
import provider
import gapnet_QG as MODEL

parser = argparse.ArgumentParser()

parser.add_argument('--params', default='[10,1,32,128,128,128,2,64,128,128,128,128,256]', help='DNN parameters[[k,H,A,F,F,F,H,A,F,C,F]]')

parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='gapnet_QG', help='Model name')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=100, help='Point Number [default: 100]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 0.0001]')

parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=2000000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--wd', type=float, default=0.0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--wd_adam', type=float, default=0.0, help='Weight Decay for Adam [Default: 0.0]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--data_dir', default='data', help='directory with data [default: data]')
parser.add_argument('--nfeat', type=int, default=8, help='Number of features [default: 8]')
parser.add_argument('--ncat', type=int, default=2, help='Number of categories [default: 2]')
parser.add_argument('--min', default='acc', help='Condition for early stopping loss or acc [default: acc]')

FLAGS = parser.parse_args()

DATA_DIR = FLAGS.data_dir
H5_DIR = DATA_DIR
EPOCH_CNT = 0
params = ast.literal_eval(FLAGS.params)
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FEAT = FLAGS.nfeat
NUM_CLASSES = FLAGS.ncat
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

#MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR,'..' ,'models', FLAGS.model+'.py')
LOG_DIR = os.path.join('..','logs',FLAGS.log_dir)

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LEARNING_RATE_CLIP = 1e-7
HOSTNAME = socket.gethostname()
EARLY_TOLERANCE=10

TRAIN_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(H5_DIR, 'test_files.txt'))
                                                                   

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl,  labels_pl, global_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NUM_FEAT) #REMEMBER I CHANGED THE ORDER
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            print("--- Get model and loss")


            pred,coefs,coefs2,adj = MODEL.get_model(pointclouds_pl, is_training=is_training_pl,global_pl = global_pl,params=params,
                                   bn_decay=bn_decay,
                                   num_class=NUM_CLASSES, weight_decay=FLAGS.wd)
            
            
            loss = MODEL.get_loss(pred, labels_pl,NUM_CLASSES)
            tf.summary.scalar('loss', loss)

            
            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                #optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=FLAGS.wd_adam,learning_rate=learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)


        
        
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        log_string("Total number of weights for the model: " + str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
      
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'global_pl':global_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'adj':adj,
               'coefs': coefs,
               'coefs2': coefs2,            
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
        }

        best_acc = -1
        
        
        if FLAGS.min == 'loss':early_stop = np.inf
        else:early_stop = 0
        earlytol = 0
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            lss = eval_one_epoch(sess, ops, test_writer)
            cond = lss < early_stop if FLAGS.min=='loss' else lss > early_stop
            if cond:
                early_stop = lss
                earlytol = 0

                # Save the variables to disk.
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model_{0}.ckpt'.format(epoch)))
                log_string("Model saved in file: %s" % save_path)
            else:
                if earlytol >= EARLY_TOLERANCE:break
                else:
                    log_string("No improvement for {0} epochs".format(earlytol))
                    earlytol+=1

            train_one_epoch(sess, ops, train_writer)            
def get_batch(data,label,global_pl,  start_idx, end_idx):
    batch_label = label[start_idx:end_idx]
    batch_global = global_pl[start_idx:end_idx,:]
    #batch_label = label[start_idx:end_idx,:]
    batch_data = data[start_idx:end_idx,:,:]
    return batch_data, batch_label, batch_global

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_idxs)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    y_val=[]
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_file = os.path.join(H5_DIR,TRAIN_FILES[train_idxs[fn]])
        current_data, current_label, current_global = provider.load_h5(current_file,'class',glob=True)
        current_data, current_label,current_global, _ = provider.shuffle_data(current_data, np.squeeze(current_label),global_pl=current_global)
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
                                                                
        log_string(str(datetime.now()))
        
          
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            batch_data, batch_label,batch_global = get_batch(current_data, current_label,current_global, start_idx, end_idx)
          
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training,                         
                         ops['global_pl']:batch_global,
                        
            }
            summary, step, _, loss_val, pred_val, coefs, coefs2 = sess.run([ops['merged'], ops['step'],
                                                                            ops['train_op'], ops['loss'],
                                                                            ops['pred'],
                                                                            ops['coefs'],ops['coefs2']],
                                                                   feed_dict=feed_dict)

            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE)
            loss_sum += np.mean(loss_val)

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_FILES))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    y_val=[]
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_file = os.path.join(H5_DIR,TEST_FILES[test_idxs[fn]])
        current_data, current_label, current_global = provider.load_h5(current_file,'class',glob=True)
        current_data, current_label,current_global, _ = provider.shuffle_data(current_data, np.squeeze(current_label),global_pl=current_global)
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
                                                                
        
        log_string(str(datetime.now()))
        log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            batch_data, batch_label,batch_global = get_batch(current_data, current_label,current_global, start_idx, end_idx)
            cur_batch_size = end_idx-start_idx
            
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training,
                         ops['global_pl']:batch_global,
            }
            if batch_idx ==0:
                start_time = time.time()
                
            summary, step, loss_val, pred_val, coefs, coefs2, adj = sess.run([ops['merged'], ops['step'],
                                                                              ops['loss'], ops['pred'],
                                                                              ops['coefs'],ops['coefs2'],
                                                                              ops['adj'],],
                                                                        feed_dict=feed_dict)
            if batch_idx ==0:
                 duration = time.time() - start_time
                 log_string("Eval time: "+str(duration)) 

            test_writer.add_summary(summary, step)
            pred=pred_val
            pred_val = np.argmax(pred_val, 1)
            
            correct = np.sum(pred_val == batch_label)
            #print (correct)
            total_correct += correct
            total_seen += (BATCH_SIZE)
            loss_sum += np.mean(loss_val)
            if len(y_val)==0:
                y_val=batch_label
                y_sc=pred[:,1]                
            else:
                y_val=np.concatenate((y_val,batch_label),axis=0)
                y_sc=np.concatenate((y_sc,pred[:,1]),axis=0)

    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_sc, pos_label=1)
    bineff30 = np.argmax(tpr>0.3)
    log_string('1/effB at {0} effS: {1}'.format(tpr[bineff30],1.0/fpr[bineff30]))
        
    total_loss = loss_sum*1.0 / float(num_batches)
    log_string('mean loss: %f' % (total_loss))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

    EPOCH_CNT += 1
    if FLAGS.min == 'acc':
        return total_correct / float(total_seen)
    else:
        return total_loss
    


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
