import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util
from gat_layers import attn_feature


def placeholder_inputs(batch_size, num_point, num_features):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_features))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    global_pl = tf.placeholder(tf.float32, shape=(batch_size,2)) #Im lazy
    return pointclouds_pl,  labels_pl,global_pl


def gap_block(k,n_heads,nn_idx,net,point_cloud,edge_size,bn_decay,weight_decay,is_training,scname):
    attns = []
    local_features = []
    for i in range(n_heads):
        edge_feature, coefs, locals = attn_feature(net, edge_size[1], nn_idx, activation=tf.nn.relu,
                                                   in_dropout=0.6,
                                                   coef_dropout=0.6, is_training=is_training, bn_decay=bn_decay,
                                                   layer='layer{0}'.format(edge_size[0])+scname, k=k, i=i)
        attns.append(edge_feature)# This is the edge feature * att. coeff. activated by RELU, one per particle
        local_features.append(locals) #Those are the yij


    neighbors_features = tf.concat(attns, axis=-1)
    #coefs = tf.reduce_sum(neighbors_features,axis=-1)
    neighbors_features = tf.concat([tf.expand_dims(point_cloud, -2), neighbors_features], axis=-1)

    locals_transform = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keep_dims=True)

    return neighbors_features, locals_transform, coefs


def get_model(point_cloud, is_training, num_class,global_pl,params, 
                weight_decay=None, bn_decay=None,scname=''):
    ''' input: BxNxF
    Use https://arxiv.org/pdf/1902.08570 as baseline
    output:BxNx(cats*segms)  '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -2)

  
    k = params[0]
    #adj = tf_util.pairwise_distance(point_cloud[:,:,:3])
    adj = tf_util.pairwise_distanceR(point_cloud[:,:,:3])
    n_heads = params[1]
    nn_idx = tf_util.knn(adj, k=k)

    
    net, locals_transform, coefs= gap_block(k,n_heads,nn_idx,point_cloud,point_cloud,('filter0',params[2]),bn_decay,weight_decay,is_training,scname)
    print('shape',net.get_shape())

    net = tf_util.conv2d(net, params[4], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='gapnet01'+scname, bn_decay=bn_decay)
    net01 = net


    net = tf_util.conv2d(net, params[5], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='gapnet02'+scname, bn_decay=bn_decay)

    net02 = net
    
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)    
    adj_conv = nn_idx
    n_heads = params[6]

    net, locals_transform1, coefs2= gap_block(k,n_heads,nn_idx,net,point_cloud,('filter1',params[7]),bn_decay,weight_decay,is_training,scname)

    net = tf_util.conv2d(net, params[9], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='gapnet11'+scname, bn_decay=bn_decay)
    net11 = net



    net = tf_util.conv2d(net, params[10], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='gapnet12'+scname, bn_decay=bn_decay)

    net12= net
    global_expand = tf.reshape(global_pl, [batch_size, 1, 1, -1])
    global_expand = tf.tile(global_expand, [1, num_point, 1, 1])
    global_expand = tf_util.conv2d(global_expand, 16, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=True, is_training=is_training,
                                   scope='global_expand'+scname, bn_decay=bn_decay)
    
    net = tf.concat([
        net01,
        net02,
        net11,
        net12,
        global_expand,
        locals_transform,
        locals_transform1
    ], axis=-1)


    net = tf_util.conv2d(net, params[8], [1, 1], padding='VALID', stride=[1, 1], 
                         activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='agg'+scname, bn_decay=bn_decay)

    net = tf_util.avg_pool2d(net, [num_point, 1], padding='VALID', scope='avgpool'+scname)
    
    net = tf.reshape(net, [batch_size, -1]) 
    net = tf_util.fully_connected(net, params[11], bn=True, is_training=is_training, activation_fn=tf.nn.relu,
                                 scope='fc1'+scname, bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.9, is_training=is_training,
                       scope='dp1'+scname)
    net = tf_util.fully_connected(net, params[12], bn=True, is_training=is_training, activation_fn=tf.nn.relu,
                                  scope='fc2'+scname, bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training,
                          scope='dp2'+scname)

    net = tf_util.fully_connected(net, num_class,activation_fn=None, scope='fc3'+scname)
    net = tf.cond(is_training,lambda:net,lambda:tf.nn.softmax(net))
        
    

    return net, coefs, coefs2, adj_conv

  



def get_loss(pred, label,num_class):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=num_class)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred)
  classify_loss = tf.reduce_mean(loss)
  

  return classify_loss

if __name__=='__main__':
  batch_size = 2
  num_pt = 4
  pos_dim = 4
  nfeat = 8
  ncat = 3
  num_part = ncat*num_pt
  input_feed = np.random.rand(batch_size, num_pt, num_part)
  label_feed = np.random.randint(0,2,size =(batch_size, num_pt))
  print(input_feed, 'prediction')
  print(label_feed,'true label')
  with tf.Graph().as_default():
    true_pl = tf.placeholder(tf.int64, shape=(batch_size, num_pt))
    #pred_pl = tf.placeholder(tf.float64, shape=(batch_size, num_pt,num_part))
    pl = tf.placeholder(tf.float64, shape=(batch_size, num_pt,ncat))
    is_training_pl = tf.placeholder(tf.bool, shape=())
    
    pred_pl = get_model(pl, is_training, ncat,true_pl)
    loss = get_loss(pred_pl, true_pl,norms)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      norm = np.zeros((batch_size,num_pt), dtype=np.float64)
      norm_list = [3.489, 3.4916, 2.342]
      for ib, b in enumerate(label_feed):
        for ip, p in enumerate(b):
          norm[ib][ip] = norm_list[p]
      print(norm,"norms")
      feed_dict = {pred_pl: input_feed, true_pl: label_feed,norms: norm}
      per_instance_pred = sess.run([ per_instance_seg_pred_res], feed_dict=feed_dict)
      #print(loss,"loss")
      print(per_instance_pred,"per_instance_pred")
      #print(res1.shape)
      #rint(res1)

