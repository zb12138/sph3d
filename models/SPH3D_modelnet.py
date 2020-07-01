import tensorflow as tf
import sys
import os
ROOTDIR = "/home/xiaom/workspace/sph3dR/"
sys.path.append(ROOTDIR)
sys.path.append(os.path.join(ROOTDIR, 'utils'))
import utils.sph3gcn_util2 as s3g_util
import pca

def normalize_xyz(points):
    points -= tf.reduce_mean(points,axis=1,keepdims=True)
    scale = tf.reduce_max(tf.reduce_sum(tf.square(points),axis=-1,keepdims=True),axis=1,keepdims=True)
    scale = tf.sqrt(scale,name='normalize')
    points /= scale

    return points


def _separable_conv3d_block(net, list_channels, bin_size, nn_index, nn_count, filt_idx,
                            name, depth_multiplier=None, weight_decay=None, reuse=None,
                            with_bn=True, with_bias=True, is_training=None):
    for l, num_out_channels in enumerate(list_channels):
        scope = name + '_' + str(l+1) # number from 1, not 0
        net = s3g_util.separable_conv3d(net, num_out_channels, bin_size,
                                        depth_multiplier[l], scope, nn_index,
                                        nn_count, filt_idx, weight_decay=weight_decay,
                                        with_bn=with_bn, with_bias=with_bias,
                                        reuse=reuse, is_training=is_training)
    return net

def calcAngle(data,intra_dst,indices,intra_cnt,out_channels=1):
    """ 
    data (B,N,3) channels to invariance transform
    indices (B,N,nn_uplimit)the idx of neighbors e.g (32,2048,64)
    intra_dst (B,N,nn_uplimit)the distance of neighbors
    intra_cnt (B,N) the num of neighbors
    """
    assert(len(list(data.shape))==3)
    B = data.shape[0]
    N = data.shape[1]
    out_tenosr = tf.reduce_sum(intra_dst,axis=2)/ tf.cast(intra_cnt,tf.float32)
    out_tenosr = tf.expand_dims(out_tenosr,axis=2)
    return out_tenosr

def get_model(points, is_training, config=None):
    """ Classification Network, input is BxNx3, output Bx40 """
    batch_size = points.get_shape()[0].value
    num_point = points.get_shape()[1].value
    end_points = {}

    assert(num_point==config.num_input)
    # points = pca.tf_pca(points) #'MatMul_2:0'
    if config.normalize:
        points = normalize_xyz(points)

    xyz = points
    xyzinr =tf.norm(xyz,axis=2,keepdims=True,name='xyzinr')#'xyzinr/Sqrt:0'
    # xyzinr = xyz
    query = tf.reduce_mean(xyz, axis=1, keepdims=True)  # the global viewing point
    # query = tf.ones([xyz.shape[0],1,xyz.shape[2]])/1000
    reuse = None
    # net = s3g_util.pointwise_conv3d(xyzinr, config.mlp, 'mlp1',
    #                                 weight_decay=config.weight_decay,
    #                                 with_bn=config.with_bn, with_bias=config.with_bias,
    #                                 reuse=reuse, is_training=is_training)
    # binnum = tf.Variable(config.kernel[0]*config.kernel[1]*config.kernel[2],trainable=False,dtype=tf.float32)
    global_feat = []
    # convRXyz = xyzinr
    for l in range(len(config.radius)):

            # net = tf.ones([xyz.shape[0],xyz.shape[1],config.mlp])
        # the neighbor information is the same within xyz_pose_1 and xyz_pose_2.
        # Therefore, we compute it with xyz_pose_1, and apply it to xyz_pose_2 as well
        #'BuildSphereNeighbor:0'
        intra_idx, intra_cnt, \
        intra_dst, indices = s3g_util.build_graph(xyz, config.radius[l], config.nn_uplimit[l],
                                                  config.num_sample[l], sample_method=config.sample)
        # intra_idx 每个点邻域点(<=64)的idx (32,2048,64)
        # intra_cnt 每个点邻域点的个数
        # intra_dst 每个点邻域点距离中心的距离
        # indices FPS后点的索引(32,512,2) 2? or 1
        
        # 为每个点的邻域(64,radius)找到bin (32,2048,64)
        # bins 'SphericalKernel:1'
        filt_idx,rotateXyz = s3g_util.spherical_kernel(xyz, xyz, intra_idx, intra_cnt,
                                             intra_dst, config.radius[l],
                                             kernel=config.kernel)#(32,2048,64) (32,512,64)
        if l==0:
            net = tf.concat([xyzinr, tf.cast(filt_idx,tf.float32)], axis=-1)
        if config.use_raw and l>0: #会循环加入
            with tf.variable_scope("conv2_"+str(l+1)):
                conv2_weights = tf.get_variable(
                    "weight", [1, rotateXyz.shape[2], rotateXyz.shape[3], rotateXyz.shape[3]],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv2_biases = tf.get_variable("bias", [rotateXyz.shape[3]], initializer=tf.constant_initializer(0.0))
                conv2 = tf.nn.conv2d(rotateXyz, conv2_weights, strides=[1, 1, rotateXyz.shape[2], 1], padding='VALID')
                relu2 =tf.squeeze(tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases)))
            net = tf.concat([net,relu2,tf.cast(filt_idx,tf.float32)], axis=-1)
 
        # 把每个点进行球卷积（2次，以multiplier参数的分离卷积实现）
        # (32,N,input_channels) -> (32,N,output_channels) net(32,2048,64)->net(32,2048,128)
        # net = _separable_conv3d_block(net, config.channels[l], config.binSize, intra_idx, intra_cnt,
        #                               filt_idx, 'conv'+str(l+1), config.multiplier[l], reuse=reuse,
        #                               weight_decay=config.weight_decay, with_bn=config.with_bn,
        #                               with_bias=config.with_bias, is_training=is_training)

        weight = s3g_util.pointwise_conv3d(net, net.shape[2], 'conv'+str(l+1),
                                        weight_decay=config.weight_decay,
                                        with_bn=config.with_bn, with_bias=config.with_bias,
                                        reuse=reuse, is_training=is_training)
        net = weight*net
        net = s3g_util.pointwise_conv3d(net,config.channels[l], 'conv2'+str(l+1),
                                weight_decay=config.weight_decay,
                                with_bn=config.with_bn, with_bias=config.with_bias,
                                reuse=reuse, is_training=is_training)
                            
        if config.num_sample[l]>1:
            # ==================================gather_nd====================================
            # xyzinr = tf.gather_nd(xyzinr, indices)
            xyz = tf.gather_nd(xyz, indices)
            inter_idx = tf.gather_nd(intra_idx, indices) # indices仅仅为了粗化
            inter_cnt = tf.gather_nd(intra_cnt, indices) # intar_idx/inter_idx 同时作为层内和层间邻域
            # inter_dst = tf.gather_nd(intra_dst, indices)
        # =====================================END=======================================
        # 从被FPS选中的点的邻域中选择最大/均值作为该点输出(32,2048,128)->(32,1024,128)
        net = s3g_util.pool3d(net, inter_idx, inter_cnt,method=config.pool_method, scope='pool'+str(l+1))
        
        net = tf.nn.elu(net)

        global_maxpool = tf.reduce_max(net, axis=1, keepdims=True) # 排列不变性(32,1,128)
        global_feat.append(global_maxpool) # 将池化后的特征均值作为本层的全局特征(32,1,channels)
    # =============================global feature extraction in the final layer=============================
    global_radius = 100.0 # global_radius(>=2.0) should connect all points to each point in the cloud(no 64 limit)
    nn_idx, nn_cnt, nn_dst = s3g_util.build_global_graph(xyz, query, global_radius)# 'BuildSphereNeighbor_2:x'
    filt_idx,rotateXyz = s3g_util.spherical_kernel(xyz, query, nn_idx, nn_cnt, nn_dst,# SphericalKernel:1
                                        global_radius, kernel=[8,8,4]) #(32,1,128) nn_uplimit is all the points 128


    #net(32,2048,128)
    # convRXyz = s3g_util.conv2d(rotateXyz,num_output_channels=128,kernel_size=[1,1],scope='convRG',
    #                                         padding="VALID",bn=False)#(32,1,2048,128)
    # convRXyz = tf.nn.max_pool(convRXyz,[1,1,2048,1],strides=[1,1,1,1],padding='VALID',name='PconvRG')
    # convRXyz = tf.squeeze(convRXyz,name='FconvRG',axis=1)

    # net = tf.concat([net, convRXyz], axis=-1)#(32,2048,128+32)

    net = s3g_util.separable_conv3d(net, config.global_channels, 8*8*4+1, config.global_multiplier,
                                    'global_conv', nn_idx, nn_cnt, filt_idx, reuse=reuse,
                                    weight_decay=config.weight_decay, with_bn=config.with_bn,
                                    with_bias=config.with_bias, is_training=is_training) #(32,1,512) 
    global_feat.append(net)
    # global_feat.append(convRXyz)

    net = tf.concat(global_feat,axis=2) # (32,1,128) (32,1,512) -> (32,1,640)
    # =====================================================================================================
    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = s3g_util.fully_connected(net, 512, scope='fc1', weight_decay=config.weight_decay,
                                   with_bn=config.with_bn, with_bias=config.with_bias, is_training=is_training)
    net = tf.layers.dropout(net, 0.7, training=is_training, name='fc1_dp')
    net = s3g_util.fully_connected(net, 256, scope='fc2', weight_decay=config.weight_decay,
                                   with_bn=config.with_bn, with_bias=config.with_bias, is_training=is_training)
    net = tf.layers.dropout(net, 0.7, training=is_training, name='fc2_dp')
    net = s3g_util.fully_connected(net, config.num_cls, scope='logits', with_bn=False, with_bias=config.with_bias,
                                   activation_fn=None, is_training=is_training)
    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss