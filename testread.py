import tensorflow as tf
import pca
import scipy.io as scio
# # 读取文件。
# reader = tf.TFRecordReader()

# # 创建一个队列维护输入列表
# filename_queue = tf.train.string_input_producer(["/home/chunyang/workspace/SPH3D/SPH3D-GCN/data/s3dis_3cm_overlap/Area_1_conferenceRoom_1.tfrecord"])


# # 读取一个样本，也可以用read_up_to读取多个样本
# _,serialized_example = reader.read(filename_queue)

# features = tf.parse_single_example(
#     serialized_example,
#     features={
#         'xyz_raw': tf.FixedLenFeature([], dtype=tf.string),
#         'rel_xyz_raw':tf.FixedLenFeature([], dtype=tf.string),
#         'rgb_raw': tf.FixedLenFeature([], dtype=tf.string),
#         'seg_label':tf.FixedLenFeature([], dtype=tf.string),
#         'inner_label':tf.FixedLenFeature([], dtype=tf.string),
#         'index_label':tf.FixedLenFeature([], dtype=tf.string),
#         'scene_label':tf.FixedLenFeature([], dtype=tf.int64),
#         'scene_idx':tf.FixedLenFeature([], dtype=tf.int64)
#         })

# xyz = tf.decode_raw(features['xyz_raw'], tf.float32)
# rel_xyz = tf.decode_raw(features['rel_xyz_raw'], tf.float32)
# rgb = tf.decode_raw(features['rgb_raw'], tf.float32)
# seg_label = tf.decode_raw(features['seg_label'], tf.float32)
# inner_label = tf.decode_raw(features['inner_label'], tf.float32)
# index_label =  tf.decode_raw(features['index_label'], tf.float32)
# scene_label = tf.cast(features['scene_label'], tf.int64)
# scene_idx = tf.cast(features['scene_idx'], tf.int64)

# xyz = tf.reshape(xyz, [-1, 3])
# rel_xyz = tf.reshape(rel_xyz, [-1, 3])
# rgb = tf.reshape(rgb, [-1, 3])
# seg_label = tf.reshape(seg_label, [-1, 1])
# inner_label = tf.reshape(inner_label, [-1, 1])
# index_label = tf.reshape(index_label, [-1, 1])


# sess = tf.Session()

# # 启动多线程处理输入数据。
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess,coord=coord)

# for i in range(10):
#     xyz, rel_xyz,rgb, seg_label,inner_label,index_label,scene_label,scene_idx = sess.run([xyz, rel_xyz,rgb, seg_label,inner_label,index_label,scene_label,scene_idx])
#     print(seg_label.shape)
BATCH_SIZE = 16
NUM_POINT = 2048
def parse_fn(item):
    features = tf.parse_single_example(
        item,
        features={
            'xyz_raw': tf.FixedLenFeature([], dtype=tf.string),
            'label': tf.FixedLenFeature([], dtype=tf.int64)})

    xyz = tf.decode_raw(features['xyz_raw'], tf.float32)
    label = tf.cast(features['label'], tf.int32)
    xyz = tf.reshape(xyz,[-1,3])

    return xyz, label

def input_fn(filelist, batch_size=16, buffer_size=10000):
    dataset = tf.data.TFRecordDataset(filelist)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset

def placeholder_inputs(batch_size, num_point):
    xyz_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    label_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return xyz_pl, label_pl

trainlist = "/home/chunyang/workspace/SPH3D/SPH3D-GCN/data/modelnet40/data_test0.tfrecord"
trainset = input_fn(trainlist, BATCH_SIZE, 10000)
train_iterator = trainset.make_initializable_iterator()
next_train_element = train_iterator.get_next()
with tf.Session() as sess:
    sess.run(train_iterator.initializer)
    batch_xyz, batch_label = sess.run(next_train_element)
    r = sess.run(pca.tf_pca(batch_xyz))
print(r)
# scio.savemat('data_test0_tfrecord0_15.mat',{'data':batch_xyz,'label':batch_label,'pcadata':r})