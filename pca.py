import tensorflow as tf

# def tf_pca(x):
#     '''
#         Compute PCA on the bottom two dimensions of x,
#         eg assuming dims = [..., observations, features]
#     '''
#     # Center
#     x -= tf.reduce_mean(x, -2, keepdims=True)

#     # Currently, the GPU implementation of SVD is awful.
#     # It is slower than moving data back to CPU to SVD there
#     # https://github.com/tensorflow/tensorflow/issues/13222

#     with tf.device('/cpu:0'):
#         ss, us, vs = tf.svd(x, full_matrices=False, compute_uv=True)

#     ss = tf.expand_dims(ss, -2)    
#     projected_data = us * ss

#     # Selection of sign of axes is arbitrary.
#     # This replicates sklearn's PCA by duplicating flip_svd
#     # https://github.com/scikit-learn/scikit-learn/blob/7ee8f97e94044e28d4ba5c0299e5544b4331fd22/sklearn/utils/extmath.py#L499
#     r = projected_data
#     abs_r = tf.abs(r)
#     m = tf.equal(abs_r, tf.reduce_max(abs_r, axis=-2, keepdims=True))
#     signs = tf.sign(tf.reduce_sum(r * tf.cast(m, r.dtype), axis=-2, keepdims=True))
#     result = r * signs

#     return result

def tf_pca(x):
    '''
        Compute PCA on the bottom two dimensions of x,
        eg assuming dims = [..., observations, features]
    '''
    # Center
    mx = tf.reduce_mean(x, -2, keepdims=True)
    x = x -mx
    covx = tf.matmul(x,x,transpose_a=True)
    # with tf.device('/cpu:0'):
    ss, us, vs = tf.svd(covx, full_matrices=False, compute_uv=True)

    # sig = tf.matmul(mx,us)

    result = tf.matmul(x,us)
    return result