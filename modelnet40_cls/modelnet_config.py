import numpy as np


dataDir = "/home/xiaom/workspace/data/modelnet40"

num_input = 2048
num_cls = 40

mlp = 32

# radius = [0.1, 0.2, 0.4]
# nn_uplimit = [64, 64, 64]
# channels = [[64, 64], [64, 128], [128, 128]]
# multiplier = [[2, 1], [1, 2], [1, 1]]

# num_sample = [num_input//4**(i+1) for i in range(10) if (num_input//4**(i+1))>100] #FPS下采样后的点数
# num_sample = [1024,512]
# radius = [0.3, 0.5] #球查询的半径
# nn_uplimit = [64, 128] #球查询的邻域上限
# channels = [[64, 64], [64, 128]] #输出channels
# multiplier = [[2, 1], [1, 2]] #分离卷积深度特征图增益

num_sample = [1024]
radius = [0.3] #球查询的半径
nn_uplimit = [64] #球查询的邻域上限
channels = [[64, 128]] #输出channels
multiplier = [[5, 5]] #分离卷积深度特征图增益

assert(len(num_sample)==len(radius))
assert(len(num_sample)==len(nn_uplimit))
assert(len(num_sample)==len(channels))
assert(len(num_sample)==len(multiplier))

# =====================for final layer convolution=====================
global_channels = 512
global_multiplier = 2
# =====================================================================

weight_decay = 0

kernel=[8,4,2]
binSize = np.prod(kernel)+1

normalize = True
pool_method = 'max'
nnsearch = 'sphere'
sample = 'FPS' #{'FPS','IDS','random'}

use_raw = True
with_bn = True
with_bias = False
# with_bias = True

ORIDATA = False
TRAIN_RATATION = False
EVAL_RATATION = True

LOADMODEL = False

def IF_EVAL(global_step):
    if(global_step%4 == 0):
        return True
    return False

print("from config > num_sample:",num_sample)