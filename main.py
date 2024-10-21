import os
import time
import cv2
import scipy.io
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras import layers, Model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Conv2DTranspose
import tf_slim as slim
import numpy as np
from discriminator import build_discriminator
import scipy.stats as st
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="pre-trained",
                    help="path to folder containing the model/pre-trained")
parser.add_argument("--data_syn_dir", default="./root_training_synthetic_data/",
                    help="path to synthetic dataset")
parser.add_argument("--data_real_dir", default="./root_training_real_data/", help="path to real dataset")
parser.add_argument("--save_model_freq", default=5,
                    type=int, help="frequency to save model")
parser.add_argument("--is_hyper", default=1, type=int,
                    help="use hypercolumn or not")
parser.add_argument("--is_training", default=0, help="training or testing")
parser.add_argument("--continue_training", action="store_true",
                    help="search for checkpoint in the subfolder specified by `task` argument")
ARGS = parser.parse_args()

task = ARGS.task
is_training = ARGS.is_training == 1
continue_training = ARGS.continue_training
hyper = ARGS.is_hyper == 1

tf.disable_eager_execution()

# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
if is_training:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
EPS = 1e-12
channel = 64  # number of feature channels to build the model, set to 64

train_syn_root = [ARGS.data_syn_dir]
train_real_root = [ARGS.data_real_dir]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def build_net(ntype, nin, nwb=None, name=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name)+nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def lrelu(x):
    return tf.maximum(x*0.2, x)


def relu(x):
    return tf.maximum(0.0, x)

# 权重重置器（归一化）
def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2], shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

# 输入数据归一化
def nm(x):
    w0 = tf.Variable(1.0, name='w0')
    w1 = tf.Variable(0.0, name='w1')
    return w0*x+w1*slim.batch_norm(x)

# 预训练数据储存
vgg_path = scipy.io.loadmat('./VGG_Model/imagenet-vgg-verydeep-19.mat')
print("[i] Loaded pre-trained vgg19 parameters")
# build VGG19 to load pre-trained parameters


def build_vgg19(input, reuse=False):
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net = {}
        vgg_layers = vgg_path['layers'][0]
        net['input'] = input - \
            np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
        net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(
            vgg_layers, 0), name='vgg_conv1_1')
        net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(
            vgg_layers, 2), name='vgg_conv1_2')
        net['pool1'] = build_net('pool', net['conv1_2'])
        net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(
            vgg_layers, 5), name='vgg_conv2_1')
        net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(
            vgg_layers, 7), name='vgg_conv2_2')
        net['pool2'] = build_net('pool', net['conv2_2'])
        net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(
            vgg_layers, 10), name='vgg_conv3_1')
        net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(
            vgg_layers, 12), name='vgg_conv3_2')
        net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(
            vgg_layers, 14), name='vgg_conv3_3')
        net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(
            vgg_layers, 16), name='vgg_conv3_4')
        net['pool3'] = build_net('pool', net['conv3_4'])
        net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(
            vgg_layers, 19), name='vgg_conv4_1')
        net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(
            vgg_layers, 21), name='vgg_conv4_2')
        net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(
            vgg_layers, 23), name='vgg_conv4_3')
        net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(
            vgg_layers, 25), name='vgg_conv4_4')
        net['pool4'] = build_net('pool', net['conv4_4'])
        net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(
            vgg_layers, 28), name='vgg_conv5_1')
        net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(
            vgg_layers, 30), name='vgg_conv5_2')
        return net


class CBAM_Module(tf.keras.Model):
    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.channels=channels
        self.reduction=reduction
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc1 = layers.Conv2D(channels // reduction, kernel_size=1, strides=1, padding='same')
        self.relu = layers.ReLU()
        self.fc2 = layers.Conv2D(channels, kernel_size=1,strides=1, padding='same')
        self.sigmoid_channel = layers.Activation('sigmoid')

        self.conv_after_concat = layers.Conv2D(1, kernel_size=3, padding='same') 
        self.sigmoid_spatial = layers.Activation('sigmoid') 
        self.conv_agg_avg = layers.Conv2D(1, kernel_size=3, padding='same')
        self.conv_agg_max = layers.Conv2D(1, kernel_size=3, padding='same') 
    def multi_scale_pooling(self, feature_map):
        sizes = [1, 4, 8, 16]
        avg_pools = []
        max_pools = []

        for size in sizes:
        # 获取尺寸
            shape = tf.shape(feature_map)
            height, width = shape[1], shape[2]

        # 缩小特征图
            resized_feature_map = tf.image.resize(
                feature_map, 
                (height // size, width // size)
            )

        # 平均池化和最大池化（保留空间维度）
            pool_avg = tf.reduce_mean(resized_feature_map, axis=-1, keepdims=True)
            pool_max = tf.reduce_max(resized_feature_map, axis=-1, keepdims=True)

        # 将缩小的结果恢复到原始特征图大小
            pool_avg = tf.image.resize(
                pool_avg, 
                (height, width), 
                method='bilinear'
            )
            pool_max = tf.image.resize(
                pool_max, 
                (height, width), 
                method='bilinear'
            )
            avg_pools.append(pool_avg)
            max_pools.append(pool_max)

        # 拼接池化结果
        avg_pools = tf.concat(avg_pools, axis=-1)
        max_pools = tf.concat(max_pools, axis=-1)
        return avg_pools, max_pools
    
    def call(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        avg=tf.reshape(avg,(-1,1,1,self.channels))
        mx = self.max_pool(x)
        mx=tf.reshape(mx,(-1,1,1,self.channels))
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        
        x = module_input * x
        avg_pools, max_pools=self.multi_scale_pooling(x)
        avg_pooled = self.conv_agg_avg(avg_pools)
        max_pooled = self.conv_agg_max(max_pools) 
        spatial_attention = tf.concat([avg_pooled, max_pooled], axis=-1) 
        spatial_attention = self.conv_after_concat(spatial_attention) 
        spatial_attention = self.sigmoid_spatial(spatial_attention) 
        x = module_input * spatial_attention 
        return x





class ConvLayer(tf.keras.layers.Layer):
     def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, dilation=1, norm=None, act=None): 
        super(ConvLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D( 
            filters=channels_out, kernel_size=kernel_size, strides=stride, padding='same', dilation_rate=dilation) 
        self.norm = None
        if norm is not None: 
            self.norm = tf.keras.layers.BatchNormalization()
        self.act = None 
        if act is not None:
            self.act = tf.keras.activations.relu

     def call(self, x):
        x = self.conv(x) 
        if self.norm is not None:
            x = self.norm(x) 
        if self.act is not None: 
            x = self.act(x)
        return x 



class SELayer(tf.keras.layers.Layer):
    def __init__(self, channels, reduction):
        super(SELayer, self).__init__()
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels , activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')

    def call(self, x):
        se = self.global_avg_pool(x)
        se=tf.expand_dims(tf.expand_dims(se,1),1)
        se = self.fc1(se)
        se = self.fc2(se)
        return x *se






class Original_YTMT(tf.keras.Model):
    def __init__(self, channels, dilation=1, norm=None, reduction=None):
        super(Original_YTMT, self).__init__()
        self.relu = tf.keras.activations.relu

        self.conv_r = ConvLayer(channels, channels,norm=None,act=None)
        self.conv_t = ConvLayer(channels, channels, norm=None,act=None)

        self.conv_fus_r = ConvLayer(channels, channels, norm=None,act=self.relu)
        self.conv_fus_t = ConvLayer(channels, channels,norm=None, act=self.relu)
        
        self.block_rn = tf.keras.Sequential([ ConvLayer(channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=None ,act=self.relu), ConvLayer(channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=None, act=self.relu) ])
        self.block_tn = tf.keras.Sequential([ ConvLayer(channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=None, act=self.relu), ConvLayer(channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=None, act=self.relu) ]) 
        
        self.CBAM_rn = CBAM_Module(channels,reduction=1)
        self.CBAM_tn = CBAM_Module(channels,reduction=1)

        self.se_r = SELayer(channels, reduction)
        self.se_t = SELayer(channels, reduction)

    def call(self, list_rt):
        in_r, in_t = list_rt

        r = self.conv_r(in_r)
        t = self.conv_t(in_t)

        r_p, r_n = self.relu(r), r - self.relu(r)
        t_p, t_n = self.relu(t), t - self.relu(t)

        r_n =self.block_rn(r_n)
        t_n =self.block_rn(t_n)

        r_n =self.CBAM_rn(r_n)
        t_n =self.CBAM_tn(t_n)

        out_r = tf.concat([r_p, t_n], axis=-1)
        out_t = tf.concat([t_p, r_n], axis=-1)
        
        out_r1 = tf.concat([out_r, in_r], axis=-1)
        out_t1 = tf.concat([out_t, in_t], axis=-1)
        

        out_r = self.se_r(self.conv_fus_r(out_r1))
        out_t = self.se_t(self.conv_fus_t(out_t1))

        return  out_r,out_t


def build_network(input_tensor, channel, num_blocks=1):
    # 初始设置（高配置服务站可调制num_blocks=6）
    net = input_tensor
    R = net  # 假设 R 初始为输入张量
    T = tf.identity(net)  # T 初始为输入张量的副本

    # 定义 Original_YTMT 实例
    ytmt_block = Original_YTMT(channel, norm=None, dilation=1)

    # 串联六个 Original_YTMT 块
    for i in range(num_blocks):
        R, T = ytmt_block([R, T])

    return R, T






class ResidualBlock(tf.keras.Model):
    def __init__(self, channels, dilation=1, reduction=1, norm=None,res_scale=0.1, att_flag='se'):
        super(ResidualBlock, self).__init__()
        self.relu = tf.keras.activations.relu
        self.conv1 = ConvLayer(channels, channels, stride=1, dilation=dilation, norm=None, act=self.relu)
        self.conv2 = ConvLayer(channels, channels, stride=1, dilation=dilation, norm=None, act=None)
        self.att_layer = None
        self.res_scale = 0.1
        if reduction is not None:
            if att_flag == 'se':
                self.att_layer = SELayer(channels, reduction)
            # Add CBAM_Module conversion here if needed
            elif att_flag == 'cbam':
                self.att_layer = CBAM_Module(channels, reduction)
    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.att_layer:
            out = self.att_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out


def make_network(input_tensor, channel, Residualnum_blocks=2):
    # 初始设置（高配置服务站可调制Residualnum_blocks=3）
    net = input_tensor

    # 定义 Original_YTMT 实例
    Residual_block = ResidualBlock(channel, norm=None, dilation=1)

    # 串联六个 Original_YTMT 块
    for i in range(Residualnum_blocks):
        net = Residual_block(net)

    return net


class PyramidPooling(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=64,norm=None):
        super(PyramidPooling, self).__init__()
        self.stages = [self._make_stage(in_channels, scale, ct_channels) for scale in scales]
        self.bottleneck = tf.keras.layers.Conv2D(out_channels, kernel_size=1, strides=1, padding='same')
        self.relu = tf.keras.activations.relu

    def _make_stage(self, in_channels, scale, ct_channels):
        return tf.keras.Sequential([
            tf.keras.layers.AvgPool2D(pool_size=(scale, scale)),
            tf.keras.layers.Conv2D( ct_channels, kernel_size=1, use_bias=False, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ])
        
    def call(self, inputs, ):
        h, w = tf.shape(inputs)[1], tf.shape(inputs)[2]
        priors = [inputs]
        
        for stage in self.stages:
            x = stage(inputs)
            x = tf.image.resize(x, size=(h, w), method='nearest')
            priors.append(x)
        
        priors = tf.concat(priors, axis=-1)  # Concatenate along the channel dimension
        out = self.bottleneck(priors)
        return self.relu(out)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_feats):
        super(Decoder, self).__init__() 
        self.relu = tf.keras.activations.relu
        self.layers = [ 
            # tf.keras.layers.Conv2DTranspose(  n_feats, kernel_size=4, strides=(2,2), activation=self.relu), 
            # ConvLayer( n_feats, n_feats, kernel_size=3, stride=1, norm=None, act=self.relu),
            PyramidPooling(n_feats, n_feats, scales=(4, 8, 16, 32), ct_channels=n_feats // 4),
            ConvLayer(n_feats, 3, kernel_size=1, stride=1, norm=None, act=self.relu) 
        ]
    def call(self, inputs):
        x = inputs 
        for layer in self.layers:
             x = layer(x) 
        return x



def build(input,channel=256):
    if hyper:
        print("[i] Hypercolumn ON, building hypercolumn features ... ")
        vgg19_features = build_vgg19(input[:, :, :, 0:3]*255.0)
        for layer_id in range(1, 6):
            vgg19_f = vgg19_features["conv%d_2" % layer_id]
            input = tf.concat([tf.image.resize_bilinear(
                vgg19_f, (tf.shape(input)[1], tf.shape(input)[2]))/255.0, input], axis=3)
    else:
        vgg19_features = build_vgg19(input[:, :, :, 0:3]*255.0)
        for layer_id in range(1, 6):
            vgg19_f = vgg19_features["conv%d_2" % layer_id]
            input = tf.concat([tf.image.resize_bilinear(tf.zeros_like(
                vgg19_f), (tf.shape(input)[1], tf.shape(input)[2]))/255.0, input], axis=3)   
    # (依照环境硬件配置，自行控制多尺度空间注意力机制)
    net = slim.conv2d(input, channel, [1, 1], rate=1, activation_fn=lrelu,
                      normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv0')
    # net=CBAM_Module(channel,reduction=1)(net)
    net = slim.conv2d(net, channel, [3, 3], rate=4, activation_fn=lrelu,
                      normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv1')
    # # net=CBAM_Module(channel,reduction=1)(net)
    # net = slim.conv2d(net, channel, [3, 3], rate=2, activation_fn=lrelu,
    #                   normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv2')
    # net=CBAM_Module(channel,reduction=1)(net)
    # net = slim.conv2d(net, channel, [3, 3], rate=4, activation_fn=lrelu,
    #                   normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv3')
    # net=CBAM_Module(channel,reduction=1)(net)
    # net = slim.conv2d(net, channel, [3, 3], rate=8, activation_fn=lrelu,
    #                   normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv4')
    # net=CBAM_Module(channel,reduction=1)(net)                  
    # net = slim.conv2d(net, channel, [3, 3], rate=4, activation_fn=lrelu,
                    #   normalizer_fn=None, weights_initializer=None, scope='g_conv5')
    # net=CBAM_Module(channel,reduction=1)(net)                  
    net = slim.conv2d(net, channel, [3, 3], rate=4, activation_fn=lrelu,
                      normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv6')
    net=CBAM_Module(channel,reduction=1)(net)
    net = slim.conv2d(net, channel, [3, 3], rate=4, activation_fn=lrelu,
                      normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv7')
    # net=CBAM_Module(channel,reduction=1)(net)
    net = slim.conv2d(net, channel, [3, 3], rate=2, activation_fn=lrelu,
                      normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv9')
    R,T =build_network(net,channel)
    R=make_network(R,channel)
    T=make_network(T,channel)
    R = slim.conv2d(R, 3, [1, 1], rate=1,
                      activation_fn=None, scope='g_conv_last')
    T = slim.conv2d(T, 3, [1, 1], rate=1,
                      activation_fn=None, scope='g_conv_last2')
    return T , R
   

def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = kernel/kernel.max()
    return kernel



g_mask = gkern(560, 3)
g_mask = np.dstack((g_mask, g_mask, g_mask))


def syn_data(t, r, sigma):
    # 伽马矫正
    t = np.power(t, 2.2)
    r = np.power(r, 2.2)

    sz = int(2*np.ceil(2*sigma)+1)
    r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
    blend = r_blur+t

    att = 1.08+np.random.random()/10.0

    for i in range(3):
        maski = blend[:, :, i] > 1
        mean_i = max(1., np.sum(blend[:, :, i]*maski)/(maski.sum()+1e-6))
        r_blur[:, :, i] = r_blur[:, :, i]-(mean_i-1)*att
    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0

    h, w = r_blur.shape[0:2]
    neww = np.random.randint(0, 560-w-10)
    newh = np.random.randint(0, 560-h-10)
    alpha1 = g_mask[newh:newh+h, neww:neww+w, :]
    alpha2 = 1-np.random.random()/5.0
    r_blur_mask = np.multiply(r_blur, alpha1)
    blend = r_blur_mask+t*alpha2

    t = np.power(t, 1/2.2)
    r_blur_mask = np.power(r_blur_mask, 1/2.2)
    blend = np.power(blend, 1/2.2)
    blend[blend >= 1] = 1
    blend[blend <= 0] = 0

    return t, r_blur_mask, blend


def compute_loss(input, output):
    return tf.reduce_mean(tf.abs(input-output))


def compute_percep_loss(input, output, reuse=False):
    vgg_real = build_vgg19(output*255.0, reuse=reuse)
    vgg_fake = build_vgg19(input*255.0, reuse=True)
    p0 = compute_loss(vgg_real['input'], vgg_fake['input'])
    p1 = compute_loss(vgg_real['conv1_2'], vgg_fake['conv1_2'])
    p2 = compute_loss(vgg_real['conv2_2'], vgg_fake['conv2_2'])
    p3 = compute_loss(vgg_real['conv3_2'], vgg_fake['conv3_2'])
    p4 = compute_loss(vgg_real['conv4_2'], vgg_fake['conv4_2'])
    p5 = compute_loss(vgg_real['conv5_2'], vgg_fake['conv5_2'])
    return p0+p1+p2+p3+p4+p5


def compute_mutex_loss(img1, img2, level=3):
    gradx_loss = []
    grady_loss = []

    for l in range(level):
        gradx1, grady1 = compute_gradient(img1)
        gradx2, grady2 = compute_gradient(img2)
        alphax = 2.0*tf.reduce_mean(tf.abs(gradx1)) / \
            tf.reduce_mean(tf.abs(gradx2))
        alphay = 2.0*tf.reduce_mean(tf.abs(grady1)) / \
            tf.reduce_mean(tf.abs(grady2))

        gradx1_s = gradx1
        grady1_s = grady1
        gradx2_s = gradx2*alphax
        grady2_s = grady2*alphay

        gradx_loss.append(tf.reduce_mean(tf.multiply(
            tf.abs(gradx1_s), tf.abs(gradx2_s))))
        gradx_loss.append(tf.reduce_mean(tf.multiply(
            tf.abs(grady1_s), tf.abs(grady2_s))))

        img1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        img2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    return gradx_loss, grady_loss


def compute_gradient(img):
    gradx = img[:, 1:, :, :]-img[:, :-1, :, :]
    grady = img[:, :, 1:, :]-img[:, :, :-1, :]
    return gradx, grady


# set up the model and define the graph
with tf.variable_scope(tf.get_variable_scope()):
    input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    target = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    reflection = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    issyn = tf.placeholder(tf.bool, shape=[])

    # build the model
    transmission_layer, reflection_layer = build(input)

    # piexl and gradient Loss
    X,Y=compute_gradient(transmission_layer)
    X1,Y1=compute_gradient(target)
    x,y=compute_gradient(reflection_layer)
    x1,y1=compute_gradient(reflection)
    loss_t=compute_loss(transmission_layer, target)+[compute_loss(X,X1)+compute_loss(Y,Y1)]
    loss_r = tf.where(issyn, compute_loss(reflection_layer, reflection)+compute_loss(x,x1)+compute_loss(y,y1), 0.)
    loss_piexls = tf.where(issyn, loss_t+loss_r, loss_t)
    
    # Perceptual Loss
    loss_percep_t = compute_percep_loss(transmission_layer, target)
    loss_percep_r = tf.where(issyn, compute_percep_loss(
        reflection_layer, reflection, reuse=True), 0.)
    loss_percep = tf.where(issyn, loss_percep_t+loss_percep_r, loss_percep_t)

    # Adversarial Loss
    with tf.variable_scope("discriminator"):
        predict_real, pred_real_dict = build_discriminator(input, target)
    with tf.variable_scope("discriminator", reuse=True):
        predict_fake, pred_fake_dict = build_discriminator(
            input, transmission_layer)

    d_loss = (tf.reduce_mean(-(tf.log(predict_real + EPS) +
              tf.log(1 - predict_fake + EPS)))) * 0.5
    g_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))

    # mutex loss
    loss_gradx, loss_grady = compute_mutex_loss(
        transmission_layer, reflection_layer, level=3)
    loss_gradxy = tf.reduce_sum(sum(loss_gradx)/3.) + \
        tf.reduce_sum(sum(loss_grady)/3.)
    loss_grad = tf.where(issyn, loss_gradxy/2.0, 0)

    loss = loss_piexls+loss_grad+loss_percep

train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'discriminator' in var.name]
g_vars = [var for var in train_vars if 'g_' in var.name]
g_opt = tf.train.AdamOptimizer(learning_rate=0.0015).minimize(
    loss+g_loss, var_list=g_vars)  # optimizer for the generator
d_opt = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(
    d_loss, var_list=d_vars)  # optimizer for the discriminator

for var in tf.trainable_variables():
    print("Listing trainable variables ... ")
    print(var)

saver = tf.train.Saver(max_to_keep=10)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(task)
print("[i] contain checkpoint: ", ckpt)
if ckpt and continue_training:
    saver_restore = tf.train.Saver([var for var in tf.trainable_variables()])
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restore.restore(sess, ckpt.model_checkpoint_path)

else:
    saver_restore = tf.train.Saver(
        [var for var in tf.trainable_variables() if 'discriminator' not in var.name])
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restore.restore(sess, ckpt.model_checkpoint_path)
    print("No checkpoint found or continue_training is False.")

maxepoch = 1000
k_sz = np.linspace(1, 5, 80)  
if is_training:
    def prepare_data(train_path):
        input_names = []
        image1 = []
        image2 = []
        for dirname in train_path:
            train_t_gt = dirname + "transmission_layer/"
            train_r_gt = dirname + "reflection_layer/"
            train_b = dirname + "blended/"
            for root, _, fnames in sorted(os.walk(train_r_gt)):
                for fname in fnames:
                    if is_image_file(fname):
                        path_input = os.path.join(train_b, fname)
                        path_output1 = os.path.join(train_t_gt, fname)
                        path_output2 = os.path.join(train_r_gt, fname)
                        input_names.append(path_input)
                        image1.append(path_output1)
                        image2.append(path_output2)
        return input_names, image1, image2

    
    _, syn_image1_list, syn_image2_list = prepare_data(train_syn_root)
    input_real_names, output_real_names1, output_real_names2 = prepare_data(
        train_real_root)  
    print("[i] Total %d training images, first path of real image is %s." %
          (len(syn_image1_list)+len(output_real_names1), input_real_names[0]))

    num_train = len(output_real_names1)
    all_l = np.zeros(num_train, dtype=float)
    all_percep = np.zeros(num_train, dtype=float)
    all_grad = np.zeros(num_train, dtype=float)
    all_g = np.zeros(num_train, dtype=float)
    for epoch in range(1, maxepoch):
        input_images = [None]*num_train
        output_images_t = [None]*num_train
        output_images_r = [None]*num_train

        if os.path.isdir("%s/%04d" % (task, epoch)):
            continue
        cnt = 0
        for id in np.random.permutation(num_train):
            st = time.time()
            if input_images[id] is None:
                magic = np.random.random()
                if magic < 0.7 and id< len(syn_image1_list):  
                    is_syn = True
                    syn_image1 = cv2.imread(syn_image1_list[id], -1)
                    neww = np.random.randint(256, 480)
                    newh = round(
                            (neww/syn_image1.shape[1])*syn_image1.shape[0])
                    output_image_t = cv2.resize(np.float32(
                            syn_image1), (neww, newh), cv2.INTER_CUBIC)/255.0
                    imread_img = cv2.imread(syn_image2_list[id], -1)
                    output_image_r = cv2.resize(np.float32(imread_img), (neww, neww), cv2.INTER_CUBIC)/255.0
                    file = os.path.splitext(
                            os.path.basename(syn_image1_list[id]))[0]
                    sigma = k_sz[np.random.randint(0, len(k_sz))]
                    if np.mean(output_image_t)*1/2 > np.mean(output_image_r):
                            continue
                    _, output_image_r, input_image = syn_data(
                            output_image_t, output_image_r, sigma)
                else:  
                    is_syn = False
                    _id = id
                    inputimg = cv2.imread(input_real_names[_id], -1)
                    file = os.path.splitext(
                        os.path.basename(input_real_names[_id]))[0]
                    neww = np.random.randint(256, 480)
                    newh = round((neww/inputimg.shape[1])*inputimg.shape[0])
                    input_image = cv2.resize(np.float32(
                        inputimg), (neww, newh), cv2.INTER_CUBIC)/255.0
                    output_image_t = cv2.resize(np.float32(cv2.imread(
                        output_real_names1[_id], -1)), (neww, newh), cv2.INTER_CUBIC)/255.0
                    output_image_r = output_image_t  
                    sigma = 0.0
                input_images[id] = np.expand_dims(input_image, axis=0)
                output_images_t[id] = np.expand_dims(output_image_t, axis=0)
                output_images_r[id] = np.expand_dims(output_image_r, axis=0)

                
                if output_images_r[id].max() < 0.15 or output_images_t[id].max() < 0.15:
                    print("Invalid reflection file %s (degenerate channel)" % (file))
                    continue
                if input_images[id].max() < 0.1:
                    print("Invalid file %s (degenerate image)" % (file))
                    continue

              
                if cnt % 2 == 0:
                    fetch_list = [d_opt]
                    # update D
                    _ = sess.run(fetch_list, feed_dict={
                                 input: input_images[id], target: output_images_t[id]})
                fetch_list = [g_opt, transmission_layer, reflection_layer,
                              d_loss, g_loss,
                              loss, loss_percep, loss_grad]
                
                _, pred_image_t, pred_image_r, current_d, current_g, current, current_percep, current_grad =\
                    sess.run(fetch_list, feed_dict={
                             input: input_images[id], target: output_images_t[id], reflection: output_images_r[id], issyn: is_syn})

                all_l[id] = current
                all_percep[id] = current_percep
                all_grad[id] = current_grad*255
                all_g[id] = current_g
                g_mean = np.mean(all_g[np.where(all_g)])
                print("iter: %d %d || D: %.2f || G: %.2f %.2f || all: %.2f || loss: %.2f %.2f || mean: %.2f %.2f || time: %.2f" %
                      (epoch, cnt, current_d, current_g, g_mean,
                       np.mean(all_l[np.where(all_l)]),
                       current_percep, current_grad*255,
                       np.mean(all_percep[np.where(all_percep)]), np.mean(
                           all_grad[np.where(all_grad)]),
                       time.time()-st))
                cnt += 1
                input_images[id] = 1.
                output_images_t[id] = 1.
                output_images_r[id] = 1.
                

      
        
        if epoch % ARGS.save_model_freq == 1:
            os.makedirs("%s/%04d" % (task, epoch))
            saver.save(sess, "%s/model.ckpt" % task)
            saver.save(sess, "%s/%04d/model.ckpt" % (task, epoch))
            fileid = os.path.splitext(os.path.basename(output_real_names1[id]))[0]
            if not os.path.isdir("%s/%04d/%s" % (task, epoch, fileid)):
                os.makedirs("%s/%04d/%s" % (task, epoch, fileid))
            pred_image_t = np.minimum(np.maximum(pred_image_t, 0.0), 1.0)*255.0
            pred_image_r = np.minimum(np.maximum(pred_image_r, 0.0), 1.0)*255.0
            print("shape of outputs: ", pred_image_t.shape, pred_image_r.shape)
            cv2.imwrite("%s/%04d/%s/int_t.png" % (task, epoch, fileid),
                        np.uint8(np.squeeze(input_image*255.0)))
            cv2.imwrite("%s/%04d/%s/out_t.png" %
                        (task, epoch, fileid), np.uint8(np.squeeze(pred_image_t)))
            cv2.imwrite("%s/%04d/%s/out_r.png" %
                        (task, epoch, fileid), np.uint8(np.squeeze(pred_image_r)))
# test the model on images with reflection
else:
    def prepare_data_test(test_path):
        input_names = []
        for dirname in test_path:
            for _, _, fnames in sorted(os.walk(dirname)):
                for fname in fnames:
                    if is_image_file(fname):
                        input_names.append(os.path.join(dirname, fname))
        return input_names

    # Please replace with your own test image path
    test_path = ["./test_images/real/"]  # ["./test_images/real/"]
    subtask = "real"  # if you want to save different testset separately
    val_names = prepare_data_test(test_path)

    for val_path in val_names:
        testind = os.path.splitext(os.path.basename(val_path))[0]
        if not os.path.isfile(val_path):
            continue
        img = cv2.imread(val_path)
        input_image = np.expand_dims(np.float32(img), axis=0)/255.0
        st = time.time()
        output_image_t, output_image_r = sess.run(
            [transmission_layer, reflection_layer], feed_dict={input: input_image})
        print("Test time %.3f for image %s" % (time.time()-st, val_path))
        output_image_t = np.minimum(np.maximum(output_image_t, 0.0), 1.0)*255.0
        output_image_r = np.minimum(np.maximum(output_image_r, 0.0), 1.0)*255.0
        if not os.path.isdir("./test_results/%s/%s" % (subtask, testind)):
            os.makedirs("./test_results/%s/%s" % (subtask, testind))
        cv2.imwrite("./test_results/%s/%s/input.png" % (subtask, testind), img)
        cv2.imwrite("./test_results/%s/%s/t_output.png" % (subtask, testind),
                    np.uint8(output_image_t[0, :, :, 0:3]))  # output transmission layer
        cv2.imwrite("./test_results/%s/%s/r_output.png" % (subtask, testind),
                    np.uint8(output_image_r[0, :, :, 0:3]))  # output reflection layer
