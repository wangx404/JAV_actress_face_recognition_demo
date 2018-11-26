# -*- coding: utf-8 -*-
from mxnet import gluon
import numpy as np


class CenterLoss(gluon.HybridBlock):
    """
    Center loss block, not a ndarray matrix.
    It can save feature array and calculate center loss.
    使用gluon的HybridBlock实现的center loss类，能存储feature特征并计算center loss。
    """
    def __init__(self, num_classes, feature_dim, lmbd, **kwargs):
        '''
        :param num_classes: number of actress class
        :param feature_dim: feature dimension
        :param lmbd: lambda factor of center loss
        '''
        super(CenterLoss, self).__init__(**kwargs)
        self._lmda = lmbd
        self.centers = self.params.get('centers', shape=(num_classes, feature_dim)) # 参数矩阵

    def hybrid_forward(self, F, feature, label, centers):
        '''
        :param feature: output feature of mini batch data
        :param label: label of mini batch data
        :param centers: center features of all classes
        '''
        # 计算label的统计分布（0:counter0, 1:counter1, maximun_value:count_m）
        # 从零开始统计到最大值区间内label出现的频次
        hist = F.array(np.bincount(label.asnumpy().astype(int))) 
        # 取出label对应的频次（在本batch中），用于对label的损失降权
        # 从而保证所有的向量都得到同等程度的更新
        centers_count = F.take(hist, label)
        # 取出label对应位置的特征向量（和np.take略有不同，dimension kept）
        centers_selected = F.take(centers, label)
        # 计算输出特征和特征矩阵的差值
        diff = feature - centers_selected 
        # loss = loss_weight * 0.5 * sum(diff**2, axis=1) / label_weight
        loss = self._lmda * 0.5 * F.sum(F.square(diff), axis=1) / centers_count
        # mean along axis beside 0?
        return F.mean(loss, axis=0, exclude=True) 


def _make_conv_block(block_index, num_channel=32, num_layer=2, kernel_size=5, strides=1, padding=2):
    '''
    Input some variable, output a convolution block, 
    which contains (convolution, activation, maxpooling) * 2.
    输入一些变量，得到一个包含(卷积，激活，池化)×2的卷积模块。
    :param block_index: block index to name block
    :param num_channel: number of convolution channel
    :param num_layer: number of convolution layer
    :param kernel_size: kernel size in convolution layer
    :param strides: striders size in convolution layer 
    :param padding: padding size in convolution layer
    :return out: convolution block
    '''
    out = gluon.nn.HybridSequential(prefix='block_%d_' % block_index)
    with out.name_scope():
        for _ in range(num_layer):
            out.add(gluon.nn.Conv2D(num_channel, kernel_size=kernel_size, 
                                    strides=strides, padding=padding, activation='relu'))
            #out.add(gluon.nn.LeakyReLU(alpha=0.01))
        out.add(gluon.nn.MaxPool2D())
    return out


class LeNetPlus(gluon.nn.HybridBlock):
    '''
    A modified version of LeNet.
    LeNet的变形版本。
    '''
    def __init__(self, classes=10, feature_dim=2, **kwargs):
        super(LeNetPlus, self).__init__(**kwargs) # initi with super class function
        '''
        :param classes: class number of LeNetPlus
        :param feature_dim: feature dimension
        '''
        num_channels = [32, 64, 128]
        with self.name_scope():
            self.features = gluon.nn.HybridSequential(prefix='')

            for i, num_channel in enumerate(num_channels):
                self.features.add(_make_conv_block(i, num_channel=num_channel))
            
            self.features.add(gluon.nn.Flatten()) # flatten layer
            self.features.add(gluon.nn.Dense(feature_dim)) # feature layer
            self.output = gluon.nn.Dense(classes) # output layer

    def hybrid_forward(self, F, x, *args, **kwargs):
        feature = self.features(x)
        output = self.output(feature)
        return output, feature
    
