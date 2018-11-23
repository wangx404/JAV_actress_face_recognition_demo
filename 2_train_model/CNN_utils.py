# -*- coding: utf-8 -*-
import numpy as np
import mxnet as mx
from mxnet import gluon, image, nd
import os


def evaluateAccuracy(data_iterator, net, ctx):
    '''
    Evaluate the accuracy of the model on dataset (mainly used on validation set).
    评估模型的准确率。
    :param data_iterator: 
    :param net: CNN net model
    :param ctx: computing context
    :return: top-1 accuracy
    '''
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output, feature = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label) # 直接调用的mx.metric.Accuracy进行的评估
    return acc.get()[1]


def transformTrain(data, label):
    '''
    Image transform function for train dataset, 
    include normalize, resize, augmentation, channel transpose.
    训练数据的图片处理函数，包括了归一化，缩放，数据增广，通道调整。
    :param data: image data, mx.nd.ndarray
    :param label: image label
    :return: image and label after transformation
    '''
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 160, 140), resize=140,
                                    rand_crop=False, rand_mirror=True,
                                    mean = np.array([0.5, 0.5, 0.5]),
                                    std = np.array([0.2, 0.2, 0.2]),
                                    brightness = 0.05,
                                    contrast = 0.05,
                                    saturation = 0.05)
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar())


def transformVal(data, label):
    '''
    Image transform function for validate dataset, 
    include normalize, resize, augmentation, channel transpose.
    验证数据的图片处理函数，包括了归一化，缩放，数据增广，通道调整。
    :param data: image data, mx.nd.ndarray
    :param label: image label
    :return: image and label after transformation
    '''
    im = data.astype('float32') / 255
    im = image.resize_short(im, 140)
    im, _ = image.center_crop(im, (160, 140))
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    return (im, nd.array([label]).asscalar())


def transformPredict(data):
    '''
    Input an image, after normalization, resizement, channel transpose, stack it
    and it's mirror together to get an batch data.
    输入一张图片，经过归一化，缩放和通道调整后，将它与其镜像进行拼接，得到batch输入数据。
    :param data: image data, (H, W, C)
    :return im: batch data, (2, C, H, W)
    '''
    im = data.astype('float32') / 255
    im = image.resize_short(im, 140)
    im, _ = image.center_crop(im, (160, 140))
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    im = nd.stack(im,
                  im[:,:,::-1]) # stack image and its mirror image
    return im


def dataLoader(input_dir, batch_size, num_workers):
    '''
    Input image directory, batch size, number of workers, return train and val data-iter.
    输入图片所在的文件夹、批次大小和图片处理线程数，返回训练数据和验证数据的迭代器。
    :param input_dir: input image directory
    :param batch_size: batch size
    :param num_workers: number of wokers to process input image data
    :return :
    '''
    train_data_dir = os.path.join(input_dir, 'train')
    val_data_dir = os.path.join(input_dir, 'val')
    # dataset directory -> dataset class -> dataset loader
    train_data_iter = gluon.data.DataLoader(
            gluon.data.vision.ImageFolderDataset(train_data_dir,
                transform=transformTrain),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='rollover')
    val_data_iter = gluon.data.DataLoader(
            gluon.data.vision.ImageFolderDataset(val_data_dir,
                transform=transformVal),
            batch_size=batch_size, shuffle=False, num_workers = num_workers)
    return train_data_iter, val_data_iter
