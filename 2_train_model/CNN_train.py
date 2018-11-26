# -*- coding: utf-8 -*-
import argparse
import os, time, logging, math
import mxnet as mx
from mxnet import gluon
from mxnet import nd, autograd
from CNN_utils import evaluateAccuracy, dataLoader
from CNN_model import CenterLoss, LeNetPlus


def parseArgs():
    '''
    Parsing some parameters from command line to run this script.
    从命令行解析相关参数，用于脚本的运行。
    :param None:
    :return args: parameter dictory
    '''
    parser = argparse.ArgumentParser('Convolutional Neural Networks')
    # File related
    parser.add_argument('--prefix', default='face-recoginition', type=str, 
                        help='prefix of model param')
    parser.add_argument('--ckpt_dir', default='../data/ckpt', type=str, 
                        help='check point directory')
    parser.add_argument('--input_dir', type=str, help='face image dataset directory')
    # Training related
    parser.add_argument('--use_gpu', default=False, type=bool, 
                        help='whether to use gpu')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='number of cpu to process input image data')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='start learning rate')
    parser.add_argument('--lr_step', default=10, type=int, help='learning rate decay period')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--lmbd', default=1, type=float, 
                        help='lambda in the paper, center loss weight') # center loss的权重大小
    parser.add_argument('--alpha', default=0.5, type=float, 
                        help='alpha in the paper, learning rate of center loss net') # center loss net 参数更新速度
    # Model related
    parser.add_argument('--num_classes', default=10, type=int, help='number of actresses')
    parser.add_argument('--feature_dim', default=128, type=int, help='dimension of feature')
    # whether to train or test    
    parser.add_argument('--train', action='store_true', help='choose to train the model')
    parser.add_argument('--test', action='store_true', help='choose to test the model')
    args = parser.parse_args()
    return args


def progressBar(i, n, bar_len=40):
    '''
    Print the progress of training or testing.
    打印训练或者测试的进度。
    :param : batch index
    :param : batch number
    :param : bar length
    :return None:
    '''
    percents = math.ceil(100.0 * i / float(n))
    filled_len = int(round(bar_len * i / float(n)))
    prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print("[%s] %s%s" % (prog_bar, percents, "%"), end="\r")


def train(args):
    '''
    Train CNN model on train dataset, and save parameters of the model with the best accuracy.
    在训练集上训练一个CNN模型，并将准确率最高的模型参数保存。
    :param args: arguments parsed from command line
    :return None:
    '''
    print('Start to train.')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    train_iter, test_iter = dataLoader(args.input_dir, args.batch_size, args.num_workers)
    num_batch = len(train_iter)
    ctx = mx.gpu() if args.use_gpu else mx.cpu() # chooose gpu or cpu
    # load model and initialize
    model = LeNetPlus(args.num_classes, args.feature_dim) # 通过类别数目和特征数目确定模型的构成
    model.hybridize()
    model.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    # loss and trainer
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(),
                            optimizer='sgd', optimizer_params={'learning_rate': args.lr, 'wd': args.wd})
    # center loss network and trainer
    # actually center loss is a model not a function
    center_loss = CenterLoss(args.num_classes, args.feature_dim, lmbd=args.lmbd)
    center_loss.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer_center = gluon.Trainer(center_loss.collect_params(),
                                       optimizer='sgd', optimizer_params={'learning_rate': args.alpha})
    
    moving_loss = 0.0
    smoothing_constant = 0.01
    best_acc = 0.0
    for epoch in range(args.epochs):
        start_time = time.time()
        if (epoch !=0) and (epoch % args.lr_step == 0): # set learning rate after every lr_step epoch
            trainer.set_learning_rate(trainer.learning_rate*args.lr_decay)
            trainer_center.set_learning_rate(trainer_center.learning_rate*args.lr_decay)
        
        for i, (data, label) in enumerate(train_iter):
            #print('training...')
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output, features = model(data) # forward computing 
                loss_softmax = softmax_cross_entropy(output, label)
                loss_center = center_loss(features, label) # 加入center loss
                loss = loss_softmax + loss_center

            loss.backward()# backward computing
            trainer.step(data.shape[0]) # update params of model
            trainer_center.step(data.shape[0]) # update params of center net
            
            # smoth loss, to avoid loss fluctuation
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (epoch == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
            
            progressBar(i, num_batch) 
        lasting_time = time.time() - start_time
        # 在每训练完一轮后，使用model进行前向计算。
        train_accuracy = evaluateAccuracy(train_iter, model, ctx)
        test_accuracy = evaluateAccuracy(test_iter, model, ctx)
        # output training information into log
        logging.warning("Epoch [%d]: Loss=%f" % (epoch, moving_loss))
        logging.warning("Epoch [%d]: Train-Acc=%f" % (epoch, train_accuracy))
        logging.warning("Epoch [%d]: Test-Acc=%f" % (epoch, test_accuracy))
        logging.warning("Epoch [%d]: Epoch-time=%f" % (epoch, lasting_time))
        # 保存accuracy最高的model参数
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            model.save_params(os.path.join(args.ckpt_dir, args.prefix + '-best.params'))


def test(args):
    '''
    Test the accuracy of the model on validation dataset.
    测试模型在验证集上的准确率。
    :param args: arguments parsed from command line
    :return None:
    '''
    print('Start to test.')
    ctx = mx.gpu() if args.use_gpu else mx.cpu()
    _, test_iter = dataLoader(args.input_dir, args.batch_size, args.num_workers)

    model = LeNetPlus(args.num_classes, args.feature_dim)
    model.load_params(os.path.join(args.ckpt_dir, args.prefix + '-best.params'), ctx=ctx)

    start_time = time.time()
    test_accuracy = evaluateAccuracy(test_iter, model, ctx)
    lasting_time = time.time() - start_time

    print("Test_accuracy: %s, Lasting_time: %f s" % (test_accuracy, lasting_time))


if __name__ == '__main__':
    args = parseArgs()
    if args.train:
        train(args)
    if args.test:
        test(args)
    
