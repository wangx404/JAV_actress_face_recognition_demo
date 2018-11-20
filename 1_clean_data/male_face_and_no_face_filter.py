# -*- coding: utf-8 -*-
import os, argparse
import mxnet as mx
from mxnet import gluon, image, nd


def parseArgs():
    '''
    Parse command line arguments.
    解析命令行参数。
    '''
    parser = argparse.ArgumentParser(description='Gluon for human face classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, type=str,
                        help='name of the pretrained model from model zoo.') # 模型类型
    parser.add_argument('--class_number', required=True, type=int, default=4,
                        help='class number to predict') # 模型类别
    parser.add_argument('--worker_number', default=4, type=int,
                        help='number of preprocessing workers') # 图片处理线程数
    parser.add_argument('--use_gpu', default=False, type=bool,
                        help='number of gpus to use, 0 indicates cpu only') # 模型运行设备
    parser.add_argument('--input_dir', required=True, type=str,
                        help='input image directory') # 输入文件夹
    parser.add_argument('--output_dir', type=str,
                        help='output image directory') # 输出文件夹
    parser.add_argument('--threshold', type=float,
                        help='probability threshold to remove an input image') # 输出文件夹
    args = parser.parse_args()
    return args


def sixCrop(img, size):
    '''
    Given an image and a size, return six images after random croped.
    输入一张图片和尺寸数据，返回六张随机剪裁后的图片。
    :param img: input image, mx.ndarray, h*w*c
    :param size: target image size, h*w
    :return crops: six images stacked together, mx.ndarray
    '''
    H, W = size
    iH, iW = img.shape[1:3]

    if iH < H or iW < W: # to confirm that image size is larger than target size
        raise ValueError('image size is smaller than crop size')

    crops = nd.stack(
        img[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img[:, 0:H, 0:W],
        img[:, 0:H, iW - W:iW],
        img[:, (iH-H)//2 : (iH+H)//2, 0: W],
        img[:, (iH-H)//2 : (iH+H)//2, (iW-W): iW],
        img[:, 0: H, (iW - W) // 2:(iW + W) // 2]
    )
    return crops


def transformPredict(img):
    '''
    Input an image, transform and random crop the image, return images after processed.
    输入一张图片，对其进行数据形式转换和随机剪裁堆叠后返回。
    :param img: image data, mx.ndarray, h*w*c
    :return img: imafe data, mx.ndarray, b*c*h*w
    '''
    img = img.astype('float32') / 255 # 0-255 to 0-1
    img = image.resize_short(img, 120) # resize
    img = nd.transpose(img, (2,0,1)) # channel transpose to batch * channel * h * w
    img = mx.nd.image.normalize(img, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)) # normalize
    img = sixCrop(img, (112, 112)) # random crop
    return img


def loadModel(model_name, task_num_class, task_param, ctx):
    '''
    Given model name, class number and model param file, return a param-loaded model.
    根据模型名称，类别数据和参数文件，加载并返回一个模型。
    :param model_name: model name, pretrained model in model zoo by MXNet
    :param task_num_class: class number to be predicted
    :param task_param: model param file
    :param ctx: to use cpu or gpu
    :return net: param loaded model structure
    '''
    # 创建新的model,加载参数，重设ctx，hybridize()
    net = gluon.model_zoo.vision.get_model(model_name, classes=task_num_class, prefix='net_')
    net.collect_params().load(task_param, ctx = mx.cpu())
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    return net


def writeResult(string):
    '''
    Write the result into txt file.
    将模型预测结果写入txt文件中。
    :param string: predict result to be writen, string
    :return None:
    '''
    with open('../data/male_face_and_no_face_filter_results.txt', 'a+') as f:
        f.write(string)


def predict(net, ctx, input_dir, threshold=0.9):
    '''
    Using param-loaded model to predict the classification probabilty of input image.
    使用CNN模型预测输入图片的分类概率。
    :param net: param loaded CNN net
    :param ctx: computing device
    :param input_dir: input image directory
    :param threshold: probability threshold
    :return None:
    '''
    movie_list = os.listdir(input_dir) # movie list
    movie_list.sort()
    for movie in movie_list:
        image_list = os.listdir(os.path.join(input_dir, movie)) # image list
        for _image in image_list:
            image_file = os.path.join(input_dir, movie, _image)
            try: # try to read and decode
                with open(image_file, 'rb') as f:
                    img = image.imdecode(f.read())
            except Exception as e:
                print('Fail to read image %s in movie %' % (_image, movie))
                print('And the error is ', e)
                continue
            # predict
            data = transformPredict(img)
            data = data.as_in_context(ctx)
            out = net(data)
            out = nd.SoftmaxActivation(out).mean(axis=0) # softmax process
            out = out.asnumpy().tolist() # array to list
            # judge and delete
            if (out[2] > threshold) or (out[3] > threshold):
                os.remove(os.path.join(input_dir, movie, _image))
            # you can just write the result into file without doing anything.       
            out = [str(number) for number in out]
            string = '%s:%s' % (image_file, ','.join(out))
            writeResult(string+'\n')
            # you can also move these images to another directory
        
        print('Movie %s finished.' % movie)


if __name__ == "__main__":
    # parse command line arguments
    args = parseArgs()
    task = 'face_classification'
    model_name = args.model
    task_num_class = args.class_number
    task_param = '..data/%s_%s.params' % (model_name, task)
    use_gpu = args.use_gpu
    ctx = mx.gpu() if use_gpu else mx.cpu()
    num_workers = args.worker_number
    input_dir = args.input_dir
    threshold = args.threshold
    
    net = loadModel(model_name, task_num_class, task_param, ctx)
    predict(net, input_dir, threshold)
