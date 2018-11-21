# -*- coding: utf-8 -*-
import os, shutil, random
import argparse


def parseArgs():
    parser = argparse.ArgumentParser(description="Resort image by actress's name, and split the dataset into \
                                                    train dataset and val dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--movie_image_dir', required=True, type=str,
                        help='movie image directory')
    parser.add_argument('--movie_detail_dir', required=True, type=str,
                        help='movie detail directory')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='output image directory')
    parser.add_argument('--split_ratio', type=float,
                        help='size of validate dataset')
    args = parser.parse_args()
    return args


def resortImagesByName(image_dir, detail_dir, output_dir):
    '''
    According to information in movie detail, copm image from image directory 
    to actress directory.
    根据影片资料中关于出演女优的信息将图片从影片文件间copy至对应的女优文件夹中。
    :param image_dir: movie image directory
    :param detail_dir: movie detail directory
    :param output_dir: output image directory
    :return None:
    '''
    movie_list = os.listdir(image_dir)
    movie_list.sort()
    for movie in movie_list:
        # according to movie detail to judge if this movie has single actress.
        if not os.exists(os.path.join(detail_dir, movie, '%s.txt' % movie)):
            print('Movie %s has no detail.')
            continue
        # open detail, and parse actress number
        with open(os.path.join(detail_dir, movie, '%s.txt' % movie)) as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        actress = lines[12]
        
        if movie not in lines[1]:
            print('Detail ID is not consistent with movie dir.')
            continue
        if 'Actress' not in actress:
            print('Not actress item in line 13, you should check the detail file.')
            continue
        actress = actress.split(':')[1]
        if 'None' in actress:
            print('No actress in movie %s' % movie)
            continue
        actress = actress.split(',')
        if len(actress) != 1: # multiple actresses
            continue
        actress = actress[0]
        
        # copy images to another directory by actree name
        if not os.path.exists(os.path.join(output_dir, 'train', actress)):
            os.makedirs(os.path.join(output_dir, 'train', actress))
        image_list = os.listdir(os.path.join(image_dir, movie))
        for image in image_list:
            shutil.copy2(os.path.join(image_dir, movie, image), 
                        os.path.join(output_dir, 'train', actress, image))


def splitDataset(output_dir, split_ratio=0.2):
    '''
    Split the dataset into train dataset and validate dataset.
    将数据集切割为训练集和验证集两部分。
    :param output_dir: output image directory
    :param split_ratio: image proportion of validate dataset, default value is 20%
    :return None:
    '''
    actress_list = os.listdir(output_dir, 'train')
    actress_list.sort()
    
    for actress in actress_list:
        image_list = os.listdir(os.path.join(output_dir, 'train', actress))
        random.shuffle(image_list)
        length = len(image_list)
        image_list = image_list[:int(length*split_ratio)]
        
        if not os.path.exists(os.path.join(output_dir, 'val', actress)):
            os.makedirs(output_dir, 'val', actress)
        for image in image_list:
            shutil.move(os.path.join(output_dir, 'train', actress, image), 
                        os.path.join(output_dir, 'val', actress, image))

    
if __name__ == '__main__':
    args = parseArgs()
    image_dir = args.movie_image_dir
    detail_dir = args.movie_detail_dir
    output_dir = args.output_dir
    split_ratio = args.split_ratio
    
    resortImagesByName(image_dir, detail_dir, output_dir)
    splitDataset(output_dir, split_ratio)
    
