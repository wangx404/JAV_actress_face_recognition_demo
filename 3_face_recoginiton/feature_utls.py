# -*- coding: utf-8 -*-
import os, json
import dlib
import numpy as np

dlib_cnn_face_detector = dlib.cnn_face_detection_model_v1('model/mmod_human_face_detector.dat')
dlib_landmark_predictor_5 = dlib.shape_predictor('model/shape_predictor_5_face_landmarks.dat')
dlib_face_recognizor = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')


def computeFeature(image, face_detector= dlib_cnn_face_detector, landmark_detector= dlib_landmark_predictor_5, face_recognizor= dlib_face_recognizor):
    '''
    Input an image, and compute face feature (128d) with dlib.
    输入一张图片，使用dlib计算人脸的特征向量（128维）。
    :param image: 人脸图片
    :param face_detector: 人脸检测器
    :param landmark_detector: 人脸关键点检测器
    :param face_recognizor: 人脸特征计算器
    :return face_feature: 人脸特征
    '''
    img = dlib.load_rgb_image(image)
    faces = face_detector(img, 1) # detect face
    face = faces[0].rect
    landmark = landmark_detector(img, face) # detect face landmark
    face_feature = face_recognizor.compute_face_descriptor(img, landmark) # compute face feature
    face_feature = np.array(face_feature)
    return face_feature


def computeFeatures(input_dir):
    '''
    Input an image directory and compute all images' features.
    输入图片文件夹，计算所有图片的特征向量。
    :param input_dir: image directory
    :return: features of all images in image directory
    '''
    features = []
    image_list = os.listdir(input_dir)
    image_list.sort()
    for image in image_list:
        image = os.path.join(input_dir, image)
        try:
            feature = computeFeature(image)
        except Exception: # 跳过检测错误
            continue
        feature = feature.tolist()
        features.append(feature)
    if features == []:
        return np.zeros((1, 128))
    return np.array(features)


def calcFeatureCenter(features):
    '''
    Calculate feature center of feature arrays in one class.
    计算一个类别的特征向量的中心点。
    :param features: feature arrays of all images of one class
    :return f_center: feature center
    '''
    f_center = features.mean(axis=0)
    return f_center


def calcFeatureDistance(feature, f_center):
    '''
    Calculate euclidean distance between two feature array.
    计算两个特征向量的欧氏距离。
    :param f_center: feature center of one class
    :param f_array: one feature array of one image
    :return: euclidean distance
    '''
    return np.sum((f_center - feature)**2)


def calcMeanAndDeviation(features, f_center):
    '''
    Calculate mean distance and distance deviation of features to center feature.
    :param f_arrays: feature arrays of all images of one class
    :param f_center: feature center
    :return mean_distance: mean distance of all feature arrays to the feature center
    :return distance_std: distance deviation of all feature arrays to the feature center
    '''
    f_distance = (features - f_center)**2 # 到中心向量的差值的平方（即欧氏距离）
    f_distance = f_distance.sum(axis=1) # 将各个维度上的距离进行加合
    mean_distance = f_distance.mean(axis=0) # 距离的平均值
    distance_std = f_distance.std() # 距离的标准差
    return mean_distance, distance_std


def inDistanceBorder(distance, mean_distance, distance_deviation, sigma=3):
    '''
    Judge feature distance larger than distance border or not,
    according to 3 sigma principle.
    根据3sigma原则判断一个距离是否超出了特征向量的边界。
    :param distance: distance between a feature to the feature center
    :param mean_distance: mean distance value
    :param distance_deviation: distance deviation value
    :param sigma: border size
    :return: whether or not the distance is in the distance border
    '''
    distance_border = distance_deviation * sigma
    if abs(distance - mean_distance) <= distance_border:
        return True
    else:
        return False


def cleanWrongFace(input_dir, dist_threshold=0.6):
    '''
    Delete image whose distance from the feature center is larger than distance threshold.
    将actress文件夹中距离平均特征的过远的图片删除。
    :param input_dir: actress directory
    :dist_threshold: distance threshold
    :return None:
    '''
    # 先找到actress的平均特征，再计算每张图片到平均特征的距离
    image_feature_dict = {}
    image_list = os.listdir(input_dir)
    for image in image_list:
        image = os.path.join(input_dir, image)
        try:
            feature = computeFeature(image)
        except Exception:
            continue
        image_feature_dict[image] = feature
    # calculate feature center
    features = [f.tolist() for f in image_feature_dict.values()]
    features = np.array(features)
    feature_center = calcFeatureCenter(features)
    # 逐一比较图片特征距平均特征的距离
    for image, feature in image_feature_dict.items():
        distance = calcFeatureDistance(feature, feature_center)
        if distance > dist_threshold:
            os.remove(image) # print image name and delete
            print(image, distance)


def getActressFeatures(input_dir, csv_file='actress_feature.csv'):
    '''
    Input a directory contains multiple actresses, 
    coumputer feature center for every actress and save into csv file.
    输入包含多个actress的文件夹，为每一个actress计算平均特征并保存到csv文件中。
    :param input_dir: actresses' directory
    :return None:
    '''
    actress_features = []
    actress_list = os.listdir(input_dir)
    actress_list.sort()
    for actress in actress_list:
        actress_feature = computeFeatures(os.path.join(input_dir, actress))
        actress_feature = calcFeatureCenter(actress_feature)
        actress_features.append(actress_feature.tolist())
        print('Feature of actress %s has been computed.' % actress)
    # format transform and save 
    actress_features =  np.array(actress_features)
    np.savetxt(csv_file, actress_features, delimiter=',')


def saveIndex2Actress(input_dir, json_file='index2actress.json'):
    '''
    Traversal input directory contains all actress, 
    and save index-actress key-value pair to json file.
    遍历女优所在文件夹，将索引-女优名作为键值对存储到json文件中。
    :param input_dir: directory contains actress-directory
    :param json_file: json file to save the index-actress dict
    :return None:
    '''
    index2actress = dict() # index to actress dict
    actress_list = os.listdir(input_dir)
    actress_list.sort() # this step cann't be ignored.
    for index, actress in enumerate(actress_list):
        index2actress[index] = actress
    # save the dict to json file
    with open(json_file, 'w') as f:
        json.dump(index2actress, f)
    

def loadIndex2Actress(json_file='index2actress.json'):
    '''
    Load index to actress dict from json file.
    从json文件中加载索引-女优名字典。
    :param json_file: json file
    :return index2actress: index to actress dict
    '''
    with open(json_file, 'r') as f:
        index2actress = json.load(f)
    return index2actress


def loadActressFeatures(csv_file='actress_feature.csv'):
    '''
    Load actress-feature matrix from csv file.
    从文件中加载actress-feature矩阵。
    '''
    actress_features = np.loadtxt(csv_file, delimiter=',')
    return actress_features


def KNNSearch(actress_features, unknown_feature, nearest_num=1):
    '''
    Using KNN algorithm to search the nearest face feature index, 
    which corresponds to related actress.
    使用KNN搜索算法找到和特征向量最近的人脸特征索引。
    :param features: all actresses' face feature
    :param unkown_feature: unkown woman's face feature
    :param numbers: number of returned indexs
    :return: feature index
    '''
    unknown_feature = unknown_feature.reshape((1,-1))
    distances = (actress_features - unknown_feature)**2
    distances = distances.sum(axis=1) # sum along feature axis
    indexs = distances.argsort()
    indexs = indexs.tolist()  
    return indexs[:nearest_num]
    
