# -*- coding: utf-8 -*-
import os, math
import cv2
import dlib
import numpy as np
import argparse
import multiprocessing


def parseArgs():
    parser = argparse.ArgumentParser(description='Choose model and treadnumbers to align face images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', required=True, type=str,
                        help='where to input movie images')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='where to save movie images')
    parser.add_argument('--face_detector', required=True, type=str,
                        help='which face detect function to use')
    parser.add_argument('--landmark_detector', required=True, type=str,
                        help='which landmark detect function to use')
    parser.add_argument('--threads', required=True, type=int,
                        help='process number to run face detect function')
    args = parser.parse_args()
    return args


dlib_detector = dlib.get_frontal_face_detector()
def dlibFrontalFaceDetect(img, upsampling=1):
    '''
    Using dlib frontal face detector to detect faces in image.
    使用dlib提供的前脸检测器进行人脸检测。
    :param img: face image
    :param upsampling: upsample ratio for dlib frontal face detector
    :return result: face box in format [[xmin, ymin, xmax, ymax], ]
    '''
    if type(img) == str:
        img = dlib.load_rgb_image(img)
    faces = dlib_detector(img, upsampling)
    if len(faces) == 0:
        return None
    result = []
    for face in faces:
        result.append([face.left(), face.top(), face.right(), face.bottom()])
    return result


dlib_cnn_face_detector = dlib.cnn_face_detection_model_v1('../data/mmod_human_face_detector.dat')
def dlibCNNFaceDetect(img):
    '''
    Using dlib cnn face detector to detect faces in image.
    使用dlib的cnn人脸检测器进行人脸检测。
    :param img:  face image
    :return result: face box in format [[xmin, ymin, xmax, ymax], ]
    '''
    if type(img) == str:
        img = dlib.load_rgb_image(img)
    faces = dlib_cnn_face_detector(img, 1)
    if len(faces) == 0:
        return None
    result = []
    for face in faces:
        result.append([face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()])
    return result


def faceCrop(img, xmin, ymin, xmax, ymax, scale=2.0):
    '''
    Input an image and the location of face, crop face with margin.
    输入图片和人脸坐标，将带有边缘的人脸剪裁返回。
    :param img: image, numpy ndarray
    :param xmin,ymin,amax,ymax: face box location
    :param scale: the bigger the scale is, the bigger the margin around face is
    :return face: face with margin, numpy ndarray
    '''
    hmax, wmax, _ = img.shape
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin) * scale
    h = (ymax - ymin) * scale
    # new xmin, ymin, xmax and ymax
    xmin = x - w/2
    xmax = x + w/2
    ymin = y - h/2
    ymax = y + h/2
    # 坐标修正为有效数字
    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(wmax, int(xmax))
    ymax = min(hmax, int(ymax))
    # crop and return
    face = img[ymin:ymax,xmin:xmax,:]
    return face


def findEye(landmarks):
    '''
    Find out the center coordinate of left eye and right eye,
    then they will be used to rotate face.
    找出左右眼中心的x,y坐标值，用于脸部的旋转对齐。
    '''
    left_eye = landmarks[36:42]
    left_eye = np.array([p for p in left_eye])
    left_eye = left_eye.mean(axis=0)
    right_eye = landmarks[42:48]
    right_eye = np.array([p for p in right_eye])
    right_eye = right_eye.mean(axis=0)
    return left_eye, right_eye


def findNose(landmarks):
    '''
    Find out the center coordinate of  nose.
    找到鼻子的中心点（暂时无用）。
    '''
    nose = landmarks[31:36]
    nose = np.array([p for p in nose])
    nose = nose.mean(axis=0)
    return nose


def findMouth(landmarks):
    '''
    Find out the center coordinate of  mouth.
    找到嘴巴的中心点（暂时无用）。
    '''
    mouth = landmarks[48:]
    mouth = np.array([p for p in mouth])
    mouth = mouth.mean(axis=0)
    return mouth
    

dlib_landmark_predictor_68 = dlib.shape_predictor('../data/shape_predictor_68_face_landmarks.dat')
def dlib68FacialLandmarkDetect(img, face):
    '''
    Input an image with face area, output facial landmark with 68 points.
    输入图片以及人脸的区域，检测68个人脸关键点。
    :param img: image
    :param face: face area, dib.rectangle
    :return: landmarks, coordinates of key points
    '''
    if type(img) == str:
        img = dlib.load_rgb_image(img)
    # 检测facial landmarks
    landmarks = dlib_landmark_predictor_68(img, face)
    landmarks = landmarks.parts()
    left_eye, right_eye = findEye(landmarks)
    nose = findNose(landmarks)
    mouth = findMouth(landmarks)
    return  left_eye, right_eye, nose, mouth


dlib_landmark_predictor_5 = dlib.shape_predictor('../data/shape_predictor_5_face_landmarks.dat')
def dlib5FacialLandmarkDetect(img, face):
    '''
    Input an image with face area, output facial landmark with 5 points.
    输入图片以及人脸的区域，检测5个人脸关键点。
    :param img: image
    :param face: face area, dib.rectangle
    :return: coordinate of left eye, right eye, nose and mouse
    '''
    if type(img) == str:
        img = dlib.load_rgb_image(img)
    landmarks = dlib_landmark_predictor_5(img, face)
    landmarks = landmarks.parts()
    
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    mouth = (landmarks[3] + landmarks[4]) / 2
    return left_eye, right_eye, nose, mouth


def calcFaceDegree(left_eye, right_eye):
    '''
    Calculate slope degree of face from eye coordinates.
    从眼睛坐标计算脸部的倾斜角度。
    :param left_eye: coordinate of left eye, dlib.point
    :param right_eye: coordinate of right eye, dlib.point
    :return degree: slope degree
    '''
    x1 = left_eye.x
    y1 = left_eye.y
    x2 = right_eye.x
    y2 = right_eye.y
    deltaX = x2 - x1
    deltaY = y2 - y1
    # deltaX等于0时，意味着人脸是横向的，需要向左或者向右旋转90度
    if deltaX == 0:
        return None
    degree = np.arctan(deltaY / deltaX) * 180.0 / np.pi
    return degree


def rotateFace(img, center, degree, scale=1.0):
    '''
    Invoke function in cv2, rotate face to vertical state.
    调用cv2中的函数,返回一个旋转至竖直状态的人脸图片。
    :param img: face image, numpy.ndarray
    :param center: rotation center, usually it's eye center
    :param degree: slope degree
    :param scale: to see it in cv2 document
    :return rotated_img: face image after rotated.
    '''
    if type(img) == str:
        img = cv2.imread(img)
    h, w = img.shape[:2]
    # get rotation matrix 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, degree, scale)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    return rotated_img


def getVerticalFace(image, face_detector, landmark_detector):
    '''
    Detect face and landmark, rotate face to vertical state.
    检测人脸和人脸关键点，旋转得到一个竖直状态的人脸。
    :param image: face image
    :param face_detector: face detector function
    :param landmark_detector: face landmark detector function
    :return vertical_face: vertical face image
    '''
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # color space transform
    # detect face
    faces = face_detector(img)
    if faces == None:
        print('No face in image %s' % image)
        return
    # detect facial landmark and calculate slope degree
    face = faces[0]
    face = dlib.rectangle(*face)
    left_eye, right_eye, _nose, _mouth = landmark_detector(img, face)
    degree = calcFaceDegree(left_eye, right_eye)
    # rotate face
    x = (left_eye.x + right_eye.x) / 2.0
    y = (left_eye.y + right_eye.y) / 2.0
    vertical_face = rotateFace(img, (x, y), degree)
    return vertical_face


def correctFaceLocation(img, face, left_eye, right_eye):
    '''
    Correct face location to let eye in the image center;
    increase length of face location at the same time.
    修正脸部坐标以便让两眼位于图片正中，同时增加纵向长度使长宽比为8:7。
    :param img: face image
    :param face: face box location (xmin, ymin, xmax, ymax)
    :param left_eye: left eye, dlib.point
    :param right_eye: right eye, dlib.point
    :return: face location after correction
    '''
    hmax, wmax, _ = img.shape # image maximum height and width
    xmin, ymin, xmax, ymax = face # face location
    center_x = (left_eye.x + right_eye.x) / 2.0 # (x, y) of eye center
    width = xmax - xmin # width of face box
    # calculate new coordinates
    xmin = center_x - width/2.0 # 以眼部正中为基准计算两侧的边界
    xmax = center_x + width/2.0
    ymin = ymin
    ymax = ymin + width * 8.0 / 7.0
    # 
    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(wmax, int(xmax))
    ymax = min(hmax, int(ymax))
    return (xmin, ymin, xmax, ymax)


def faceAlignment(image, face_detector, landmark_detector):
    '''
    Get a vertical face, detect face and landmark once again, 
    align face to let eye in the center of image.
    先得到一个竖直的人脸，再一次进行人脸和关键点检测，最终进行人脸对齐。
    :param image: face image, str
    :param face_detector: face detector function
    :param landmark_detector: face landmark detector function
    :return align_face_img:
    '''
    vertical_face_img = getVerticalFace(image, face_detector, landmark_detector)
    # detect face once again
    faces = face_detector(vertical_face_img)
    if faces == None:
        print('No face in image %s' % image)
        return
    # detect facial landmark and calculate slope degree
    face = faces[0]
    _face = dlib.rectangle(*face) # face copy
    left_eye, right_eye, _nose, _mouth = landmark_detector(vertical_face_img, _face)
    # calculate crop coordinates
    new_face = correctFaceLocation(vertical_face_img, face, left_eye, right_eye)
    align_face_img = faceCrop(vertical_face_img, *new_face, scale=1.0)
    align_face_img = cv2.cvtColor(align_face_img, cv2.COLOR_RGB2BGR) # 转回BGR色彩空间
    return align_face_img


def singleThreadFaceAlignment(input_dir, output_dir, face_detector, landmark_detector, threads=1, thread_index=0):
    '''
    Input many movie image directories, align and save face images.
    输入图片文件夹，对其中的图片进行人脸检测对齐保存。
    :param input_dir: input directory where to input movie images
    :param output_dir: output directory where to output movie images
    :param face_detector: dlib fromtal face detector or dlib cnn face detector
    :param landmark_detector: threshold to discard small faces
    :param threads: thread numbers
    :param thread_index: thread index
    :return None:
    '''
    print('Thread %d start.' % thread_index)
    movie_list = os.listdir(input_dir)
    movie_list.sort()
    length = math.ceil(len(movie_list)/threads)
    movie_list = movie_list[thread_index*length: (thread_index+1)*length]
    for movie in movie_list:
        if not os.path.exists(os.path.join(output_dir, movie)):
            os.makedirs(os.path.join(output_dir, movie))
        else:
            continue
        
        image_list = os.listdir(os.path.join(input_dir, movie))
        for image in image_list:
            image_file = os.path.join(os.path.join(input_dir, movie, image))
            try:
                img = faceAlignment(image_file, face_detector, landmark_detector)
                new_image_file = os.path.join(os.path.join(output_dir, movie, image))
                cv2.imwrite(new_image_file, img)
            except Exception as e:
                print('Something went wrong when process image ', image_file)
                print('And the error is ', e)
        print('Images of movie %s processed.' % movie)
    
def mutilThreadFaceAlignment(input_dir, output_dir, face_detector, landmark_detector, threads):
    '''
    mutil process fuction which invoke facesCropAndFilter() function.
    调用单进程函数的多进程人脸对齐函数。
    :param input_dir: the root input directory where to input movie images
    :param output_dir: the root input directory where to save movie images
    :param detector: dlib fromtal face detector od dlib cnn face detector
    :param threshold: threshold to discard small faces
    :param threads: thread numbers
    :return None:
    '''
    pool = multiprocessing.Pool(processes=threads)
    for thread_index in range(threads):
        pool.apply_async(singleThreadFaceAlignment, (input_dir, output_dir,  
                                                     face_detector, landmark_detector, 
                                                     threads, thread_index))

    pool.close()
    pool.join()


if __name__ == '__main__':
    args = parseArgs()
    input_dir = args.input_dir
    output_dir = args.output_dir
    if args.face_detector == 'dlib_cnn':
        face_detector = dlibCNNFaceDetect
    elif args.face_detector == 'dlib':
        face_detector = dlibFrontalFaceDetect
    else:
        raise 'You should choose a right face detect function.'
    if args.landmark_detector == 'dlib_68':
        landmark_detector = dlib68FacialLandmarkDetect
    elif args.landmark_detector == 'dlib_5':
        landmark_detector = dlib5FacialLandmarkDetect
    else:
        raise 'You should choose a right landmark detect function.'
    threads = args.threads
    mutilThreadFaceAlignment(input_dir, output_dir, face_detector, landmark_detector, threads)
    
