# -*- coding: utf-8 -*-
"""Shared dlib face detection and cropping helpers."""
import dlib


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
