# -*- coding: utf-8 -*-
import os, math
import dlib, cv2
import multiprocessing
import argparse


def parseArgs():
    parser = argparse.ArgumentParser(description='Choose model and treadnumbers to clean images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', required=True, type=str,
                        help='where to input movie images')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='where to save movie images')
    parser.add_argument('--face_detector', required=True, type=str,
                        help='choose face detection model')
    parser.add_argument('--threshold', required=True, type=int,
                        help='threshold to filter small faces')
    parser.add_argument('--threads', required=True, type=int,
                        help='process number to run face detect function')
    args = parser.parse_args()
    return args


# opencv自带的人脸检测器存在着较为严重的漏检和误检的情况；
# 而dlib的人脸检测器对于较大的人脸的检测效果比较差；
# dlib的cnn检测器效果还可以。
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


def faceCropAndFilter(input_dir, output_dir, image, detector, threshold=55):
    '''
    Input an image from input directory, detect and crop face with margin,
    save face image into output directory.
    从文件夹中输入一张图片，检测人脸并将人脸×scale大小的区域剪裁保存到输出文件夹中。
    :param input_dir: input directory where to input image
    :param output_dir: output directory where to output image
    :param image: image file
    :param detector: dlib fromtal face detector od dlib cnn face detector
    :param threshold: threshold to discard small faces
    :return None:
    '''
    image_prefix = image.split('.')[0]
    image = os.path.join(input_dir, image) # image file
    img = cv2.imread(image) # face ndarray
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_img)
    if faces == None:
        return
    # save faces in for loop
    index = 0
    for face in faces:
        xmin, ymin, xmax, ymax = face
        if (xmax - xmin) < threshold or (ymax - ymin) < threshold:
            continue # small face filter, not save those faces
        face_img = faceCrop(img, xmin, ymin, xmax, ymax)
        out_image = '%s_%d.jpg' % (image_prefix, index)
        out_image = os.path.join(output_dir, out_image)
        cv2.imwrite(out_image, face_img)
        index += 1


def facesCropAndFilter(input_dir, output_dir, detector, threshold, threads=1, thread_index=0):
    '''
    Input many movie image directories, detect face, filter face, crop face and save face.
    对一系列文件夹中的图片进行人脸检测、过滤和剪裁处理。
    :param input_dir: input directory where to input movie images
    :param output_dir: output directory where to output movie images
    :param detector: dlib fromtal face detector od dlib cnn face detector
    :param threshold: threshold to discard small faces
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
        _input_dir = os.path.join(input_dir, movie)
        _output_dir = os.path.join(output_dir, movie)
        if not os.path.exists(_output_dir):
            os.makedirs(_output_dir)
        else:
            continue
        
        image_list = os.listdir(_input_dir)
        for image in image_list:
            try:
                faceCropAndFilter(_input_dir, _output_dir, image, detector, threshold)
            except Exception as e:
                print('Something went wrong when detecting face in image ', _input_dir, image)
                print('And the error is ', e)
        print('Images of movie %s are processed.' % movie)
    print('Thread %d end.' % thread_index)


def mutilThreadFaceProcess(input_dir, output_dir, detector, threshold, threads):
    '''
    mutil process fuction which invoke facesCropAndFilter() function.
    调用单进程函数的多进程检测函数。
    :param input_dir: the root input directory where to input movie images
    :param output_dir: the root input directory where to save movie images
    :param detector: dlib fromtal face detector or dlib cnn face detector
    :param threshold: threshold to discard small faces
    :param threads: thread numbers
    :return None:
    '''
    pool = multiprocessing.Pool(processes=threads)
    for thread_index in range(threads):
        pool.apply_async(facesCropAndFilter, (input_dir, output_dir, detector, 
                                              threshold, threads, thread_index))

    pool.close()
    pool.join()


if __name__ == '__main__':
    args = parseArgs()
    input_dir = args.input_dir
    output_dir = args.output_dir
    if args.face_detector == 'dlib':
        detector = dlibFrontalFaceDetect
    elif args.face_detector == 'dlib_cnn':
        detector = dlibCNNFaceDetect
    else:
        print('You should choose the model in dlib and dlib_cnn.')
        raise 'Wrong model Error.' 
    threshold = args.threshold
    threads = args.threads
    mutilThreadFaceProcess(input_dir, output_dir, detector, threshold, threads)
    
