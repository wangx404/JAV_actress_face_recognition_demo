## Script Descriptions

Scripts in this folder is used to help clean noise in images.

There are three scripts in this part. The script `face_crop_and_size_filter.py` is used to crop face with margin from image and discard those small faces detected. The script `man_face_and_no_face_filter.py` is using CNN classificatyion model to filter images with no face and faces which belong to male actors. The script `face_align.py` is used to align faces and crop faces with special width/height ratio.

You should run these scripts in order, otherwise there would be some unexpected results.

### face_crop_and_size_filter

#### Background

Raw images we have downloaded are usually large and contain unuseful parts. So in this script, we crop faces from images with margin. For example, if the face size in face detection result is (100,100), we would crop and save an image with a size of about (200, 200). The margin around the face helps us avoid losing information duiring face alignment process.

In addition, we would discard those small faces which are smaller than the threshold size. In many papers, it has been found out that small faces (low resolution) will lead to low accuracy. So we would like to discard those images in order to achieve high accuracy. If you don't collect enough images, you can set a small threshold ro keep more small images.

#### Usage Description

You can run this script using command like `python face_crop_and_size_filter.py --input_dir dir --output_dir dir --model str --threshold int --threads int`. The option descriptions are shown below.

- input_dir: input image directory

- output_dir: output image directory to save images after processed

- model: choose model to detect face. There are two options here, dlib and dlib_cnn. Dlib uses fromtal face detector in dlib, and its speed is fast with a nice accuracy. Dlib_cnn uses cnn face detector in dlib, and its speed is very slow, but its accuracy is also very high. So you can choose anyone according to your computer.

- threshold: face filter threshold. Face size smaller than threshold will be discarded.

- threads: thread number to run the script. As dlib_cnn runs at a very slow speed, a multi thread/process function is necessary to save time. You can set threads equal to the kernel number of your cpu to speed up.

#### Notice

Before using, you should install dlib and download `mmod_human_face_detector.dat` from dlib website. Then put the file in data directory, otherwise there would be a `file not found` error.

### man_face_and_no_face_filter

#### Background

Though what we really want are images of those beautiful actresses, we still downloaded some male actors' images. And this is quite normal because an adult video always has both actress and actor. These male faces are noise in our dataset.In addition, the face detection model in dlib is not always right especially when you use dlib frontal face detector. To remove all these male-face images and images without faces, I choose to train a CNN classification model.

You can run this script with command `python man_face_and_no_face_filter.py --threshold float`. The threshold is the probability threshold. If the third class (male face image) or the forth class (no face image) probability bigger than the threshold, the script with delete corresponding image. Of course you can change the script to change its behavior. For example, you could move the image to another directory if you don't want to remove them directly.

#### Notice
Before usage, you should download or train a classfication model. I have uploaded a model in google drive, you can download it in [this link](https://drive.google.com/open?id=1y8Nz45jZt9K8QxaSE_XM66o-JkAGwYuk). With this model you can predict four image classes. 0 is good face image; 1 is bad face image, 2 is male actor image, 3 is no face image. The training log shows a pretty high accuracy for the latter two classes. And to separate good faces from bad faces, I'm still looking for a good ans simple method. After downloading the model, you should put the model params in data directory.

### face_align

#### Background

After clearning with the above two scripts, we already have a good dataset. But to reach state-of-the-art accuracy, we need to align the face to the same position and normalize the image size with a width/height ratio 7:8, especially our dataset is mainly composed of asian women. 

