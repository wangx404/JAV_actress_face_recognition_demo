Scripts in this folder is used to help clean noise in images.

## Script Description

There are three scripts in this part. The script `face_crop_and_size_filter.py` is used to crop face with margin from image and discard those small faces detected. The script `man_face_and_no_face_filter.py` is using CNN classificatyion model to filter images with no face and faces which belong to male actors. The script `face_align.py` is used to align faces and crop faces with special width/height ratio.

You should run these scripts in order, otherwise there would be some unexpected results.

## Usage Description

Different script has different option. Related information is listed below.

### face_crop_and_size_filter

Raw images we have downloaded are usually large and contain unuseful parts. So in this script, we crop faces from images with margin. For example, if the face size in face detection result is (100,100), we would crop and save an image with a size of about (200, 200). The margin around the face helps us avoid losing information duiring face alignment process.

In addition, we would discard those small faces which are smaller than the threshold size. In many papers, it has been found out that small faces (low resolution) will lead to low accuracy. So we would like to discard those images in order to achieve high accuracy. If you don't collect enough images, you can set a small threshold ro keep more small images.

You can run this script using command like `python face_crop_and_size_filter.py --input_dir dir --output_dir dir --model str --threshold int --threads int`. The option descriptions are shown below.

- input_dir: input image directory

- output_dir: output image directory to save images after processed

- model: choose model to detect face. There are two options here, dlib and dlib_cnn. Dlib uses fromtal face detector in dlib, and its speed is fast with a nice accuracy. Dlib_cnn uses cnn face detector in dlib, and its speed is very slow, but its accuracy is also very high. So you can choose anyone according to your computer.

- threshold: face filter threshold. Face size smaller than threshold will be discarded.

- threads: thread number to run the script. As dlib_cnn runs at a very slow speed, a multi thread/process function is necessary to save time. You can set threads equal to the kernel number of your cpu to speed up.

**Before using, you should install dlib and download `mmod_human_face_detector.dat` from dlib website. Then put the file in data directory, otherwise there would be a `file not found` error.**

### man_face_and_no_face_filter



**Before using, you should download or train a classfication model.**
### face_align
