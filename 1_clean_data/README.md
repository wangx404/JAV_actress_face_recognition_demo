Scripts in this folder is used to help clean noise in images.

## Script Description

There are three scripts in this part. The script `face_crop_and_size_filter.py` is used to crop face with margin from image and discard those small faces. The script `man_face_and_no_face_filter.py` is used to filter images with no face and faces which belong to male actors. The script `face_align.py` is used to align faces and crop faces with special width/height ratio.

You should run these scripts in order, otherwise there would be some unexpected result.

## Usage Description

Different script has different option. The information is listed below.

### face_crop_and_size_filter

Raw images we have downloaded are usually large and contain unuseful parts. So in this script, we crop faces from images with margin. If the face size in face detection result is (100,100), we would crop and save an image about (200, 200) around the face. The margin around the face helps us avoid losing information duiring face alignment.

In addition, we would discard those small faces less than threshold size. In many papers, it has been found out that small faces will lead to low accuracy. So we discard whose images in order to achieve high accuracy. If you don't collect enough image, you can set a small threshold.

You can run this script like this `python face_crop_and_size_filter.py --input_dir dir --output_dir dir --model str --threshold int`. The option descriptions are shown below.

- input_dir: input image directory

- output_dir: output image directory to save images after processed

- model: choose model to detect face. There are two options here, dlib and dlib_cnn. Dlib uses fromtal face detector in dlib, and its speed is faster with a nice accuracy. Dlib_cnn uses cnn face detector in dlib, and its speed is very slow, but its accuracy is also very high. So you can choose one according to your computer.

- threshold: face filter threshold. Face size smaller than this threshold will be discarded.

### man_face_and_no_face_filter


### face_align
