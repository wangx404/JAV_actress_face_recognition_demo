# PROBLEMS

Taken such a long time to process these images, finally I have only about 20K clean faces and 80K faces with noise (about 2000 actresses). Maybe using other dataset which is much bigger to train a model is easier. After this, using this model to compute the features of actresses will be easy...

Using this script, you can train the CNN model to do the main jod in face recognition.

2018.11.16, this script has no problem to run. But there still exists a serious problem. When training the model, the loss can be nan. Even if the loss is not nan, the accuracy of the model is pretty low. I still have not located the problem. If you know what is the problem, please tell me.

2018.11.29 ready to use face recognition model in dlib to test.

## Background

In this part, I use a simple CNN network structure and center loss to train a face recognition model. Wanting to know more about center loss? You can download Y. Wen's paper [A Discriminative Feature Learning Approach for Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf).

## Script Description

- CNN_model.py: the CNN network structure he center loss structure are stored in this script.

- CNN_utils.py: dataset loader and related functions are stored in this script.

- CNN_train.py: main script to train or test a face recognition model. More details about this script are listed in **Usage**.

## Usage

You can train/test a model like this `python CNN_train.py --train --input_dir ../data/xxx --num_classes 1024 --feature_dim 42`. The options are descripted as below.

- input_dir: face image directory. In this directory, there are two directories named as train and val.

- train: train a model. If you want to test your model, just change it to --val.

- num_classes: number of classes in your own dataset.

- feature_dim: feature dimension.

There are more options that can be set. You can see related description in the script.

## Problems

1.  nan loss. Haven't solved.

2.  accuracy evaluate. A KNN model should be used to evaluate accuracy rather than maximum output probability.

3.  network structure optimization.
