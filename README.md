# tensorflow_vgg16_image_classification
VGG16 based classifier has been trained for oxfordflower102 dataset. A pre-trained vgg16 model for imagenet dataset has been utilized to initializ the weights.
vgg16:  http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz

##initially resize the dataset images to 225X225 to minimize computation requirement

run resize cell script in label_prepare.ipynb

## for training
python train.py --is-training

## for evaluation
python evaluate.py


##Results
Training data: 7000 images
Testing data: 1189 images
mean classification accuracy for test data of 67.95% has been achieved after training for 1 epoch for 102 classes.
mean classification accuracy for test data of 80.99% has been achieved after training for 2 epochs for 102 classes.
mean classification accuracy for test data of 87.55% has been achieved after training for 3 epochs for 102 classes.
mean classification accuracy for test data of 88.89% has been achieved after training for 4 epochs for 102 classes.
mean classification accuracy for test data of 91.08% has been achieved after training for 5 epochs for 102 classes.
mean classification accuracy for test data of 91.16% has been achieved after training for 6 epochs for 102 classes.
mean classification accuracy for test data of 91.33% has been achieved after training for 7 epochs for 102 classes.
