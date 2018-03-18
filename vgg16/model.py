# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from tensorflow_layers import Network
import tensorflow as tf

class VGG16Model(Network):
    def setup(self, is_training, num_classes):
        '''Network definition.
        
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv1_1')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv1_2')
             .max_pool(2, 2, 2, 2, name = 'maxpool1')
             .conv(3, 3, 128, 1, 1, biased=True, relu=True, name='conv2_1')
             .conv(3, 3, 128, 1, 1, biased=True, relu=True, name='conv2_2')
             .max_pool(2, 2, 2, 2, name = 'maxpool2')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='conv3_1')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='conv3_2')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='conv3_3')
             .max_pool(2, 2, 2, 2, name = 'maxpool3')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='conv4_1')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='conv4_2')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='conv4_3')
             .max_pool(2, 2, 2, 2, name = 'maxpool4')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='conv5_1')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='conv5_2')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='conv5_3')
             .max_pool(2, 2, 2, 2, name = 'maxpool5')
             .fc(4096, name='fc6', relu=True)
             .dropout(0.5, name='dropout1')
             .fc(4096, name='fc7', relu=True)
             .dropout(0.5, name='dropout2')
             .fc(102, name='fc8', relu=False))
             