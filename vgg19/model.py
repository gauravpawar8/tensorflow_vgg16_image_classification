# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from tensorflow_layers import Network
import tensorflow as tf

class VGG19Model(Network):
    def setup(self, is_training, num_classes):
        '''Network definition.
        
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='block1_conv1')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='block1_conv2')
             .max_pool(2, 2, 2, 2, name = 'block1_pool')
             .conv(3, 3, 128, 1, 1, biased=True, relu=True, name='block2_conv1')
             .conv(3, 3, 128, 1, 1, biased=True, relu=True, name='block2_conv2')
             .max_pool(2, 2, 2, 2, name = 'block2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='block3_conv1')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='block3_conv2')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='block3_conv3')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='block3_conv4')
             .max_pool(2, 2, 2, 2, name = 'block3_pool')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='block4_conv1')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='block4_conv2')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='block4_conv3')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='block4_conv4')
             .max_pool(2, 2, 2, 2, name = 'block4_pool')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='block5_conv1')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='block5_conv2')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='block5_conv3')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='block5_conv4')
             .max_pool(2, 2, 2, 2, name = 'block5_pool')
             .fc(4096, name='fc1', relu=True)
             .dropout(0.5, name='dp1')
             .fc(4096, name='fc2', relu=True)
             .dropout(0.5, name='dp2')
             .fc(102, name='fc3', relu=False))
             # .softmax(name='vgg_output'))