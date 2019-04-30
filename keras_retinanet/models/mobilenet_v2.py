"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from keras.applications import mobilenet_v2
from keras.utils import get_file
from ..utils.image import preprocess_image

from . import retinanet
from . import Backbone


class MobileNetV2Backbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    allowed_backbones = ['mobilenet']

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return mobilenet2_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Download pre-trained weights for the specified backbone name.
        This name is in the format mobilenet{rows}_{alpha} where rows is the
        imagenet shape dimension and 'alpha' controls the width of the network.
        For more info check the explanation from the keras mobilenet script itself.
        """

        rows_alpha = self.backbone.split('mobilenet_v2_')[1]
        alpha = float(rows_alpha.split('_')[1])
        rows = int(rows_alpha.split('_')[0])

        # load weights
        if keras.backend.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1.0'

        model_name = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{}_{}_no_top.h5'.format(alpha_text, rows)
        weights_url = mobilenet_v2.mobilenet_v2.BASE_WEIGHT_PATH + model_name
        weights_path = get_file(model_name, weights_url, cache_subdir='models')

        return weights_path

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        backbone = self.backbone.split('_')[0]

        if backbone not in MobileNetV2Backbone.allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, MobileNetV2Backbone.allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')


def mobilenet2_retinanet(num_classes, backbone='mobilenet_v2_224_1.0', inputs=None, modifier=None, size=None, **kwargs):
    """ Constructs a retinanet model using a mobilenet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('mobilenet_v2_224_1.0')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
        size: without this the model will not converge

    Returns
        RetinaNet model with a MobileNet backbone.
    """
    rows_alpha = backbone.split('mobilenet_v2_')[1]
    alpha = float(rows_alpha.split('_')[1])
    rows = int(rows_alpha.split('_')[0])

    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, size, size))
        else:
            inputs = keras.layers.Input(shape=(size, size, 3))

    if modifier is not None:
        print("Having modifier:" + str(modifier))

    backbone = mobilenet_v2.MobileNetV2(input_tensor=inputs, alpha=alpha, include_top=False, pooling=None, weights='imagenet')

    # create the full model
    #layer_names = ['block_5_depthwise_relu', 'block_12_depthwise_relu', 'out_relu']
    layer_names = ['block_5_add', 'block_12_add', 'out_relu']
    layer_outputs = [backbone.get_layer(name).output for name in layer_names]
    backbone = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=backbone.name)

    # invoke modifier if given
    if modifier:
        backbone = modifier(backbone)

    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=backbone.outputs, **kwargs)
