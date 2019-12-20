import keras

class BatchNormalization(keras.layers.BatchNormalization):
    """
    Replace BatchNormalization layers with this new layer.
    This layer has fixed momentum 0.9 so when we are doing
    transfer learning on small dataset the learning is a bit faster.

    Usage:
        keras.layers.BatchNormalization = BatchNormalization

        base_model = keras.applications.MobileNetV2(
            weights="imagenet", input_shape=self.shape, include_top=False, layers=keras.layers
        )
    """

    def __init__(self, momentum=0.9, name=None, **kwargs):
        super(BatchNormalization, self).__init__(momentum=0.9, name=name, **kwargs)

    def call(self, inputs, training=None):
        return super().call(inputs=inputs, training=training)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        return config
