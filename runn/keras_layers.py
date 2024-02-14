import tensorflow as tf


class Gather(tf.keras.layers.Layer):
    """Custom layer to gather the elements of a tensor based on an index tensor.

    Args:
        indices: A tensor with the indices of the elements to gather.
        axis: The axis along which to gather the elements. Default: 1.
    """

    def __init__(self, indices: tf.Tensor, axis: int = 1, **kwargs):
        super(Gather, self).__init__(**kwargs)
        self.indices = indices
        self.axis = axis

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Perform the gather operation
        return tf.gather(inputs, indices=self.indices, axis=self.axis)

    def get_config(self) -> dict:
        config = super(Gather, self).get_config()
        config.update({"indices": self.indices, "axis": self.axis})
        return config
