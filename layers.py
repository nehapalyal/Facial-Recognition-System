# Custom L1 distance layer module to load custom model

# import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer


# custom L1 layer from jupyter 
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    


