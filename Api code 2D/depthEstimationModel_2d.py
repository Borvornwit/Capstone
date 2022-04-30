import tensorflow as tf
from tensorflow.keras import layers

class Convolutional(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), strides=1, padding="same", **kwargs
    ):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, strides, padding, activation="elu")
        self.bn = layers.BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Pooling(layers.Layer):
    def __init__(
        self, **kwargs
    ):
        super().__init__(**kwargs)
        self.pool = layers.MaxPool2D((2, 2), (2, 2))
        self.resize = layers.Resizing(64, 64)

    def call(self, x):
        x = self.pool(x)
        x = self.resize(x)
        return x

class DepthEstimationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.convolutional = [
            Convolutional(64),

            Convolutional(128),
            Convolutional(196),
            Convolutional(128),

            Convolutional(128),
            Convolutional(196),
            Convolutional(128),

            Convolutional(128),
            Convolutional(196),
            Convolutional(128),

            Convolutional(128),
            Convolutional(64),
        ]
        self.pooling = [
            Pooling(),
            Pooling(),
            Pooling(),
        ]
        self.conc = layers.Concatenate()
        self.conv_layer = layers.Conv2D(1, (1, 1), activation="elu")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='elu')
        self.dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, x):

        c0 = self.convolutional[0](x)

        c1 = self.convolutional[1](c0)
        c2 = self.convolutional[2](c1)
        c3 = self.convolutional[3](c2)
        p1 = self.pooling[0](c3)

        c4 = self.convolutional[4](p1)
        c5 = self.convolutional[5](c4)
        c6 = self.convolutional[6](c5)
        p2 = self.pooling[1](c6)

        c7 = self.convolutional[7](p2)
        c8 = self.convolutional[8](c7)
        c9 = self.convolutional[9](c8)
        p3 = self.pooling[2](c9)

        f = self.conc([p1, p2, p3])

        c10 = self.convolutional[10](f)
        c11 = self.convolutional[11](c10)
        c12 = self.conv_layer(c11)

        fl = self.flatten(c12)
        d1 = self.dense1(fl)
        d2 = self.dense2(d1)

        return [d2, c12]
