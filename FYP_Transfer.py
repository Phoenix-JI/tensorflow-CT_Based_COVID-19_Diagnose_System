
import numpy as np

import tensorflow as tf

from tensorflow import keras

X_train = np.load("X_train.npy")
print(X_train.shape)


X_test = np.load("X_test.npy")


y_train = np.load("y_train.npy")
print(y_train.shape)


y_test  = np.load("y_test.npy")


INPUT_shape = (128,128,64,1)

inputs = keras.layers.Input(shape=INPUT_shape)

cov1 = keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
pool1 = keras.layers.MaxPool3D(pool_size=2)(cov1)
norm1 = keras.layers.BatchNormalization()(pool1)

cov2 = keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(norm1)
pool2 = keras.layers.MaxPool3D(pool_size=2)(cov2)
norm2 = keras.layers.BatchNormalization()(pool2)

cov3 = keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(norm2)
pool3 = keras.layers.MaxPool3D(pool_size=2)(cov3)
norm3 = keras.layers.BatchNormalization()(pool3)

cov4 = keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(norm3)
pool4 = keras.layers.MaxPool3D(pool_size=2)(cov4)
norm4 = keras.layers.BatchNormalization()(pool4)

pool5 = keras.layers.GlobalAveragePooling3D()(norm4)
hidden = keras.layers.Dense(units=512, activation="relu")(pool5)
drop = keras.layers.Dropout(0.3)(hidden)

outputs = keras.layers.Dense(units=1, activation="sigmoid")(drop)
    
model = keras.Model(inputs, outputs, name="3dcnn")


model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])
print(model.summary())


epochs = 20
with tf.device('/physical_device:XLA_GPU:0'):
    model.fit(
        X_train,
        y_train,
        batch_size = 10,
        epochs=epochs,
        shuffle=False,
        verbose=2,
        validation_data=(X_test,y_test),

    )

