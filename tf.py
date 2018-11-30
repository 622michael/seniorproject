import tensorflow as tf
import celeb_data as cd

import cv2
import numpy as np

def main():
    (x_training, y_training), (x_validation, y_validation), _ = cd.load_data_wrapper("/Users/michaelcrabtree/Downloads/celeba-dataset", limit=50, gray_scale=False)

    print(x_training.shape)
    print(y_training.shape)
    opt = tf.keras.optimizers.SGD(lr=0.1)
    # model = tf.keras.models.Sequential([
    #   tf.keras.layers.Conv2D(10, 25, strides=2, dilation_rate=3, activation=tf.nn.relu),
    #   tf.keras.layres.MaxPool2D(2, 2, "same"),
    #   tf.keras.layers.Flatten(),
    #   tf.keras.layers.Dense(100, at)
    #   tf.keras.Dense(1, activation=tf.keras.backend.hard_sigmoid)
    # ])
    model = tf.keras.models.Sequential([
      # tf.keras.layers.Conv2D(96, 5, activation=tf.nn.relu),
      # tf.keras.layers.MaxPool2D(2, 2, "same"),
      # tf.keras.layers.BatchNormalization(), 
      # tf.keras.layers.Conv2D(96, 5, activation=tf.nn.relu),
      # tf.keras.layers.MaxPool2D(2, 2, "same"),
      # tf.keras.layers.BatchNormalization(), 
      tf.keras.layers.Flatten(input_shape=(218, 178, 3)),
      # tf.keras.layers.Dense(512, activation=tf.nn.relu),
      # tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
      tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_training, y_training, epochs=2)
    l = model.evaluate(x_validation, y_validation)
    print(l)


if __name__ == '__main__':
    main()

