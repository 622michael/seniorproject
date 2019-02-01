import tensorflow as tf
import adult_data as ad 


model = tf.keras.models.load_model("current-model")
(x_training, y_training), (x_validation, y_validation), _ = ad.load_data_wrapper("/Users/michaelcrabtree/Downloads/Adult", limit=50, gray_scale=False)
model.fit(x_training, y_training, epochs=2)