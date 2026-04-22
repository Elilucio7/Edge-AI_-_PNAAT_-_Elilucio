import tensorflow as tf
import os

model = tf.keras.models.load_model("model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_d = converter.convert()
tfl_dyn_path = "model.tflite"
open(tfl_dyn_path, "wb").write(tflite_d)