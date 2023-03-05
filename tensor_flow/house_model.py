import tensorflow as tf
import numpy as np

from tensorflow import keras

def house_model():
	xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
	ys = np.array([1, 1.5, 2, 2.5, 3, 3.5], dtype=float)

	# Define a new model
	model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

	# Compiles the model
	model.compile(optimizer='sgd', loss='mean_squared_error')
	
	# Train the model
	model.fit(xs, ys, epochs=1000)

	return model

model = house_model()

new_x = 7.0
prediction = model.predict([new_x])[0]
print(prediction)
