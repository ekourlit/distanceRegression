import tensorflow as tf

def getSimpleMLP(hiddenNodes, input_shape=6, **settings):
	'''
	Construct a fully connected MLP.
	return
		model: keras model
	'''
	
	# construct
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(hiddenNodes[0], input_shape=(input_shape,), activation=settings['activation']))
	for layer in range(1, len(hiddenNodes)):
		model.add(tf.keras.layers.Dense(hiddenNodes[layer], activation=settings['activation']))
	model.add(tf.keras.layers.Dense(1, activation=settings['output_activation']))
	# check kernel_initializer='he_uniform'. it doesn't learn.

	# compile
	model.compile(
		loss=settings['loss'],
		optimizer=settings['optimizer'],
		metrics=['mae'])

	# print model summary
	model.summary()
	
	return model