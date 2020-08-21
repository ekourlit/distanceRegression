import timeit, pdb, json, argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def timeModel(model, iterations=100):
	# retrieve model negPunish
	timestamp = model.strip('.h5').split('model_')[-1]
	logDic = json.load(open('data/modelConfig_'+timestamp+'.json'))
	retValNegPunish = logDic['negPunish'] if 'negPunish' in logDic.keys() else 1

	setup = "import tensorflow as tf; \
			 from models import getNoOverestimateLossFunction, overestimationMetric; \
			 tf.config.threading.set_inter_op_parallelism_threads(0); \
			 import numpy as np; \
			 mlp_model = tf.keras.models.load_model('data/mlp_model_"+str(timestamp)+".h5', custom_objects={'noOverestimateLossFunction':getNoOverestimateLossFunction(negPunish="+str(retValNegPunish)+"),'overestimationMetric':overestimationMetric}); \
			 Input=np.array([0.39866815, 0.01248495, 0.65736947, 0.00752856, 0.72758594, -0.68597523]).reshape(1,6); \
			 Input=tf.constant(Input); \
			 tf.compat.v1.disable_eager_execution(); \
			 mlp_model.predict(Input,steps=1)"

	# I fuck don't know why I need to do this....
	try:
		timeit.timeit(setup=setup, stmt='mlp_model.predict(Input,steps=1)', number=1)
	except ValueError:
		pass

	# real job is happening here
	iterations = iterations
	time_list = timeit.repeat(setup=setup, stmt='mlp_model.predict(Input,steps=1)', number=1, repeat=iterations)

	print("-- Fastest execution took %f ms (min. of %i iterations) --" % ((min(time_list))*1000, iterations))
	print("-- Average execution is %f ms (avg. of %i iterations) --" % ((sum(time_list)/len(time_list))*1000, iterations))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluate trained model inference time.')
	parser.add_argument('model', help='Model to evaluate.', type=str)
	parser.add_argument('--iterations', help='Evaluation iterations.', type=int, required=False, default=100)
	args = parser.parse_args()

	timeModel(args.model, args.iterations)
