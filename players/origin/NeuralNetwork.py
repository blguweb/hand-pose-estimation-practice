import tensorflow as tf
import numpy as np
import os
import argparse
import shutil
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from tqdm import tqdm

from ReadData import read_data_files, split_data

N_KEYPOINTS = 63

def parse_args():
	parser = argparse.ArgumentParser(description = 'Train neural network')
	# You have to give individual csv files based on the order in which position id of poses are made.
	parser.add_argument('csv_files', help = 'Comma separated list of paths of training files', type = str)
	parser.add_argument('--output-path', dest = 'output_path', type = str, default = None,
						help = 'Path of folder where to store the trained model')
	parser.add_argument('--train-test-split', dest = 'train_test_split', default = 0.85, type = float,
					    help = 'Ratio of train to test dataset(0-1)')
	parser.add_argument('--learning-rate', dest = 'learning_rate', default = 0.0001, type = float,
					    help = 'Learning rate of the model')
	parser.add_argument('--epochs', dest = 'epochs', type = int, default = 30,
						help = 'Number of epochs to train data on')
	parser.add_argument('--batch-size', dest = 'batch_size', type = int, default = 64,
						help = 'Batch size while training')
	parser.add_argument('--max-samples', dest = 'max_samples', type = int, default = 10000,
						help = 'Maximum number of samples per class allowed')
	parser.add_argument('--break-training-at', dest = 'break_training_at', default = 0.96, type = float,
					    help = 'Break out of training when test accuracy increases than')
	args = parser.parse_args()
	return args

def fc_layer(X, weight, bias, should_activate = True):
    fc = tf.nn.xw_plus_b(X, weight, bias)
    if should_activate:
        fc = tf.nn.relu(fc)
    return fc

def pose_classify_net(X, n_classes):
    with tf.variable_scope('Layer1'):
        weight = tf.get_variable('weight1', initializer = tf.truncated_normal((N_KEYPOINTS, 22), mean = 0.0, stddev = 0.1))
        bias = tf.get_variable('bias1', initializer = tf.ones((22)))
        fc1 = fc_layer(X, weight, bias)
    
    with tf.variable_scope('Layer2'):
        weight = tf.get_variable('weight2', initializer = tf.truncated_normal((22, n_classes), mean = 0.0, stddev = 0.1))
        bias = tf.get_variable('bias2', initializer = tf.ones((n_classes)))
        fc2 = fc_layer(fc1, weight, bias, should_activate = False)        
        
    return fc2

def evaluate(X_data, y_data, accuracy, X, y, batch_size):
    sum_at = 0.0
    for offset in range(0, len(y_data), batch_size):
        end = offset + batch_size
        batch_X, batch_y = X_data[offset: end], y_data[offset: end]
        
        eval_sess = tf.get_default_session()
        acc = eval_sess.run(accuracy, feed_dict = {X: batch_X, y: batch_y})
        sum_at += acc * len(batch_y)
    return sum_at / len(y_data)

def build_and_train_network(train_data, test_data, parameters, output_at):
	X_train, y_train = train_data
	X_test, y_test = test_data

	tf.reset_default_graph()
	X = tf.placeholder(tf.float32, shape = (None, N_KEYPOINTS))
	y = tf.placeholder(tf.int32, shape = (None))
	count = tf.get_variable('count', shape = [], dtype = tf.int32, trainable = False)
	one_hot_y = tf.one_hot(y, parameters['n_classes'])

	logits = pose_classify_net(X, parameters['n_classes'])
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_y, logits = logits))
	optimizer = tf.train.AdamOptimizer(learning_rate = parameters['learning_rate']).minimize(loss_op, global_step = count)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(one_hot_y, 1), tf.argmax(logits, 1)), tf.float32))

	saver = tf.train.Saver()
	batch_size = parameters['batch_size']

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		train_accuracies = []
		test_accuracies = []
		
		for epoch_no in range(parameters['epochs']):
			X_train, y_train = shuffle(X_train, y_train)
			
			with tqdm(total = len(y_train)) as pbar:
				for offset in range(0, len(y_train), batch_size):
					end = offset + batch_size
					batch_X, batch_y = X_train[offset: end], y_train[offset: end]

					sess.run(optimizer, feed_dict = {X: batch_X, y: batch_y})
					pbar.update(len(batch_y))
			# print("2",parameters)
			train_acc = evaluate(X_train, y_train, accuracy, X, y, batch_size)
			test_acc = evaluate(X_test, y_test, accuracy, X, y, batch_size)
			train_accuracies.append(train_acc)
			test_accuracies.append(test_acc)
			
			print('Epoch: {}'.format(epoch_no + 1))
			print('Train accuracy: {:.3f}, Test accuracy: {:.3f}'.format(train_acc, test_acc))

			if test_acc >= parameters['break_training_at']:
				print('Breaking out of training')
				break
		
		shutil.rmtree(output_at, ignore_errors = True)
		os.mkdir(output_at)
		saver.save(sess, '{}/Checkpoint'.format(output_at))
		print('Saved checkpoint')

		plt.plot(range(len(train_accuracies)), train_accuracies, 'b-', label = 'Train')
		plt.plot(range(len(test_accuracies)), test_accuracies, 'r-', label = 'Test')
		plt.savefig('{}/accuracy.png'.format(output_at))

def get_value_by_name(name):
    tensor = tf.get_default_graph().get_tensor_by_name('{}:0'.format(name))
    sess = tf.get_default_session()
    val = sess.run(tensor)
    reqd_tensor = tf.constant(val, name = name)
    return reqd_tensor

def export_created_graph(output_at):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint(output_at))

		g = tf.get_default_graph()
		g_x = tf.placeholder(tf.float32, shape = [None, N_KEYPOINTS], name = 'input')

		weight1 = get_value_by_name('Layer1/weight1')
		bias1 = get_value_by_name('Layer1/bias1')
		fc1 = fc_layer(g_x, weight1, bias1)
		
		weight2 = get_value_by_name('Layer2/weight2')
		bias2 = get_value_by_name('Layer2/bias2')
		fc2 = fc_layer(fc1, weight2, bias2, should_activate = False)

		output = tf.nn.softmax(fc2, name = 'output')

		graph_def = g.as_graph_def()
		tf.train.write_graph(graph_def, output_at, 'graph.pb', as_text = False)
		print('Saved model')


if __name__ == '__main__':
	args = parse_args()

	data_files = [os.path.abspath(p.strip()) for p in args.csv_files.split(',')]
	X_data, y_data = read_data_files(data_files, args.max_samples)
	train_data, test_data = split_data(X_data, y_data, args.train_test_split)
	
	# Keep adding parameters here
	parameters = {'batch_size': args.batch_size,
	              'epochs': args.epochs,
				  'learning_rate': args.learning_rate,
				  'n_classes': len(data_files),
				  'break_training_at': args.break_training_at}
	# print("parameters",parameters)
	# If output folder is None, will create an output folder
	# Also, the output folder will be deleted and recreated again.
	output_at = 'Checkpoint' if args.output_path is None else args.output_path
	# print("1",parameters)
	build_and_train_network(train_data, test_data, parameters, output_at)
	export_created_graph(output_at)
