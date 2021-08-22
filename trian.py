import os
import sys
import json
import time
import warnings
import logging
from process import load_dataset, batch_iter
import numpy as np
import pdb
import tensorflow as tf
from TextCNN import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings('ignore')

def train_cnn():
	"""Step 0: training parameters"""
	params = json.loads(open('parameters.json').read())

	"""Step 1: pad each sentence to the same length and map each word to an id"""
	print("=======================loading train dataset===================")
	x_raw, y_raw = load_dataset("./data/train_data.p", loadflag=False, isTrain=True)
	max_sequence_length = max([len(x) for x in x_raw])
	logging.info('The maximum length of all sentences: {}'.format(max_sequence_length))
	print("======================training dataset len:{}================".format(len(x_raw)))

	# 根据所有已分词好的文本建立好一个词典，然后找出每个词在词典中对应的索引，不足长度或者不存在的词补0
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_sequence_length)
	# vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length)
	x = np.array(list(vocab_processor.fit_transform(x_raw))) # 转换为索引
	y = np.array(y_raw)

	vocab_size=len(vocab_processor.vocabulary_)
	print("vacab_size:" ,vocab_size)

	"""Step 2: split the original dataset into train and test sets"""
	x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

	"""Step 3: shuffle the train set and split the train set into train and dev sets"""
	shuffle_indices = np.random.permutation(np.arange(len(y_)))
	x_shuffled = x_[shuffle_indices]
	y_shuffled = y_[shuffle_indices]
	x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)
	# x_raw_test = load_dataset("./data/test_data.p", loadflag=False, isTrain=False)
	print("======================test dataset len:{}================".format(len(x_test)))


	# """Step 4: save the labels into labels.json since predict.py needs it"""
	# with open('./labels.json', 'w') as outfile:
	# 	json.dump(labels, outfile, indent=4)

	logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
	logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

	"""Step 5: build a graph and cnn object"""
	pdb.set_trace()
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = TextCNN(
				sequence_length=x_train.shape[1],
				num_classes=params['num_classes'],
				vocab_size=len(vocab_processor.vocabulary_),
				embedding_size=params['embedding_dim'],
				filter_sizes=list(map(int, params['filter_sizes'].split(","))),
				num_filters=params['num_filters'],
				l2_reg_lambda=params['l2_reg_lambda'])

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model"))

			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver()
			# Save the word_to_id map since predict.py needs it
			# vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
			sess.run(tf.global_variables_initializer())

			# Training starts here
			train_batches = batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
			best_accuracy, best_at_step = 0, 0

			"""Step 6: train the cnn model with x_train and y_train (batch by batch)"""
			for train_batch in train_batches:
				x_train_batch, y_train_batch = zip(*train_batch)
				feed_dict = {cnn.input_x: x_train_batch,cnn.input_y: y_train_batch,cnn.dropout_keep_prob: params['dropout_keep_prob']}
				_, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
				current_step = tf.train.global_step(sess, global_step)

				"""Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
				if current_step % params['evaluate_every'] == 0:
					dev_batches = batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
					total_dev_correct = 0
					for dev_batch in dev_batches:
						x_dev_batch, y_dev_batch = zip(*dev_batch)
						feed_dict = {cnn.input_x: x_dev_batch, cnn.input_y: y_dev_batch, cnn.dropout_keep_prob: 1.0}
						step, loss, acc, num_dev_correct = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct], feed_dict)
						total_dev_correct += num_dev_correct

					dev_accuracy = float(total_dev_correct) / len(y_dev)
					logging.info('Accuracy on dev set: {}'.format(dev_accuracy))

					"""Step 6.2: save the model if it is the best based on accuracy on dev set"""
					if dev_accuracy >= best_accuracy:
						best_accuracy, best_at_step = dev_accuracy, current_step
						path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
						logging.critical('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))

			"""Step 7: predict x_test (batch by batch)"""
			test_batches = batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
			total_test_correct = 0
			for test_batch in test_batches:
				x_test_batch, y_test_batch = zip(*test_batch)
				feed_dict = {cnn.input_x: x_test_batch, cnn.input_y: y_test_batch, cnn.dropout_keep_prob: 1.0}
				step, loss, acc, num_test_correct = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct], feed_dict)
				total_test_correct += num_test_correct

			test_accuracy = float(total_test_correct) / len(y_test)
			logging.critical('Accuracy on test set is {} based on the best model'.format(test_accuracy))
			logging.critical('The training is complete')

if __name__ == '__main__':
	# python3 train.py
	train_cnn()
