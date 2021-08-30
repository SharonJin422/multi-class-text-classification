import os
import sys
import json
import time
import warnings
import logging
from process import load_dataset
import numpy as np
import pdb, joblib
import tensorflow as tf
# from models.TextCNN import TextCNN
from models.TextCNN2 import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train_cnn():
	""" Step 0: training parameters """
	params = json.loads(open('parameters.json').read())
	max_sequence_length = params['max_seq_len']
	"""Step 1: pad each sentence to the same length and map each word to an id"""
	print("=======================loading train dataset===================")
	textprocessing, x_raw, y_raw = load_dataset("./data/train_data.p", loadflag=True, isTrain=True, max_sequence_len = max_sequence_length)
	logging.info("The maximum length of all sentences: {}".format(max_sequence_length))
	print("======================training dataset len:{}, label len: {}================".format(len(x_raw), len(y_raw)))

	# 根据所有已分词好的文本建立好一个词典，然后找出每个词在词典中对应的索引，不足长度或者不存在的词补0
	# vocab = joblib.load('./dictionary/word2idx_all.p')
	# vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length)
	# x = np.array(list(vocab_processor.fit_transform(x_raw))) # 转换为索引
	x = x_raw
	y = np.array(y_raw)

	vocab_size=len(textprocessing.vocaburary2id) + 2 # extra len for 'padding' and 'unkown' words
	print("vacab_size:" ,vocab_size)

	"""Step 2: split the original dataset into train and test sets"""
	x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

	"""Step 3: shuffle the train set and split the train set into train and dev sets"""
	shuffle_indices = np.random.permutation(np.arange(len(y_)))
	x_shuffled = np.array(x_)[shuffle_indices]
	y_shuffled = np.array(y_)[shuffle_indices]
	x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)
	# x_raw_test = load_dataset("./data/test_data.p", loadflag=False, isTrain=False)
	print("======================test dataset len:{}================".format(len(x_test)))
	logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
	logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

	"""Step 4: build a graph and cnn object"""
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			textCNN = TextCNN(filter_sizes = list(map(int, params['filter_sizes'].split(","))), num_filters = params['num_filters'],
							num_classes = params['num_classes'],
							learning_rate = 1e-3, batch_size = params['batch_size'], decay_steps = 2,
                          	decay_rate = 1.0 ,sequence_length = max_sequence_length,vocab_size = vocab_size,embed_size = params['embedding_dim'],
						  	usePretrainEmbeddings=params['use_embeddings'])

			out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model/"))
			print("out_dir:", out_dir)
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())

			"""Step 5: train the cnn model with x_train and y_train (batch by batch)"""
			curr_epoch = sess.run(textCNN.epoch_step)
			number_of_training_data = len(x_train)
			batch_size = params['batch_size']
			iteration = 0
			for epoch in range(curr_epoch, params['num_epochs']):
				loss, counter = 0.0, 0
				for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
					iteration = iteration + 1
					if epoch == 0 and counter == 0:
						print("trainX[start]:", x_train[start])

					feed_dict = {textCNN.input_x:x_train[start:end], textCNN.dropout_keep_prob:params['dropout_keep_prob'], textCNN.is_training_flag:True}
					feed_dict[textCNN.input_y] = y_train[start:end]
					curr_loss, lr, _ = sess.run([textCNN.loss_val, textCNN.learning_rate, textCNN.train_op], feed_dict)
					loss, counter = loss + curr_loss, counter + 1
					if counter % 500 == 0:
						print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f\t" % (
						epoch, counter, loss / float(counter), lr))

					########################################################################################################
					if start % (1000 * params['batch_size']) == 0:  # eval every 1000 steps.
						eval_loss, acc,  = do_eval(sess, textCNN, x_dev, y_dev)
						print("Epoch:{}, Validation Loss:{}, acc:{}".format(epoch, eval_loss, acc))
						# save model to checkpoint
						save_path = out_dir + "/model.ckpt"
						print("Going to save model,save path=", save_path + "-"+ str(epoch))
						saver.save(sess, save_path, global_step=epoch)
				########################################################################################################
			test_loss, test_acc = do_eval(sess, textCNN, x_test, y_test)
			print("Epoch:{}, Test Loss:{}, acc:{} ".format(epoch, test_loss, test_acc))

def do_eval(sess, textCNN, evalX, evalY):
	evalX = evalX[0:3000]
	evalY = evalY[0:3000]
	number_examples = len(evalX)
	eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0, 0.0, 0.0, 0.0
	batch_size = 1
	predict = []

	for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples + batch_size, batch_size)):
		''' evaluation in one batch '''
		feed_dict = {textCNN.input_x: evalX[start:end],
                     textCNN.input_y: np.array(evalY[start:end]),
                     textCNN.dropout_keep_prob: 1.0,
                     textCNN.is_training_flag: False}
		current_eval_loss, logits = sess.run(
            [textCNN.loss_val, textCNN.logits], feed_dict)

		predict = [*predict, np.argmax(np.array(logits[0]))]
		eval_loss += current_eval_loss
		eval_counter += 1
	y_true = [int(i) for i in evalY]
	correct_predictions = 0
	for i in range(number_examples):
		if y_true[i] == predict[i]:
			correct_predictions += 1
	print("predict:", predict[:10])
	print("gt:", y_true[:10])
	accuracy = correct_predictions / number_examples
	return eval_loss / float(eval_counter), accuracy



if __name__ == '__main__':
	# python3 train.py
	train_cnn()
