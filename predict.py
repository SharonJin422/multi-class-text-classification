import joblib,json, os,pdb
import jieba_fast as jieba
import numpy as np
from process import load_dataset
from models.TextCNN2 import TextCNN
from evaluation import f1_avg
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
word2id = joblib.load('./dictionary/word2idx_all.p')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph=tf.Graph().as_default()

def predict():
	params = json.loads(open('parameters.json').read())
	max_sequence_length = params['max_seq_len']
	batch_size = params['batch_size']
	vocab_size = len(word2id)
	textCNN = TextCNN(filter_sizes=list(map(int, params['filter_sizes'].split(","))), num_filters=params['num_filters'],
					  num_classes=params['num_classes'],
					  learning_rate=1e-3, batch_size=batch_size, decay_steps=50,
					  decay_rate=1.0, sequence_length=max_sequence_length, vocab_size=vocab_size,
					  embed_size=params['embedding_dim'],
					  usePretrainEmbeddings=params['use_embeddings'],
					  )
	graph = tf.Graph()
	with graph.as_default():
		x_test = load_dataset(filename='./data/test_data.p', loadflag=True, isTrain=False,
							  max_sequence_len=max_sequence_length)

	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=session_conf)
	y_pred = []
	with sess.as_default():
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=2000)
		ckpt_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model/"))
		print(ckpt_dir, os.path.exists(ckpt_dir))

		if os.path.exists(ckpt_dir):
			print("Restoring Variables from Checkpoint")
			ckpt_path = ckpt_dir + '/model.ckpt-49'
			saver.restore(sess, ckpt_path)
		else:
			print("Can't find the checkpoint.going to stop")

		number_of_test_data = len(x_test)
		print("number_of_test_data:", number_of_test_data)

		for start, end in zip(range(0, number_of_test_data, batch_size), range(batch_size, number_of_test_data, batch_size)):
			logits, predictions = sess.run([textCNN.logits,textCNN.predictions], feed_dict={textCNN.input_x: x_test[start:end],
														 textCNN.dropout_keep_prob: 1,
														 textCNN.is_training_flag:False})
			y_pred += [np.argmax(i) for i in logits]
		
		pdb.set_trace()


	return y_pred
if __name__ == '__main__':
	y_pred = predict()
	print("y_pred", len(y_pred))

	# accuray = f1_avg(y_pred, y_true)
	# print("evaluation accuray", accuray)
