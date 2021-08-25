import joblib
import numpy as np
import os, time
import jieba_fast as jieba
import pdb
from collections import defaultdict
from gensim import corpora, similarities, models, matutils, utils
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextProcessing(object):
	def __init__(self, stopwordPath, max_sequence_len):
		self.stopwordPath = stopwordPath
		self.max_sequence_len = max_sequence_len
	def load_stopwords(self):
		stopwords = open(self.stopwordPath, 'r', encoding = 'gbk').read().split('\n')
		print("stopwords counts", len(stopwords))
		print("example of stopword", stopwords[0])
		return stopwords

	def is_chinese(self, uchar):
		if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
			return True
		else:
			return False

	def jieba_tokenize(self, documents):
		print("start to tokenizations, it takes times")
		stoplist = self.load_stopwords()
		corpora_documents = []
		corpora_documents_list = []
		for item_text in documents:
			outstr = []
			sentence_seg = list(jieba.cut(item_text))
			for word in sentence_seg:
				if not self.is_chinese(word):
					continue
				if word not in stoplist and word != '\t' and word != ' ':
					outstr.append(word)
			corpora_documents.append(''.join(word for word in outstr))
			corpora_documents_list.append(outstr)
		print("tokenizations finish!")
		return corpora_documents, corpora_documents_list

	def RemoveWordAppearBelowN(self, corpora_documents, n = 1):
		frequency = defaultdict(int)
		for text in corpora_documents:
			for token in text:
				frequency[token] += 1
		corpora_documents = [[token for token in text if frequency[token] > n] for text in corpora_documents]
		return corpora_documents

	def word2index(self, tokens, vocabulary):
		print("start to convert tokens into index")
		extra_word = {'unknown': len(vocabulary), 'PAD': len(vocabulary) + 1}
		sentence2id = []
		unknown_words = set()
		for sen in tokens:
			idx = []
			for word in sen:
				if word in vocabulary.keys():
					idx.append(vocabulary[word])
				else:
					# print("word unkown:", word)
					unknown_words.add(word)
					idx.append(extra_word['unknown'])
			sentence2id.append(idx)
		print("unknown words count:", len(unknown_words))
		# padding sentences
		for i in range(len(sentence2id)):
			sentence = sentence2id[i]
			if len(sentence) < self.max_sequence_len:
				sentence += [extra_word['PAD']] * (self.max_sequence_len - len(sentence))
			else:
				sentence = sentence[-self.max_sequence_len:]
			sentence2id[i] = sentence

		return sentence2id

	def genDictionary(self, documents, is_train, load=True, **kwarg):
		if is_train:
			name = 'train'
		else:
			name = 'test'

		if load == True:
			# filtered_token = joblib.load('./dictionary/all_'+ name + '_token.p')
			# corpora_documents = joblib.load('./dictionary/all_' + name + '_filtered_text.p')
			sentences2id = joblib.load('./dictionary/sentences2id_'+name + '.p')
			return sentences2id
		corpora_documents, token = self.jieba_tokenize(documents)
		filtered_token = self.RemoveWordAppearBelowN(token, n=1)
		# print('./dictionary/all_' + name + '_token.p')
		# filtered_token = joblib.load('./dictionary/all_' + name + '_token.p')
		if is_train:
			self._dictionary = corpora.Dictionary(filtered_token)  # 生成词典
			token2id = self._dictionary.token2id
			# print(self._dictionary.keys())  #133347 key
			# print(self._dictionary.get(5))
			# print(self._dictionary.dfs) # 单词id: 在多少文档中出现
			# for key in self._dictionary.dfs.keys():
			# 	if self._dictionary.dfs[key] < 2:
			# 		print(self._dictionary.get(key))
			# pdb.set_trace()
		else:
			token2id = joblib.load('./dictionary/word2idx_all.p')


		sentence2id = self.word2index(filtered_token, token2id)
		joblib.dump(filtered_token, './dictionary/all_' + name + '_token.p')
		joblib.dump(corpora_documents, './dictionary/all_' + name + '_filtered_text.p')
		joblib.dump(sentence2id, './dictionary/sentences2id_' + name + '.p')
		if is_train:
			joblib.dump(self._dictionary, './dictionary/idx2word_all.dict')
			joblib.dump(token2id, './dictionary/word2idx_all.p')

		return sentence2id


def load_dataset(filename = 'train_data.p', stopwordPath = './data/cn_stopwords.txt', loadflag=False, isTrain=True, max_sequence_len = 600):
	dataset = joblib.load(filename, 'rb')
	print("is Loading Tokenized dataset:{}, isTrain:{}, len:{}".format(loadflag, isTrain, len(dataset)))
	textprocessing = TextProcessing(stopwordPath, max_sequence_len)
	sentence2id = textprocessing.genDictionary(dataset['text'], load=loadflag, is_train=isTrain) #list of list

	if isTrain:
		dataset['label'] = [int(np.argmax(i)) for i in dataset['label']]
		return sentence2id, dataset['label']
	else:
		return sentence2id


if __name__ == '__main__':
	token, dictionary = load_dataset('./data/train_data.p', './data/cn_stopwords.txt', loadflag=False, isTrain=True, max_sequence_len = 600)
