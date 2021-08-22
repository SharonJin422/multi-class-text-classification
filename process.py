import joblib
import numpy as np
import jieba, os
import pdb
from collections import defaultdict
from gensim import corpora,similarities,models,matutils,utils

class TextProcessing(object):
	def __init__(self, stopwordPath):
		self.stopwordPath = stopwordPath

	def load_stopwords(self):
		stopwords = open(self.stopwordPath, 'r').read().split('\n')
		print("stopwords counts", len(stopwords))
		print("example of stopword", stopwords[0])
		return stopwords

	def is_chinese(self, uchar):
		''' remove letters, punctuations, and nums
		# Arguments:
			uchar: single word
		'''
		if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
			return True
		else:
			return False


	def jieba_tokenize(self, documents):
		'''Cut the documents into a sequence of independent words.
		# Arguments:
			documents: List of news(articles).
		'''
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
		return corpora_documents, corpora_documents_list

	def RemoveWordAppearBelowN(self, corpora_documents, n = 1):
		'''Remove the words that appear once among all the tokenized news(articles).
		# Arguments:
			 corpora_documents: List of tokenized news(articles).
		'''
		frequency = defaultdict(int)
		for text in corpora_documents:
			for token in text:
				frequency[token] += 1
		corpora_documents = [[token for token in text if frequency[token] > n] for text in corpora_documents]
		return corpora_documents

	def genDictionary(self, documents, load = False, **kwarg):
		'''Generate dictionary and bow-vector of all tokenzied news(articles).
		# Arguments:
			documents: List of texts.
			saveDict: Save dictionary or not(bool type).
			saveBowvec: Save bow-vector or not(bool type).
			returnValue: Return value or not(bool type).
		'''
		if load == True:
			self._dictionary.load(kwarg['saveDictPath'])
			self._BowVecOfEachDoc = list(corpora.MmCorpus(kwarg['saveBowvecPath']))
			filtered_token = joblib.load('./dictionary/token.p')
			corpora_documents = joblib.load('./dictionary/filtered_text.p')
			return filtered_token, self._dictionary, self._BowVecOfEachDoc

		self._raw_documents = documents
		corpora_documents, token = self.jieba_tokenize(documents)
		filtered_token = self.RemoveWordAppearBelowN(token) # 过滤前文本长度为16920，过滤后长度为16767
		pdb.set_trace()
		self._dictionary = corpora.Dictionary(filtered_token)  # 生成词典
		# print(self._dictionary.keys())  #133347 key
		# print(self._dictionary.get(5))
		# print(self._dictionary.dfs) # 单词id: 在多少文档中出现
		# print(self._dictionary.token2id)
		# for key in self._dictionary.dfs.keys():
		# 	if self._dictionary.dfs[key] < 2:
		# 		print(self._dictionary.get(key))
		# pdb.set_trace()
		joblib.dump(filtered_token, './dictionary/token.p')
		joblib.dump(corpora_documents, './dictionary/filtered_text.p')
		if kwarg['saveDict']:
			self._dictionary.save(kwarg['saveDictPath'])

		self._BowVecOfEachDoc = [self._dictionary.doc2bow(text) for text in token]  # 向量化
		if kwarg['saveBowvec']:
			corpora.MmCorpus.serialize(kwarg['saveBowvecPath'], self._BowVecOfEachDoc)
		if kwarg['returnValue']:
			return corpora_documents, self._dictionary, self._BowVecOfEachDoc

		def getConvertedModel(self, dictionary, bowvec, model_exist = False):
			save_model_path = './dictionary/language_model/'
			if model_exist == False:
				tfidf = models.TfidfModel(bowvec)  # initialize tfidf model
				tfidfVec = tfidf[bowvec] # use the model to transform whole corpus
				tfidf.save(save_model_path + "tfidf_model.tfidf")
				model = models.LsiModel(tfidfVec, id2word=dictionary, num_topics=kwarg['tfDim']) # initialize an LSI transformation
				modelVec = model[tfidfVec] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
				model.save(save_model_path + "lsi_model.lsi") # same for tfidf, lda, ...
			else:
				tfidf = models.TfidfModel.load(save_model_path +"tfidf_model.tfidf")
				tfidfVec = tfidf[bowvec]
				model = models.LsiModel.load(save_model_path + "lsi_model.lsi")
				modelVec = model[tfidfVec]
			return tfidfVec, modelVec

def load_dataset(filename = 'train_data.p', stopwordPath = './data/cn_stopwords.txt', loadflag = False):
	dataset = joblib.load(filename, 'rb')
	
	min_dataset = dataset[:20000]
	dictPath = os.getcwd() + '/dictionary/'
	textprocessing = TextProcessing(stopwordPath)
	token, dictionary, BowVecOfEachDoc = textprocessing.genDictionary(
		min_dataset['text'], load = loadflag, saveDict = True, saveDictPath=dictPath + 'dict.dict',
		saveBowvec=True, saveBowvecPath=dictPath + 'bow_vec.mm', returnValue=True)


	min_dataset['text'] = token

	return min_dataset['text'], min_dataset['label']

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""Iterate the data batch by batch"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
	token, dictionary, BowVecOfEachDoc = load_dataset('./data/train_data.p', './data/cn_stopwords.txt', True)
