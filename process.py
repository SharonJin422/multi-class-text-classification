import joblib
import jieba, os
import pdb
from collections import defaultdict
from gensim import corpora,similarities,models,matutils,utils

class TextProcessing(object):
	def __init__(self, stopwordPath):
		self.stopwordPath = stopwordPath

	def load_stopwords(self):
		stopwords = open(self.stopwordPath, 'r', encoding='utf-8').read().split('\n')
		print("stopwords counts", len(stopwords))
		return stopwords

	def jieba_tokenize(self, documents):
		'''Cut the documents into a sequence of independent words.
		# Arguments:
			documents: List of news(articles).
		'''
		chnSTW = self.load_stopwords()
		corpora_documents = []
		for item_text in documents:
			outstr = []
			sentence_seg = list(jieba.cut(item_text))
			for word in sentence_seg:
				if word not in chnSTW and word != '\t' and word != ' ':
					outstr.append(word)
			corpora_documents.append(outstr)
		return corpora_documents

	def RemoveWordAppearOnce(self, corpora_documents):
		'''Remove the words that appear once among all the tokenized news(articles).
		# Arguments:
			 corpora_documents: List of tokenized news(articles).
		'''
		frequency = defaultdict(int)
		for text in corpora_documents:
			for token in text:
				frequency[token] += 1
		corpora_documents = [[token for token in text if frequency[token] > 1] for text in corpora_documents]
		return corpora_documents

	def genDictionary(self, documents, **kwarg):
		'''Generate dictionary and bow-vector of all tokenzied news(articles).
		# Arguments:
			documents: List of texts.
			saveDict: Save dictionary or not(bool type).
			saveBowvec: Save bow-vector or not(bool type).
			returnValue: Return value or not(bool type).
		'''
		self._raw_documents = documents
		token = self.jieba_tokenize(documents)  # jieba tokenize
		# corpora_documents = self.RemoveWordAppearOnce(token)  # remove thw words appearing once in the dictionary
		self._dictionary = corpora.Dictionary(token)  # 生成词典
		pdb.set_trace()
		if kwarg['saveDict']:
			self._dictionary.save(kwarg['saveDictPath'])  # store the dictionary, for future reference
		self._BowVecOfEachDoc = [self._dictionary.doc2bow(text) for text in token]  # 向量化
		if kwarg['saveBowvec']:
			corpora.MmCorpus.serialize(kwarg['saveBowvecPath'], self._BowVecOfEachDoc)  # store to disk, for later use
		if kwarg['returnValue']:
			return token, self._dictionary, self._BowVecOfEachDoc
def load_dataset(filename = 'train_data.p'):
	dataset = joblib.load(filename, 'rb')
	DictPath = os.getcwd() + '\\' + 'data'

	textprocessing  = TextProcessing('./data/cn_stopwords.txt')
	textprocessing.genDictionary(dataset['text'], saveDict=True, saveDictPath=DictPath + '\\' + 'dict.dict')





	return dataset['text'], dataset['label']

if __name__ == '__main__':
	load_dataset('./data/train_data.p')
