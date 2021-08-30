from gensim.models.word2vec import KeyedVectors
from tensorflow.keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba_fast as jieba
import joblib, json
import collections,pdb
from gensim.models.word2vec import KeyedVectors
from gensim import corpora, similarities, models, matutils, utils
import numpy as np

def load_stopwords():
    stopwords = open('./data/cn_stopwords.txt', 'r', encoding = 'gbk').read().split('\n')
    print("stopwords counts", len(stopwords))
    print("example of stopword", stopwords[0])
    return stopwords

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def jieba_tokenize(documents):
    print("start to tokenizations, it takes times")
    stoplist = load_stopwords()
    tokens_list = []
    for item_text in documents:
        outstr = []
        sentence_seg = list(jieba.cut(item_text))
        for word in sentence_seg:
            if not is_chinese(word):
                continue
            if word not in stoplist and word != '\t' and word != ' ':
                outstr.append(word)
        tokens_list.append(outstr)
    print("tokenizations finish!")
    return tokens_list

def RemoveWordAppearBelowN(corpora_documents, n = 1):
    frequency = collections.defaultdict(int)
    for text in corpora_documents:
        for token in text:
            frequency[token] += 1
    corpora_documents = [[token for token in text if frequency[token] > n] for text in corpora_documents]
    return corpora_documents

def generate_word_embeddings(dictionary):
    wv_from_text = KeyedVectors.load_word2vec_format('./data/Tencent_AILab_ChineseEmbedding.txt', binary=False)
    joblib.dump(wv_from_text, './dictionary/embeddingMatrix.dict')
    print(wv_from_text['中国'])
    embeddingDim = 200
    embeddingList = np.random.uniform(-1, 1,[(len(dictionary) + 2), embeddingDim])
    print(type(wv_from_text))
    for index,token in dictionary.items():
        try:
            embeddingList[index] = wv_from_text[token]
        except:
            print("token is not found in embedding matrix", token, "frq",dictionary.dfs[token2id[token]])
    joblib.dump(embeddingList, './data/embedding_weights.list')

if __name__ == '__main__':
    train = joblib.load(open('data/train_data.p','rb'))
    test = joblib.load(open('data/test_data.p','rb'))
    x_text = train['text']
    y_text = test['text']
    all_text = x_text.append(y_text)
    token_list = jieba_tokenize(all_text)
    dictionary = corpora.Dictionary(token_list) #{index:vocab}
    print("before len", len(dictionary))
    dictionary.filter_extremes(no_below = 5, no_above=1.0, keep_n =None ) #n=2 111607
    print("after len", len(dictionary))

    dictionary.filter_n_most_frequent(5)
    dictionary.compactify()
    print("after filter_n_most_frequent len", len(dictionary))

    token2id = dictionary.token2id # {vocab:idx}
    joblib.dump(dictionary, './dictionary/idx2vocaburary_test_train.dict')
    joblib.dump(token2id, './dictionary/token2id_test_train.dict')
    print("vocab size", len(dictionary)) #before filter:227728, after filter:[n=3]78799, [n=5]54736
    print(dictionary.keys()[:10])
    print(dictionary.get(3))
    print(list(token2id.keys())[:10])
    print(token2id['中国'])

    params = json.loads(open('parameters.json').read())
    if params['use_embeddings']:
        generate_word_embeddings(dictionary)
