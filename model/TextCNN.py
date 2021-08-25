# -*- coding: utf-8 -*-
#TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
# print("started...")
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.word2vec import KeyedVectors
from gensim.test.utils import datapath
import time
def loadPretrainedEmbeddings():
    '''
    load from: https://ai.tencent.com/ailab/nlp/en/embedding.html
    '''
    wv_from_text = KeyedVectors.load_word2vec_format('./data/Tencent_AILab_ChineseEmbedding.txt', binary=False)
    print("finish loading pretrained word embeddings")
    return wv_from_text

class TextCNN:
    def __init__(self, filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size
                 ,initializer=tf.random_normal_initializer(stddev=0.1),multi_label_flag=False,clip_gradients=5.0,decay_rate_big=0.50, usePretrainEmbeddings=False):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size


        self.filter_sizes=filter_sizes # a list of int: [3,4,5]
        self.num_filters=num_filters
        self.initializer=initializer
        self.num_filters_total=self.num_filters * len(filter_sizes)
        self.multi_label_flag=multi_label_flag
        self.clip_gradients = clip_gradients
        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None,],name="input_y")  # y:[None,]
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.learning_rate = learning_rate

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))

        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights(usePretrainEmbeddings)
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        self.loss_val = self.loss()
        self.train_op = self.train()
        
        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
        print("self.predictions:", self.predictions)
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()

    def instantiate_weights(self, usePretrainEmbeddings):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            if usePretrainEmbeddings:
                self.Embedding = loadPretrainedEmbeddings()
            self.W_projection = tf.get_variable("W_projection",shape=[self.num_filters_total, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])

    def inference(self):
        """main computation graph here: 1.embedding-->2.CONV-BN-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)#[None,sentence_length,embed_size]
        self.sentence_embeddings_expanded=tf.expand_dims(self.embedded_words,-1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        cnn_output=self.cnn_single_layer()
        #3. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(cnn_output, self.W_projection) + self.b_projection  #shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits

    def cnn_single_layer(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("convolution-pooling-%s" % filter_size):
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID',name="pool")  # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)

        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1,self.num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
        h = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)
        return h

    def loss(self,l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)#sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss=tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate=learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op
