本项目是针对法条进行分类，数据来源是人工智能司法杯2018年的数据
http://cail.cipsc.org.cn:2018/

运行顺序：
0. dataanlysis.ipynb分析了数据的分布，并解释了【max_seq_len】参数的设置

2. python generate_word_embeddings.py  生成字典和词向量

4. python train.py

    其中配置文件'parameters.json’配置模型相关参数含义如下：

    "num_epochs": 训练epoch个数，
    
    "batch_size": 数据批大小,
    
    "num_filters": 卷积核个数,
    
    "filter_sizes": CNN 卷积核kernel size,
    
    "embedding_dim": 词向量维度,
    
    "l2_reg_lambda": 正则惩罚项参，
    
    "dropout_keep_prob": 0.5
    
    "num_classes": 类别个数,
    
    "max_seq_len": 最大句子长度，
    
    “loadTokened“：是否加载已经分词索引化的数据，如果是第一次执行，设置为False
    
    "use_embeddings":是否使用预训练的词向量，下载来源：https://ai.tencent.com/ailab/nlp/en/embedding.html


3. 训练结果：

1）对比了fastText和TextCNN, 训练结果分别为【40%】和【84%】

2）以TextCNN为主干网络，加载预训练的词向量，训练结果为【】

  
