本项目是针对法条进行分类，数据来源是人工智能司法杯2018年的数据<br> 
http://cail.cipsc.org.cn:2018/ <br> 

##### 数据预处理说明<br>
1. dataanlysis.ipynb分析了数据的分布，并解释了【max_seq_len】参数的设置 <br> 
2. 对label的特殊处理，将one-hot标签改成了单标签，即[0，0，0，1，0,....] = [3] <br>
3. text的预处理：<br>
    1）去停用词，包括符号、数字、非中文<br>
    2）数据中有很多时间对法条的分类无意义，所以做了删除，包括地点、人名也是的，可以做进一步的过滤<br>
    3）因为文本长度分布不均，针对小于max_seq_len的进行padding,大于max_seq_len的文本则选中从前面截断，保留尾部信息<br>
    4) 总共有202个标签，但是实际数据中只有197类有数据，并且分布不均，针对label较少的数据可以做一个数据增强【随机删除词语并拷贝】，减少样本的不均衡<br> 
    

##### 如何训练：<br> 
1. python generate_word_embeddings.py  生成字典和词向量 <br> 
   前提：将配置文件（parameters.json）中的设置为true<br>
   说明：<br>
   1）字典的构建是融合了训练集和测试集的text，为了减少测试集中未登录词的出现<br>
   2）字典中有很多词在案例中出现的频率过低，并且无意义，所以过滤了【在案例中出现的次数】低于5的词语，将原本22万的词典缩小至6.6w<br>
   3) 过滤高频出现的5个词
   
2. python train.py <br>
    其中配置文件'parameters.json’配置模型相关参数含义如下：<br> 
    "num_epochs": 训练epoch个数，<br> 
    "batch_size": 数据批大小,<br> 
    "num_filters": 卷积核个数, <br>
    "filter_sizes": CNN 卷积核kernel size,  <br> 
    "embedding_dim": 词向量维度, <br> 
    "dropout_keep_prob": 0.5 <br>
    "num_classes": 类别个数, <br>
    "max_seq_len": 最大句子长度， <br>
    “loadTokened“：是否加载已经分词索引化的数据，如果是第一次执行，设置为False <br>
    "use_embeddings":是否使用预训练的词向量，下载来源：https://ai.tencent.com/ailab/nlp/en/embedding.html <br>


3. 训练结果:<br>
    因为该任务文本属于较长文本，考虑到序列模型的梯度消失问题，选择了CNN网络结构<br>
    1）对比了fastText和TextCNN, 训练结果分别为【40%】和【84%】 <br>
    2）以TextCNN为主干网络，加载预训练的词向量，训练结果为【84.5%】，和不加embedding的效果差不多 <br>
    3）进一步的优化想法是：<br>
    A. 从网络结构入手，可以考虑加上attention<br>
    B. 从embedding入手，针对法条分类的场景，可以在pretrain的语言模型基础上用BETR再fine-tune垂直领域的语言模型<br>
       
##### 如何预测<br>
   1）label需要进行np.argmax()处理<br>
   2）python predict.py<br>

  
