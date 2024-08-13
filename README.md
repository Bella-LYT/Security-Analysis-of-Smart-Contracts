# Security-Analysis-of-Smart-Contracts
## 对一篇论文的小规模复现 [论文地址](https://www.ndss-symposium.org/ndss-paper/smarter-contracts-detecting-vulnerabilities-in-smart-contracts-with-deep-transfer-learning/ "悬停显示")
### 方法实现
#### 数据集获取 
途径一：目前的[开源数据集](https://github.com/mwritescode/smart-contracts-vulnerabilities?tab=readme-ov-file/ "悬停显示")
该数据集一共标记了10个类别：['locked-ether', 'bad-randomness', 'other', 'reentrancy', 'arithmetic', 'unchecked-calls', 'ignore', 'safe', 'double-spending', 'access-control']

目前该数据集文件经过处理后整理上传到我的bigquery表中了，具体连接代码看retrieve_data文件夹下的example.py 

python连接的前提准备，需要连接凭据（retrieve_data文件夹下的json文件）、安装bigquery连接库（pip install google-cloud-bigquery) 

途径二：1、我准备重新收集一批标记数据集，url：https://github.com/smartbugs/smartbugs-results/tree/master/results/honeybadger/icse20，该仓库里有用一篇论文里集合了9个标记工具的honeybadger工具进行标记的漏洞数据集，约2w个，但是是根据合约地址进行标记的，最后还需要进行转换
但是解析下来只有几百条多标签漏洞数据

2、又找到了一份人工检测的数据集：Consolidated Ground Truth (CGT) is a unified and consolidated ground truth with 20,455 manually checked assessments (positive and negative) of security-related properties.
但是解析下来只有2k多条多标签漏洞数据

3、所以我又找了一个打了四个漏洞标签的数据集——https://github.com/Messi-Q/Smart-Contract-Dataset/tree/master

4、最后我找了将近1w条的漏洞数据——https://github.com/sujeetc/ScrawlD/tree/main

#### 数据集处理  
根据论文里的做法，通过将字节码转换为操作码构建vocabulary，进而转化成向量
#### 模型构建    
根据论文里的做法，构建MOL_DNN模型（基于深度神经网络(Deep Neural Network, DNN)的Multi-Output Learning (MOL)模型，一种可以同时预测多个输出目标的机器学习模型），但由于最后的实际训练效果，最后一层只输出两个标签，正例和重定向漏洞标签。
#### 模型训练  
| Layer (type) | Output Shape | Param # |  
| :----: | :----: | :----: |  
| input_3 (InputLayer) | [(None,5018)] | 0 |  
| embedding_2 (Embedding) | (None, 5018, 60) | 4620 |  
| gru_2 (GRU) | (None, 32) | 9024 |  
| batch_normalization_6 (BatchNormalization) | (None, 32) | 128 |  
| dropout_6 (Dropout) | (None, 32) | 0 |  
| dense_8 (Dense) | (None, 16) | 528 |  
| dense_10 (Dense)  | (None, 16) | 528 |  
| batch_normalization_7 (BatchNormalization) | (None, 16) | 64 |  
| batch_normalization_8 (BatchNormalization) | (None, 16) | 64 |  
| dropout_7 (Dropout) | (None, 16) | 0 |  
| dropout_8 (Dropout) | (None, 16) | 0 |  
| dense_9 (Dense) | (None, 1) | 17 |  
| dense_11 (Dense) | (None, 1) | 17 |  
#### 模型检测（我和余到时候把分析结果写在这里）








