# Security-Analysis-of-Smart-Contracts
## 对一篇论文的小规模复现 [论文地址](https://www.ndss-symposium.org/ndss-paper/smarter-contracts-detecting-vulnerabilities-in-smart-contracts-with-deep-transfer-learning/ "悬停显示")
### 技术报告——基于深度迁移学习的智能合约漏洞检测
### 团队成员（上海奇安信总部——盘古实验室）
- **唐祝寿** - 总负责人
- **陆亦恬** - 数据分析师
- **朱璋颖** - 数据分析师
- **余俊** - 数据采集工程师


### 方法实现
#### 数据集获取 
途径一：目前的[开源数据集](https://github.com/mwritescode/smart-contracts-vulnerabilities?tab=readme-ov-file/ "悬停显示")
该数据集一共标记了10个类别：['locked-ether', 'bad-randomness', 'other', 'reentrancy', 'arithmetic', 'unchecked-calls', 'ignore', 'safe', 'double-spending', 'access-control']

目前该数据集文件经过处理后整理上传到我的bigquery表中了，具体连接代码看retrieve_data文件夹下的example.py 

python连接的前提准备，需要连接凭据（retrieve_data文件夹下的json文件）、安装bigquery连接库（pip install google-cloud-bigquery) 

途径二：1、我准备重新收集一批标记数据集，url：https://github.com/smartbugs/smartbugs-results/tree/master/results/honeybadger/icse20
该仓库里有用一篇论文里集合了9个标记工具的honeybadger工具进行标记的漏洞数据集，约2w个，但是是根据合约地址进行标记的，最后还需要进行转换
但是解析下来只有几百条多标签漏洞数据

2、又找到了一份人工检测的数据集：Consolidated Ground Truth (CGT) is a unified and consolidated ground truth with 20,455 manually checked assessments (positive and negative) of security-related properties.
但是解析下来只有2k多条多标签漏洞数据

3、所以我又找了一个打了四个漏洞标签的数据集——https://github.com/Messi-Q/Smart-Contract-Dataset/tree/master

4、最后我找了将近1w条的漏洞数据——https://github.com/sujeetc/ScrawlD/tree/main  

数据集数量：15893  
训练集数量：12714  测试集数量：3179  
每个漏洞类型的数量：  
ARTHM:9492  
CDAV:32  
DOS:1189  
LE:1654  
RENT:4970   
TimeM:2103  
TimeO:1284  
UE:1520  
safe:5000  

#### 数据集处理  
根据论文里的做法，通过将字节码转换为操作码构建vocabulary，进而转化成向量
#### 模型构建    
根据论文里的做法，构建MOL_DNN模型（基于深度神经网络(Deep Neural Network, DNN)的Multi-Output Learning (MOL)模型，一种可以同时预测多个输出目标的机器学习模型），但由于最后的实际训练效果，最后一层只输出两个标签，正例和重定向漏洞标签。
#### 模型训练  
| Layer (type) | Output Shape | Param # |  
| :----: | :----: | :----: |  
| input_1 (InputLayer) | [(None,3930)] | 0 |  
| embedding (Embedding) | (None, 3930, 60) | 4620 |  
| gru (GRU) | (None, 32) | 9024 |  
| batch_normalization (BatchNormalization) | (None, 32) | 128 |  
| dropout (Dropout) | (None, 32) | 0 |  
| dense (Dense) | (None, 16) | 528 |  
| dense_2 (Dense)  | (None, 16) | 528 |  
| dense_4 (Dense)  | (None, 16) | 528 |  
| dense_6 (Dense)  | (None, 16) | 528 |  
| dense_8 (Dense)  | (None, 16) | 528 |  
| dense_10 (Dense)  | (None, 16) | 528 |  
| dense_12 (Dense)  | (None, 16) | 528 |  
| dense_14 (Dense)  | (None, 16) | 528 |  
| dense_16 (Dense)  | (None, 16) | 528 |  
| batch_normalization_1 (BatchNormalization) | (None, 16) | 64 |  
| batch_normalization_2 (BatchNormalization) | (None, 16) | 64 |  
| batch_normalization_3 (BatchNormalization) | (None, 16) | 64 |  
| batch_normalization_4 (BatchNormalization) | (None, 16) | 64 |  
| batch_normalization_5 (BatchNormalization) | (None, 16) | 64 |  
| batch_normalization_6 (BatchNormalization) | (None, 16) | 64 |  
| batch_normalization_7 (BatchNormalization) | (None, 16) | 64 |  
| batch_normalization_8 (BatchNormalization) | (None, 16) | 64 |  
| batch_normalization_9 (BatchNormalization) | (None, 16) | 64 |  
| dropout_1 (Dropout) | (None, 16) | 0 |  
| dropout_2 (Dropout) | (None, 16) | 0 |  
| dropout_3 (Dropout) | (None, 16) | 0 |  
| dropout_4 (Dropout) | (None, 16) | 0 |  
| dropout_5 (Dropout) | (None, 16) | 0 |  
| dropout_6 (Dropout) | (None, 16) | 0 |  
| dropout_7 (Dropout) | (None, 16) | 0 |  
| dropout_8 (Dropout) | (None, 16) | 0 |  
| dropout_9 (Dropout) | (None, 16) | 0 |  
| dense_1 (Dense) | (None, 1) | 17 |  
| dense_3 (Dense) | (None, 1) | 17 |  
| dense_5 (Dense) | (None, 1) | 17 |  
| dense_7 (Dense) | (None, 1) | 17 |  
| dense_9 (Dense) | (None, 1) | 17 |  
| dense_11 (Dense) | (None, 1) | 17 |  
| dense_13 (Dense) | (None, 1) | 17 |  
| dense_15 (Dense) | (None, 1) | 17 |  
| dense_17 (Dense) | (None, 1) | 17 |  

模型准确率：91.98%  
模型误报率：2.3%  
ARTHM accuracy: 0.3976  
CDAV accuracy: 0.9984  
DOS accuracy: 0.9201  
LE accuracy: 0.9025
RENT accuracy: 0.6861  
TimeM accuracy: 0.8610  
TimeO accuracy: 0.9236  
UE accuracy: 0.8912  
safe accuracy: 0.3108  

#### 模型检测（我和余到时候把分析结果写在这里）
检测结果：在线文档 https://kdocs.cn/l/cbrfAIlpfG0f  
检测数量：4/sec







