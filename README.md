# Security-Analysis-of-Smart-Contracts
## 对一篇论文的小规模复现 [论文地址](https://www.ndss-symposium.org/ndss-paper/smarter-contracts-detecting-vulnerabilities-in-smart-contracts-with-deep-transfer-learning/ "悬停显示")
### 方法实现
#### 数据集获取 
途径一：目前的开源标注数据集（https://github.com/mwritescode/smart-contracts-vulnerabilities?tab=readme-ov-file）
该数据集一共标记了10个类别：['locked-ether', 'bad-randomness', 'other', 'reentrancy', 'arithmetic', 'unchecked-calls', 'ignore', 'safe', 'double-spending', 'access-control']

目前该数据集文件经过处理后整理上传到我的bigquery表中了，具体连接代码看retrieve_data文件夹下的example.py 
python连接的前提准备，需要连接凭据（retrieve_data文件夹下的json文件）、安装bigquery连接库（pip install google-cloud-bigquery) 

#### 数据集处理
#### 模型构建
#### 模型检测








