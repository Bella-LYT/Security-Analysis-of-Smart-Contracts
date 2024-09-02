import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GRU, BatchNormalization, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super(Metrics, self).__init__()
        self.validation_data = val_data

    def on_epoch_end(self, epoch, logs={}):
        # val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        pre = self.model.predict(self.validation_data[0])
        val_targ = self.validation_data[1]
        val_predict = [np.array(line) for line in np.where(np.array(pre) > 0.5, 1, 0).tolist()]
        # (np.asarray(self.model.predict(self.validation_data[0])) > 0.5).astype(int)
        # val_targ = self.validation_data[1]

        _val_precision = precision_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_f1 = f1_score(val_targ, val_predict, average='micro')

        print(f' — val_precision: {_val_precision:.4f} — val_recall: {_val_recall:.4f} — val_f1: {_val_f1:.4f}')

        logs['val_precision'] = _val_precision
        logs['val_recall'] = _val_recall
        logs['val_f1'] = _val_f1
        return


def load_train_data():
    train_data_df = pd.read_csv(
        '/Users/zzy/Desktop/PycharmProjects/MyProject/smart-contract/train_data/train-dataset.csv')  # ,delimiter='\t')
    train_data_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    train_data_df['swc_id'] = train_data_df['swc_id'].apply(eval)
    # 一共有15893条智能合约记录

    # 新增一列标签索引，现在共有'bytecode,swc_id,opcodes,label_idx'四列
    train_data_df['label_idx'] = '[]'

    # 对标签类型进行编码,类型包括[ARTHM,CDAV,DOS,LE,RENT,TimeM,TimeO,UE,safe]
    all_labels = []
    for line in train_data_df['swc_id']:
        for l in line:
            if l not in all_labels:
                all_labels.append(l)

    enc = preprocessing.LabelEncoder()
    enc = enc.fit(all_labels)
    for i, row in train_data_df.iterrows():
        swc_id = row['swc_id']
        label_idx = enc.transform(swc_id).tolist()
        train_data_df.loc[i, 'label_idx'] = str(label_idx)

    print('每个标签的数量：', 'ARTHM:{}'.format(train_data_df['swc_id'].astype(str).str.count('ARTHM').sum()),
          'CDAV:{}'.format(train_data_df['swc_id'].astype(str).str.count('CDAV').sum()),
          'DOS:{}'.format(train_data_df['swc_id'].astype(str).str.count('DOS').sum()),
          'LE:{}'.format(train_data_df['swc_id'].astype(str).str.count('LE').sum()),
          'RENT:{}'.format(train_data_df['swc_id'].astype(str).str.count('RENT').sum()),
          'TimeM:{}'.format(train_data_df['swc_id'].astype(str).str.count('TimeM').sum()),
          'TimeO:{}'.format(train_data_df['swc_id'].astype(str).str.count('TimeO').sum()),
          'UE:{}'.format(train_data_df['swc_id'].astype(str).str.count('UE').sum()),
          'safe:{}'.format(train_data_df['swc_id'].astype(str).str.count('safe').sum()))
    # 每个标签的数量： ARTHM:9492 CDAV:32 DOS:1189 LE:1654 RENT:4970 TimeM:2103 TimeO:1284 UE:1520 safe:5000

    # train_data_df.to_csv('/Users/zzy/Desktop/PycharmProjects/MyProject/smart-contract/train_data/...', encoding='utf-8')
    print('done')
    return train_data_df


def process_data(train_data_df):
    train_data_df['label_idx'] = train_data_df['label_idx'].apply(eval)
    opcodes = train_data_df['opcodes'].tolist()
    labels = train_data_df['label_idx'].tolist()

    tokenizer = Tokenizer(num_words=None)  # num_words:None或整数,处理的最大单词数量。少于此数的单词丢掉
    tokenizer.fit_on_texts(opcodes)
    sequences = tokenizer.texts_to_sequences(opcodes)  # 得到单词的索引，受num_words影响

    max_sequence_length = int(np.mean([len(op.split()) for op in opcodes]))
    print('max sequence length:', max_sequence_length)

    input_dim = len(tokenizer.word_index) + 1
    print('input dim:', input_dim)

    opcodes_idx = pad_sequences(sequences, maxlen=max_sequence_length)  # 将长度不足xx的用 0 填充（在前端填充）

    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(labels)

    print('dataset shape:', opcodes_idx.shape)
    print('label shape:', labels_encoded.shape)

    opcodes_idx_train, opcodes_idx_val, labels_encoded_train, labels_encoded_val = train_test_split(opcodes_idx,
                                                                                                    labels_encoded,
                                                                                                    test_size=0.2,
                                                                                                    random_state=42)

    print('train-data shape:', opcodes_idx_train.shape, 'val-data shape:', opcodes_idx_val.shape)
    print('label of train-data shape:', labels_encoded_train.shape, 'label of val-data shape:',
          labels_encoded_val.shape)

    y_train = [np.array(line) for line in labels_encoded_train.T.reshape(9, labels_encoded_train.shape[0], 1).tolist()]
    y_val = [np.array(line) for line in labels_encoded_val.T.reshape(9, labels_encoded_val.shape[0], 1).tolist()]

    bytecode = train_data_df['bytecode'].tolist()
    bytecode_ori_train, bytecode_ori_val, opcodes_ori_train, opcodes_ori_val, labels_ori_train, labels_ori_val = train_test_split(
        bytecode,
        opcodes,
        labels,
        test_size=0.2,
        random_state=42)
    print(len(bytecode_ori_train), len(bytecode_ori_val))
    print(len(opcodes_ori_train), len(opcodes_ori_val))
    print(len(labels_ori_train), len(labels_ori_val))
    return input_dim, max_sequence_length, opcodes_idx_train, y_train, opcodes_idx_val, y_val, labels_encoded_val, bytecode_ori_val


print('GPU:', len(tf.config.experimental.list_physical_devices('GPU')))

# 定义模型参数
embedding_dim = 60
gru_units = 32
units_list = [16] * 9  # 每个分支的隐藏单元数量
num_classes_list = [1] * 9  # 每个分支的输出类别数量，这里假设是二分类任务


# 定义base网络结构
def base_model(max_sequence_length, input_dim, embedding_dim, gru_units):
    inputs = Input(shape=(max_sequence_length,))
    embedding = Embedding(input_dim, embedding_dim)(inputs)
    gru = GRU(gru_units)(embedding)
    bn = BatchNormalization()(gru)
    dropout = Dropout(0.6)(bn)
    return inputs, dropout


# 定义分支网络结构
def branch_model(input_layer, units, num_classes):
    dense = Dense(units, activation='relu')(input_layer)
    bn = BatchNormalization()(dense)
    dropout = Dropout(0.6)(bn)
    output = Dense(num_classes, activation='sigmoid')(dropout)
    return output


# 构建模型
def build_model(max_sequence_length, input_dim, embedding_dim, gru_units, units_list, num_classes_list):
    inputs, base_output = base_model(max_sequence_length, input_dim, embedding_dim, gru_units)
    outputs = []
    for units, num_classes in zip(units_list, num_classes_list):
        branch_output = branch_model(base_output, units, num_classes)
        outputs.append(branch_output)
    model = Model(inputs=inputs, outputs=outputs)
    return model


train_data_df = load_train_data()
input_dim, max_sequence_length, opcodes_idx_train, y_train, opcodes_idx_val, y_val, labels_encoded_val, bytecode_ori_val = process_data(
    train_data_df)

stime = time.time()
# 构建模型
model = build_model(max_sequence_length, input_dim, embedding_dim, gru_units, units_list, num_classes_list)

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)


# 加入到模型的回调函数中
metrics = Metrics(val_data=(opcodes_idx_val, y_val))


# 训练模型
model.fit(opcodes_idx_train, y_train, batch_size=512, epochs=160, validation_data=(opcodes_idx_val, y_val),
          callbacks=[early_stopping, metrics])

print('need {} min'.format((time.time() - stime) / 60))

model.summary()

res = model.predict(
    opcodes_idx_val, batch_size=64, verbose='auto', steps=None, callbacks=None
)

model.save_weights('model_weights-829.h5')
model_json = model.to_json()
with open('model_structure-829.json', 'w') as json_file:
    json_file.write(model_json)

print("all done")

"""
combined_array = np.concatenate((res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]), axis=1)
np.savetxt("model_output.csv", combined_array, delimiter=",", fmt="%f")
"""


"""
GPU: 1
每个标签的数量： ARTHM:9492 CDAV:32 DOS:1189 LE:1654 RENT:4970 TimeM:2103 TimeO:1284 UE:1520 safe:5000
done
max sequence length: 3930
input dim: 77
dataset shape: (15893, 3930)
label shape: (15893, 9)
train-data shape: (12714, 3930) val-data shape: (3179, 3930)
label of train-data shape: (12714, 9) label of val-data shape: (3179, 9)
12714 3179
12714 3179
12714 3179
Metal device set to: Apple M1
systemMemory: 16.00 GB
maxCacheSize: 5.33 GB
"""
