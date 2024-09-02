import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json


# 操作码对照表
def get_opcodes(opcodes_file):
    opcodes = {}
    opcode_length = {}
    with open(opcodes_file, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            opcode = line.replace('\n', '').split('\t')
            opcodes[opcode[0].lower()] = opcode[1]
            opcode_length[opcode[0].lower()] = int(opcode[2])
    return opcodes, opcode_length


# 将字节码转为操作码
def bytecode_to_opcodes(bytecode):
    opcodes_file = 'opcode.txt'
    opcodes, opcode_length = get_opcodes(opcodes_file)

    opcodes_list = []
    i = 0
    while i < len(bytecode):
        opcode = bytecode[i:i + 2]
        if opcode in opcodes:
            if opcode_length[opcode] != 1:
                # value = bytecode[i+2:i+opcode_length[opcode] * 2]  # 带参数
                pass
            else:
                # value = ''  # 不带操作数
                pass

            if int(opcode, 16) >= 96 and int(opcode, 16) <= 127:
                opcodes_list.append(opcodes['60'])
            elif int(opcode, 16) >= 128 and int(opcode, 16) <= 143:
                opcodes_list.append(opcodes['80'])
            elif int(opcode, 16) >= 144 and int(opcode, 16) <= 159:
                opcodes_list.append(opcodes['90'])
            elif int(opcode, 16) >= 160 and int(opcode, 16) <= 164:
                opcodes_list.append(opcodes['a0'])
            elif int(opcode, 16) == 240 or int(opcode, 16) == 245:
                opcodes_list.append(opcodes['f0'])
            elif int(opcode, 16) == 86 or int(opcode, 16) == 87:
                opcodes_list.append(opcodes['56'])
            else:
                opcodes_list.append(opcodes[opcode])

            i += opcode_length[opcode] * 2
        else:
            opcodes_list.append('XX')  # 无效操作
            i += 2
    return opcodes_list


# 加载模型
def load_model():
    with open('./model-structure/model_structure-814.json', 'r') as file:
        model_json = file.read()

    model = model_from_json(model_json)
    model.load_weights('./model-structure/model_weights-814.h5')
    model.compile(optimizer='adam', loss='categorial_crossentropy', metrics=['accuracy'])

    # model.summary()
    return model


def load_data():
    test_df = pd.read_csv('./test_data/eth_bytecode_crypto_eth_contracts-000000000017.csv')
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    # 指定时间范围
    end_time = pd.Timestamp('2024-01-01 00:00:00 UTC')

    # 删除指定范围的行
    test_df = test_df[~(test_df['timestamp'] < end_time)]
    test_df = test_df[~(test_df['bytecode'] == '0x')]

    bytecodes = test_df['bytecode']
    addresses = test_df['address']
    timestamp = test_df['timestamp']

    opcodes = []
    for bytecode in bytecodes:
        opcodes.append(' '.join(bytecode_to_opcodes(bytecode)))
    # print(opcodes[0])

    tokenizer = Tokenizer(num_words=None)  # num_words:None或整数,处理的最大单词数量。少于此数的单词丢掉
    tokenizer.fit_on_texts(opcodes)
    sequences = tokenizer.texts_to_sequences(opcodes)  # 得到单词的索引，受num_words影响

    MAX_SEQUENCE_LENGTH = 3930

    opcodes_idx = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 将长度不足xx的用 0 填充（在前端填充）

    print('test dataset size:', opcodes_idx.shape)
    return opcodes_idx, bytecodes, addresses, timestamp


def predict_contract(model, opcodes_idx):
    res = model.predict(
        opcodes_idx, batch_size=64, verbose='auto', steps=None, callbacks=None
    )
    combined_array = np.concatenate((res[0], res[1]), axis=1)
    return combined_array


def output(combined_array, bytecodes, addresses, timestamp):
    label = ['ARTHM', 'CDAV', 'DOS', 'LE', 'RENT', 'TimeM', 'TimeO', 'UE', 'safe']
    for line in zip(combined_array, bytecodes, addresses, timestamp):
        predict_res = line[0]
        elements_over_0_9 = [(index, value) for index, value in enumerate(predict_res) if value >= 0.7]  # 输出结果
        for index, value in elements_over_0_9:
            print(line[2], ',', ',', ',', label[index], ',', line[3])
    return


model_multi = load_model()
opcodes_idx, bytecodes, addresses, timestamp = load_data()
combined_array = predict_contract(model_multi, opcodes_idx)
output(combined_array, bytecodes, addresses, timestamp)
