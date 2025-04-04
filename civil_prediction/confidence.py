from json import dumps
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np


def calculate_entropy(probabilities):
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    return entropy

#计算方差
def calculate_variance(probabilities):
    return np.var(probabilities, axis=-1)



if __name__ == '__main__':
    #加载数据集
    data_dict_loaded = np.load('civil_output\CAIL_test.npy', allow_pickle=True).item()
    #所有的预测数据地址
    accu_logits = data_dict_loaded['accu_logits']
    true_accu = data_dict_loaded['true_accu'] # 这个是字符串类型
    pred_accu = data_dict_loaded['pred_accu']
    case_id = data_dict_loaded['case_id']
    fact = data_dict_loaded['fact']
    print(len(fact)) #
    #对accu_logits进行softamx
    probabilities = np.exp(accu_logits) / np.sum(np.exp(accu_logits), axis=1, keepdims=True)
    #计算最大概率
    max_probabilities = np.max(probabilities, axis=1)
    # #计算熵
    # entropies = calculate_entropy(probabilities) #1
    # #计算方差
    # vars = calculate_variance(probabilities) # 
    #保存数据
    thred = 0.7
    filtered_file: str = f"civil_output/filtered_{thred}.jsonl"
    with open(filtered_file, 'w', encoding='utf-8') as f2:
        wrong_num_all = 0
        filtered_wrong_num = 0
        num_filtered = 0
        for i,data in enumerate(fact):
            if true_accu[i] != pred_accu[i]:
                wrong_num_all += 1 #存在的错误数据
            #过滤出来的那部分数据
            # if  vars[i] < 0.007:
            # if  entropies[i] > 0.1:
            if  max_probabilities[i] < thred:
                print(dumps({
                    'fact': data,
                    "case_id": int(case_id[i]),
                    'accu_label': int(true_accu[i]),
                    'pre_accu': int(pred_accu[i])
                }, ensure_ascii=False), file=f2)  
                num_filtered += 1 
                #统计过滤的数据中,错误预测的数量
                if true_accu[i] != pred_accu[i]:
                    filtered_wrong_num += 1

        ##print
        print(f"wrong_num_all: {wrong_num_all}")
        print(f"num_filtered: {num_filtered}")
        print(f"wrong_num_filtered: {filtered_wrong_num}")

