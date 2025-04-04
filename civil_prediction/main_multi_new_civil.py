from json import dumps
from pathlib import Path
from typing import List
from transformers import (BertTokenizer,Trainer, set_seed)
from util_civil import (JsonlDataset,compute_metrics,BertForMultiTaskClassification,softmax)
import logging,os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np

local_rank = int(os.getenv('LOCAL_RANK', -1))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"  # 设置使用的GPU


if __name__ == '__main__':
    set_seed(42)
    model_path = 'macbert_civil' # 再试下：results_CAIL_819/checkpoint-9650
    Path('output_312').mkdir(exist_ok=True)


    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
    #直接获取model的checkpoint,作为模型来加载,此外要指定相关的label数
    model: BertForMultiTaskClassification = BertForMultiTaskClassification.from_pretrained(
        model_path, num_labels_task1 = 257) #仅仅关注于罪名预测

    #指定CAIL test数据集
    test_file_path = 'civil_data/test_with_id.json' #这个test_new还是要处理一下

    test_dataset = JsonlDataset(test_file_path, tokenizer, max_length=512)#这个是一个对象
    
    trainer = Trainer(model=model,compute_metrics=compute_metrics)
    #测试结果 
    test_results = trainer.predict(test_dataset)
    logger.info(f"Test Results: {test_results.metrics}")#打印预测结果

    #如果要获取eval结果，可以指定eval_dataset

    # eval_file_path = 'CAIL_small/valid_new.jsonl'
    # eval_dataset = JsonlDataset(eval_file_path, tokenizer, max_length=512)
    # eval_results = trainer.predict(eval_dataset)
    # logger.info(f"Eval Results: {eval_results}")  


    # 提取对应的概率分布
    accu_logits = test_results.predictions

    #预测值~
    bert_pred_accu = np.argmax(accu_logits, axis=-1)#(65,4,65,35,35,35)
    #真实标签
    true_accu: List[int] = test_dataset.accu_labels
    # 先储存起来上面的结果
    # 将多个列表转换为 NumPy 数组(先仅仅获取accu 和law 吧)
    data_dict = {
            'case_id':np.array(test_dataset.id),
            'fact': np.array(test_dataset.texts),
            'accu_logits': np.array(accu_logits),
            'pred_accu': np.array(bert_pred_accu),
            'true_accu': np.array(true_accu)
        }
    # 把BERT的预测结果先写入文件
    np.save('civil_output/CAIL_test.npy', data_dict)  #可以改为CAIL_sample_bert

    # 所有的预测数据地址
    out_file: str = 'civil_output/CAIL_test.jsonl' #保存到文件中
    # 过滤出的数据地址~
    filtered_file: str = 'civil_output/filtered_CAIL.jsonl'
    #预测概率分布熵的阈值,-->让GPT4o重新进行预测~

    with open(out_file, 'w', encoding='utf-8') as f:
        with open(filtered_file, 'w', encoding='utf-8') as f2:
            wrong_num_all = 0
            filtered_wrong_num = 0
            num_filtered = 0
            for i,data in enumerate(test_dataset.texts):
                print(dumps({
                        'fact': data,
                        "case_id":test_dataset.id[i],
                        'accu_label': true_accu[i],
                        'pre_accu': int(bert_pred_accu[i])
                    }, ensure_ascii=False), file = f) #保存到文件中
                
                #测试数据集中含有的错误数~
                if true_accu[i] != bert_pred_accu[i]:
                    wrong_num_all += 1 #存在的错误数据
                
                #过滤出来的那部分数据
                accu_max = np.max(softmax(accu_logits[i]), axis=-1) 
                if accu_max <0.7: #这个是确定的最佳值
                    print(dumps({
                        'fact': data,
                        "case_id": test_dataset.id[i],
                        'accu_label': true_accu[i],
                        'pre_accu': int(bert_pred_accu[i])
                    }, ensure_ascii=False), file=f2)  
                    num_filtered += 1 
                    #统计过滤的数据中,错误预测的数量
                    if true_accu[i] != bert_pred_accu[i]:
                        filtered_wrong_num += 1

            ##打印情况~
            print(f"wrong_num_all: {wrong_num_all}")
            print(f"num_filtered: {num_filtered}")
            print(f"wrong_num_filtered: {filtered_wrong_num}")

