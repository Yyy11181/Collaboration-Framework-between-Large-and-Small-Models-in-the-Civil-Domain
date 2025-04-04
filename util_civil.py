import jsonlines
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel,TrainerCallback
import evaluate,json
import numpy as np
from typing import List, Tuple



with open('civil/label2id.json', 'r', encoding='utf-8') as f:
    label2id = json.load(f)


class JsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [] #这个才是在这边调用
        self.accu_labels = [] #罪名标签
        self.id = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for obj in data:
                self.texts.append(obj['fact']) #这个变成了fact
                self.accu_labels.append(label2id[obj['action_cause']])#提取罪名标签
                self.id.append(obj['case_id']) #在LAIC里面是id/在CAIL2018里面是case_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        accu_label = self.accu_labels[idx] #注意新加的
        id = self.id[idx]
        encoding = self.tokenizer( #这个返回的是input_ids,token_type_ids和attention_mask，都是tensor的格式哎
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'text':text,
            'id':id,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels_task1': torch.tensor(accu_label, dtype=torch.long)
        }


class BertForMultiTaskClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels_task1):#
        super().__init__(config)
        self.num_labels_task1 = num_labels_task1 #任务一的标签数
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        #定义两个任务的分类层
        self.classifier_task1 = nn.Linear(config.hidden_size, num_labels_task1)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels_task1=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output)
        
        logits_task1 = self.classifier_task1(pooled_output)
      
        
        outputs = (logits_task1,) + outputs[2:]
        
        loss = None
        if labels_task1 is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_task1 = loss_fct(logits_task1.view(-1, self.num_labels_task1), labels_task1.view(-1))
            loss = loss_task1 #这边也就2个任务的loss
        
        outputs = (loss,) + outputs if loss is not None else outputs
        
        return outputs
    


#在每次epoch后打印评估结果
class EvalResultsCallback(TrainerCallback):
    def __init__(self, output_file):
        self.output_file = output_file

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics")
        if metrics and args.local_rank in [-1, 0]:  # 只打印主进程，这边只要加上args就可以了
            with open(self.output_file, "a") as f:
                json.dump({"epoch": state.epoch, "metrics": metrics}, f)
                f.write("\n")


#评价指标
def compute_metrics(eval_pred):#这个是包含logits和label的

    logits_task1 = eval_pred.predictions
    labels_task1 = eval_pred.label_ids
    predictions_task1 = np.argmax(logits_task1, axis=-1)
    
    accuracy_metric = evaluate.load("accuracy.py")
    precision_metric = evaluate.load("precision.py")
    recall_metric = evaluate.load("recall.py")
    f1_metric = evaluate.load("f1.py")
    
    accuracy_task1 = accuracy_metric.compute(predictions=predictions_task1, references=labels_task1)['accuracy']
    # accuracy_task2 = accuracy_metric.compute(predictions=predictions_task2, references=labels_task2)['accuracy']
    # accuracy_task3 = accuracy_metric.compute(predictions=predictions_task3, references=labels_task3)['accuracy']
    
    macro_precision_task1 = precision_metric.compute(predictions=predictions_task1, references=labels_task1, average='macro')['precision']
    macro_recall_task1 = recall_metric.compute(predictions=predictions_task1, references=labels_task1, average='macro')['recall']
    macro_f1_task1 = f1_metric.compute(predictions=predictions_task1, references=labels_task1, average='macro')['f1']

    micro_precision_task1 = precision_metric.compute(predictions=predictions_task1, references=labels_task1, average='micro')['precision']
    micro_recall_task1 = recall_metric.compute(predictions=predictions_task1, references=labels_task1, average='micro')['recall']
    micro_f1_task1 = f1_metric.compute(predictions=predictions_task1, references=labels_task1, average='micro')['f1']
    
    
    # macro_precision_task2 = precision_metric.compute(predictions=predictions_task2, references=labels_task2, average='macro')['precision']
    # macro_recall_task2 = recall_metric.compute(predictions=predictions_task2, references=labels_task2, average='macro')['recall']
    # macro_f1_task2 = f1_metric.compute(predictions=predictions_task2, references=labels_task2, average='macro')['f1']

    # micro_precision_task2 = precision_metric.compute(predictions=predictions_task2, references=labels_task2, average='micro')['precision']
    # micro_recall_task2 = recall_metric.compute(predictions=predictions_task2, references=labels_task2, average='micro')['recall']
    # micro_f1_task2 = f1_metric.compute(predictions=predictions_task2, references=labels_task2, average='micro')['f1']
    
    # precision_task3 = precision_metric.compute(predictions=predictions_task3, references=labels_task3, average='macro')['precision']
    # recall_task3 = recall_metric.compute(predictions=predictions_task3, references=labels_task3, average='macro')['recall']
    # f1_task3 = f1_metric.compute(predictions=predictions_task3, references=labels_task3, average='macro')['f1']
    
    mean_f1 = 0.5*macro_f1_task1 + 0.5*micro_f1_task1 #平均的宏观f1值


    metrics = { 
        "accuracy_task1": round(accuracy_task1,4),#四舍五入保留四位小数
        "map_task1": round(macro_precision_task1,4),
        "mar_task1": round(macro_recall_task1,4),
        "maf1_task1": round(macro_f1_task1,4),
        "mip_task1": round(micro_precision_task1,4),
        "mir_task1": round(micro_recall_task1,4),
        "mif1_task1": round(micro_f1_task1,4),
        "mean_f1": round(mean_f1,4)
        # "accuracy_task2": round(accuracy_task2,4),
        # "map_task2": round(macro_precision_task2,4),
        # "mar_task2": round(macro_recall_task2,4),
        # "maf1_task2": round(macro_f1_task2,4),
        # "mip_task2": round(micro_precision_task2,4),
        # "mir_task2": round(micro_recall_task2,4),
        # "mif1_task2": round(micro_f1_task2,4),
        # "accuracy_task3": accuracy_task3,
        # "precision_task3": precision_task3,
        # "recall_task3": recall_task3,
        # "f1_task3": f1_task3,
    }

    return metrics

#根据熵确定预测的不准确性，来获取使用gpt-4o进一步预测数据
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def calculate_entropy(probabilities):
    return -np.sum(probabilities * np.log(probabilities + 1e-9))#为了避免对数函数中的零值导致计算错误，因此在每个概率值上加上一个非常小的数

def evaluate_uncertainty_with_entropy(logit):
    probabilities = softmax(np.array(logit))#先计算logits的softmax
    entropy = calculate_entropy(probabilities)
    return entropy
    

def bert_predict(logits: np.ndarray) -> List[Tuple[int, float]]:
    log_softmax: torch.Tensor = torch.tensor(logits).log_softmax(1)#(batch,logits),先取softmax再取log
    predictions: List[Tuple[int, float]] = []

    for x in log_softmax:
        prediction: int = torch.argmax(x).item()
        predictions.append((prediction, x[prediction].item()))

    return predictions


#计算阈值
def cal_entropy_threshold(num_classes):
    max_entropy = np.log(num_classes)  #自然对数，最大熵值
    entropy_threshold = max_entropy*0.01 # 默认阈值为最大熵的70  #这个值
    return entropy_threshold

