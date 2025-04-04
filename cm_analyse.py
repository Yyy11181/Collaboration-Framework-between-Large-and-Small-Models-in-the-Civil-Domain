import pandas as pd
import json
from sklearn.metrics import confusion_matrix


# Function to get candidate labels and their counts from the confusion matrix
def get_candidate_labels(confusion_matrix):
    candidate_labels = {}
    for true_label in confusion_matrix.index:
        candidates = confusion_matrix.loc[true_label][confusion_matrix.loc[true_label] > 0]
        candidate_labels[true_label] = candidates.to_dict()
    return candidate_labels

#只保留那些类别之间存在双向混淆的情况，即两个类别互相混淆的情况被纳入候选类别集合
def get_bidirectional_confusions(confusion_matrix):
    candidate_labels = {}
    for true_label in range(confusion_matrix.shape[0]):
        bidirectional_confusions = {}
        for pred_label in range(confusion_matrix.shape[1]):
            if confusion_matrix[true_label, pred_label] > 0 and confusion_matrix[pred_label, true_label] > 0:
                bidirectional_confusions[pred_label] = confusion_matrix[true_label, pred_label]
        if bidirectional_confusions:
            candidate_labels[true_label] = bidirectional_confusions
    return candidate_labels



#
if __name__ == "__main__":
    with open("civil_output\CAIL_test.jsonl", "r") as f:    
        true_accu = [json.loads(line)["accu_label"] for line in f]
        f.seek(0)  # 重置文件指针到文件开头
        pred_accu = [json.loads(line)["pre_accu"] for line in f]

    id_to_accu_path = 'civil_output\label2id.json'
    with open(id_to_accu_path, 'r', encoding='utf-8') as file:
        accu2id = json.load(file)
    id_to_accu = {v: k for k, v in accu2id.items()}
    # 把true_accu转化为id
    true_accu = [accu2id.get(accu, accu) for accu in true_accu]
    
    conf_matrix = confusion_matrix(true_accu, pred_accu)

    confusion_matrix = conf_matrix.astype(float)
    
    # Recompute candidate_labels with updated confusion matrix
    candidate_labels_updated = get_bidirectional_confusions(confusion_matrix)

    # Convert candidate_labels dictionary to use accusations instead of IDs
    candidate_labels_with_accu = {}

    for true_label, candidates in candidate_labels_updated.items():
        true_label_accu = id_to_accu.get((true_label), (true_label))
        candidate_labels_with_accu[true_label_accu] = {}
        for candidate, count in candidates.items():
            candidate_accu = id_to_accu.get((candidate), (candidate))
            candidate_labels_with_accu[true_label_accu][candidate_accu] = count

    #把count变为这个罪名的定义
    #罪名的定义字典打开
    accu_definition_path = 'civil_output\crime_define.json'
    with open(accu_definition_path, 'r', encoding='utf-8') as file:
        accu_definition = json.load(file)
    
    for true_label, candidates in candidate_labels_with_accu.items():
        for candidate, count in candidates.items():
            candidates[candidate] = accu_definition[candidate]

    # Save the result to a JSON file
    output_path = 'civil_output\\bidirectional_confusions.json'
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(candidate_labels_with_accu, file, ensure_ascii=False, indent=4)
