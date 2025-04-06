import pandas as pd
import json
from sklearn.metrics import confusion_matrix


#只保留那些类别之间存在双向混淆的情况，即两个类别互相混淆的情况被纳入候选类别集合
def get_bidirectional_confusions(confusion_matrix):
    candidate_labels = {}
    for true_label in range(confusion_matrix.shape[0]):
        bidirectional_confusions = {}
        for pred_label in range(confusion_matrix.shape[1]):
            if confusion_matrix.iloc[true_label, pred_label] > 0 and confusion_matrix.iloc[pred_label, true_label] > 0:
                bidirectional_confusions[pred_label] = confusion_matrix.iloc[true_label, pred_label]
        if bidirectional_confusions:
            candidate_labels[true_label] = bidirectional_confusions
    
    #检查 candidate_labels[true_label]是否在 candidate_labels中
    for true_label in candidate_labels:
        if true_label not in candidate_labels[true_label]:
            print(f"Warning: {true_label} not found in candidate_labels")
            candidate_labels[true_label][true_label] = 1 # 添加自身作为候选标签
    return candidate_labels



if __name__ == "__main__":
    with open("civil_data\CAIL_test.jsonl", "r",encoding="utf-8") as f:    
        true_accu = [json.loads(line)["accu_label"] for line in f]
        f.seek(0)  # 重置文件指针到文件开头
        pred_accu = [json.loads(line)["pre_accu"] for line in f]

    id_to_accu_path = 'civil_data\label2id.json'
    with open(id_to_accu_path, 'r', encoding='utf-8') as file:
        accu2id = json.load(file)
    id_to_accu = {v: k for k, v in accu2id.items()}
    # 把true_accu转化为id
    true_accu = [accu2id.get(accu, accu) for accu in true_accu]
    
    conf_matrix = confusion_matrix(true_accu, pred_accu)

    #获取所有类别的并集
    all_labels = sorted(set(true_accu).union(set(pred_accu)))

    #创建DataFrame
    conf_matrix_df = pd.DataFrame(conf_matrix, index=all_labels, columns=all_labels)
    # Convert the confusion matrix to a DataFrame for better readability
    # conf_matrix_df = pd.DataFrame(conf_matrix, index=sorted(set(true_accu)), columns=sorted(set(pred_accu)))
    # Save the confusion matrix to a CSV file
    # conf_matrix_df.to_csv('civil_data\\confusion_matrix.csv', index=True, header=True, encoding='utf-8')

    confusion_matrix = conf_matrix_df.astype(float)

    #Add diagonal elements to corresponding positions in the confusion matrix
    confusion_matrix_updated = confusion_matrix.copy(deep=True)
    for i in range(confusion_matrix.shape[0]):
        col = confusion_matrix.iloc[:, i]
        for j in range(confusion_matrix.shape[1]):
            confusion_matrix_updated.iloc[i, j] = confusion_matrix_updated.iloc[i, j] + col[j]
    
    # Recompute candidate_labels with updated confusion matrix
    candidate_labels_updated = get_bidirectional_confusions(confusion_matrix_updated)

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
    accu_definition_path = 'civil_database\crime_define.json'
    with open(accu_definition_path, 'r', encoding='utf-8') as file:
        accu_definition = json.load(file)
    
    for true_label, candidates in candidate_labels_with_accu.items():
        for candidate, count in candidates.items():
            candidates[candidate] = accu_definition[candidate]

    # Save the result to a JSON file
    output_path = 'civil_data\\bidirectional_confusions_46.json'
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(candidate_labels_with_accu, file, ensure_ascii=False, indent=4)
