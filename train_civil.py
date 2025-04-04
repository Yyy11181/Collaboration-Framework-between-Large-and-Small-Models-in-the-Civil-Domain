from transformers import (BertTokenizer, Trainer,TrainingArguments,set_seed)
from util_civil import JsonlDataset,BertForMultiTaskClassification,compute_metrics,EvalResultsCallback
import os

#初始化变量，随后会由 torch.distributed.launch 根据需要自动分配合适的 GPU
local_rank = int(os.getenv('LOCAL_RANK', -1))


if __name__ == '__main__':
    set_seed(42)
    model_card: str = 'Mac-bert' #更换的模型地址/
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_card)
    #创建数据集
    model: BertForMultiTaskClassification = BertForMultiTaskClassification.from_pretrained(
        model_card, num_labels_task1 = 257 )
    train_file_path = 'civil/train_with_id.json'
    val_file_path = 'civil/dev_with_id.json'

    train_dataset = JsonlDataset(train_file_path, tokenizer, max_length = 512)#不是列表的形式，是对象的形式
    val_dataset = JsonlDataset(val_file_path, tokenizer, max_length = 512)

    #创建数据加载器
    batch_size = 16
    #定义训练参数
    training_args = TrainingArguments(
        output_dir = 'results_civil',          # 输出目录
        eval_strategy = "epoch",     # 评估策略，每个epoch评估一次s
        save_strategy = 'epoch',
        learning_rate = 2e-5,              # 学习率
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs = 10,              # 训练epoch数
        weight_decay = 0.01,               # 权重衰减,
        load_best_model_at_end=True,
        metric_for_best_model='eval_mean_f1',
        greater_is_better=True,
        optim='adamw_torch',
        report_to='none',
        dataloader_drop_last = True,  # 在多GPU训练中丢弃不完整的批次
        gradient_accumulation_steps = 1,  # 梯度累积步骤数
        fp16=True,  # 使用混合精度训练
        fp16_opt_level='O1',  # 混合精度优化级别
        local_rank= local_rank,  # 如果使用torch.distributed.launch启动脚本则设置为-1
        ddp_find_unused_parameters = False)

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics = compute_metrics,
        callbacks = [EvalResultsCallback(output_file="results_civil/evaluation_results.jsonl")]
    )

    ##开始训练,从这个地方进入到模型中去
    trainer.train()
    save_directory = 'macbert_civil'

    # 检查并创建目录
    os.makedirs(save_directory, exist_ok=True)

    # 保存分词器和模型
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

