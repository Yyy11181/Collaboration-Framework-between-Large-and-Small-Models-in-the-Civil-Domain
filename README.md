# Collaboration-Framework-between-Large-and-Small-Models-in-the-Civil-Domain
We propose a hybrid framework that collaborates both large and small models to improve charge prediction. Initially, the SMs provide an initial prediction along with a predicted probability distribution. If the maximum predicted probability falls below a predefined threshold, the LLMs step in to reflect and re-predict as needed. Additionally, we construct a confusing charges dictionary based on the confusion matrix of  SMs, which helps the LLMs make secondly  prediction. 

The proposed framework demonstrates extensive scalability, making it applicable to various legal fields and adaptable across different types of legal cases. Therefore, we successfully apply our framework to predict the cause of action within the civil domain.

For further details, please refer to the experiment:

## Additional Experiment

## Content

- [Objective of the Experiment](https://www.notion.so/Collaboration-Framework-between-Large-and-Small-Models-in-the-Civil-Domain-1b9cc726aadf804aaa98dd8b57954a1b?pvs=21)
- [Dataset and Experimental Details](https://www.notion.so/Collaboration-Framework-between-Large-and-Small-Models-in-the-Civil-Domain-1b9cc726aadf804aaa98dd8b57954a1b?pvs=21)
- Experiment setting
- [Results Analysis](https://www.notion.so/Collaboration-Framework-between-Large-and-Small-Models-in-the-Civil-Domain-1b9cc726aadf804aaa98dd8b57954a1b?pvs=21)

## Objective of the experiment

To validate the scalability of our proposed framework, we apply it to cause of action prediction task within the civil domain, aiming to assist the  judge to category the civil diputes and improving the judicial efficiency.

## Dataset and experimental details

We use a large scale civil dataset from CAIL-long, a long text civil and criminal dataset published by  xiao et al\cite{XIAO202179},. Each civil case in the  is annotated with the causes of actions and the relevant laws. The detailed statistics of the dataset are shown in Table 2.

### Table1 Statistics of the civil dataset

| Item | Total |
| --- | --- |
| Training Set Cases | 79676 |
| Validation Set Cases | 16927 |
| Test Set Cases | 17053 |
| cause of action | 257 |
| law | 330 |
| Average Length of Fact Description | 1286.88 |
## Experiment setting

we use the Mac-bert as the SMs, and GPT-4o(2018-08-06) as the LLMs for the secondary prediction. During the fine-tuning of MacBERT, the learning rate is set to 2 × 10−5
, with 10 training epochs and a batch size of 16. In the testing pahse, we obtain the results from the LLMs via API calls.The optimal max predicted probality threshold is 0.4.

## Results analysis

 Table3 shows the relevant experimental results.

### Table3 The Experimental Results

| Method | Ma-P | Ma-R | Ma-F1  | Mi-F 1 | Mean F1 |
| --- | --- | --- | --- | --- | --- |
| **GPT-4o** | 0.4712 | 0.4954 | 0.454 | 0.7455 | 0.5997 |
| **macbert** | 0.4519 | 0.4327 | 0.4209 | 0.8077 | 0.6143 |
| **Framework** | 0.4685 | 0.4411 | 0.4339 | **0.8123** | **0.6231** |

As shown in Table3, our proposed framework outperforms both the SMs-based
and LLMs-based baselines, achieving the best performance with an average improvement of 1.61% on mean F1.  Therefore, this experiment demonstrates that our proposed framework can be generalized to other different legal field , demonstrating a degree of applicability and scalability.
