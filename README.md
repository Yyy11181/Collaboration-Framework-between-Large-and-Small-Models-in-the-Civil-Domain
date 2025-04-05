# Collaboration-Framework-between-Large-and-Small-Models-in-the-Civil-Domain

We propose a hybrid framework that collaborates both large and small models to improve charge prediction. Initially, the SMs provide an initial prediction along with a predicted probability distribution. If the maximum predicted probability falls below a threshold, the LLMs step in to reflect and re-predict as needed. Additionally, we construct a confusing charges dictionary based on the confusion matrix of SMs, which helps the LLMs make secondly prediction. 

Furthermore, the proposed framework demonstrates generalisability, making it applicable to different legal fields. We have successfully applied our framework to predict the cause of action in civil law cases, determining the cause of action based on case facts. Our experiments have yielded promising results, showcasing the framework's effectiveness in this context. For further details, please refer to the experiment:
## Content

- [Objective of the Experiment](#objective-of-the-experiment)
- [Dataset and Experimental Details](#dataset-and-experimental-details)
- [Experiment Setting](#experiment-setting)
- [Results Analysis](#results-analysis)

## Objective of the experiment

To validate the generalisability of our proposed framework, we to predict the cause of action in civil law cases, determining the cause of action based on case facts.

## Dataset and experimental details

We use a large scale civil dataset from CAIL-Long, a long text civil and criminal dataset published in [Lawformer](https://www.sciencedirect.com/science/article/pii/S2666651021000176). Each civil case in the dataset is annotated with the cause of action and the relevant laws. The detailed statistics of the dataset are shown in Table 1.

### Table1 Statistics of the civil dataset

| Item | Total |
| --- | --- |
| Training Set Cases | 79676 |
| Validation Set Cases | 16927 |
| Test Set Cases | 17053 |
| cause of action | 257 |
| law | 330 |
| Average Length of Fact Description | 1286.88 |

Our analysis of the dataset's distribution has uncovered long-tailed characteristics. Additionally, the distribution of causes of action in the test data is dominated by high-frequency cases, with the majority of categories containing fewer than 10 instances each.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a67c02bd-de61-4079-bf10-faec823aa1d2" alt="图1" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/a16bd922-38a0-4efa-9201-0668beb99b94" alt="图2" width="500"/></td>
  </tr>
  <tr>
    <td align="center">Distribution of the dataset</td>
    <td align="center">Distribution of the test set</td>
  </tr>
</table>


## Experiment setting

we use the MacBERT as the SMs, and GPT-4o(gpt-4o-08-06) as the LLMs for the secondary prediction. During the fine-tuning of MacBERT, the learning rate is set to 2e-5, with 10 training epochs and a batch size of 16. In the testing pahse, we obtain the results from the LLMs via API calls. Consistent with prior work in criminal field, we maintain the threshold at 0.7, achieving an identification efficiency of 65.0% alongside a 33.8% recall rate for error samples.

## Results analysis

 Table2 shows the relevant experimental results.

### Table2 The Experimental Results

| Method | Ma-P | Ma-R | Ma-F1  | Mi-F1 | Mean F1 |
| --- | --- | --- | --- | --- | --- |
| **GPT-4o** | 47.12 | 49.54 | **45.40** | 74.55 | 59.97 |
| **macbert** | 45.19 | 43.27 | 42.09 | 80.77 | 61.43 |
| **Framework** | 46.91 | 44.82 | 43.81 | **81.35** | **62.58** |

As shown in Table2, our proposed framework outperforms both the SMs-based and LLMs-based baselines, achieving the best overall performance with an average improvement of 1.88% on Mean F1 and 3.69% on Mi-F1. Specifically, the framework shows a more significant improvement in Micro-F1 than in Macro-F1. However, given the low proportion of few-shot categories in the test set, it has not yet surpassed GPT-4o's comprehensive performance on the full test dataset.

Therefore, this experiment demonstrates that our proposed framework can be generalized to different legal fields, demonstrating its generalisability.
