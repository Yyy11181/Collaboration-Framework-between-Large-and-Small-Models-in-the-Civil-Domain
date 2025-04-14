### CivilLaw-LS-Framework

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

### Table 1 Statistics of the civil dataset

| Item | Total |
| --- | --- |
| Training Set Cases | 79676 |
| Validation Set Cases | 16927 |
| Test Set Cases | 17053 |
| cause of action | 257 |
| law | 330 |
| Average Length of Fact Description | 1286.88 |

Furthermore, Figure 1 reveals distinct long-tailed characteristics in the dataset distribution. This imbalance pattern particularly impacts recognition performance for tail classes.

<center>
<img src="https://github.com/user-attachments/assets/a67c02bd-de61-4079-bf10-faec823aa1d2" alt="Distribution" width="400">
</center>

*Figure 1: Long-tailed distribution characteristics*


## Experiment setting

we use the MacBERT as the SMs, and GPT-4o(gpt-4o-08-06) as the LLMs for the secondary prediction. During the fine-tuning of MacBERT, the learning rate is set to 2e-5, with 10 training epochs and a batch size of 16. In the testing pahse, we obtain the results from the LLMs via API calls. Consistent with prior work in criminal field, we maintain the threshold at 0.5, achieving an identification efficiency of 65.0% alongside a 33.8% recall rate for error samples.

## Results analysis

 Table2 shows the relevant experimental results.

### Table 2 The Experimental Results

| Method | Ma-P | Ma-R | Ma-F1  | Mi-F1 | Mean F1 |
| --- | --- | --- | --- | --- | --- |
| **GPT-4o** | 47.12 | 49.54 | 45.40 | 74.55 | 59.97 |
| **macbert** | 45.19 | 43.27 | 42.09 | 80.77 | 61.43 |
| **Framework** | **59.76**| **52.53** |**52.89** | **82.04** | **67.47** |

As demonstrated in Table 2, our proposed framework achieves state-of-the-art performance, surpassing both SM-based and LLM-based baselines across all metrics. The framework exhibits particularly strong improvements in macro-level evaluation, with a 9.15% increase in Macro-F1 compared to a 4.38% gain in Micro-F1, reflecting its enhanced capability in handling few-shot cases. Overall, the framework delivers a consistent 6.77% average improvement in Mean F1 score, demonstrating its robust performance across different evaluation perspectives.

Therefore, this experiment demonstrates that our proposed framework can be generalized to different legal fields, demonstrating its generalisability.
