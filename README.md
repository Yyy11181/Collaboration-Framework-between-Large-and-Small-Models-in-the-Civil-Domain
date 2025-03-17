# Collaboration-Framework-between-Large-and-Small-Models-in-the-Civil-Domain
# Collaboration Framework between Large and Small Models in the Civil Domain

We propose a hybrid framework that collaborates both large and small models to improve charge prediction. Initially, the SMs provide an initial prediction along with a predicted probability distribution. If the maximum predicted probability falls below a predefined threshold, the LLMs step in to reflect and re-predict as needed. Additionally, we construct a confusing charges dictionary based on the confusion matrix of  SMs, which helps the LLMs make secondly  prediction. 

The proposed framework demonstrates extensive scalability, making it applicable to various legal fields and adaptable across different types of legal cases. Therefore, we successfully apply our framework to predict the cause of action within the civil domain.

For further details, please refer to the experiment:

## Additional Experiment

## Content

- [Objective of the Experiment](https://www.notion.so/Collaboration-Framework-between-Large-and-Small-Models-in-the-Civil-Domain-1b9cc726aadf804aaa98dd8b57954a1b?pvs=21)
- [Dataset and Experimental Details](https://www.notion.so/Collaboration-Framework-between-Large-and-Small-Models-in-the-Civil-Domain-1b9cc726aadf804aaa98dd8b57954a1b?pvs=21)
- Prompting for LLMs
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

## Promopt in cause of action prediction task

In cause of action prediction task, we design a specific prompt for the LLMs in the secondary prediction following our proposed two-stage legal inference prompt.
    "prompt_template": """
    {
        "Assume you are an expert in the field of civil law. Please determine the cause of action for the case based on the factual part of the case. Note that this case only involves one applicable cause of action, please predict the most suitable cause of action from the given list of candidate causes of action.",
        "Let's analyze the case according to the following steps": {
            "1. Case Analysis and Preliminary Result Reflection": {
                "Current Case Analysis": "Analyze the nature of the civil legal relationship involved in the case, clarify the focus of the dispute, and refine the main litigation requests of the plaintiff and the defense points of the defendant.",
                "Case Similarity and Difference Analysis": "Based on the current case analysis, analyze the similarities and differences between the preliminary predicted cause of action applicable cases and the current case from aspects such as litigation requests and dispute focus.",
                "Reflect on Preliminary Prediction": "Based on the analysis of case similarities and differences, combined with the nature of the legal relationship involved in the preliminary predicted cause of action, reflect on whether the preliminary predicted cause of action conforms to the actual situation of the current case."
            },
            "2. Whether to Re-predict": "Yes/No. If the preliminary predicted cause of action does not conform to the actual situation of the current case, please re-predict the cause of action of the case.",
            "3. Re-prediction Steps": {
                "Cause of Action Screening": "Match the causes of action in the candidate list with the case facts and litigation requests one by one, exclude options that obviously do not conform to the nature of legal relationships, and retain related causes of action.",
                "Candidate Cause of Action Analysis": "Further analyze the screened causes of action. If there are principal and subordinate relationship causes of action, determine the cause of action based on the principal legal relationship. If there is a specific fourth-level cause of action, it should be applied preferentially. If there is no corresponding fourth-level cause of action, check the third-level, second-level until the first-level cause of action in turn. If there are competing causes of action, the final cause of action needs to be determined according to the litigation requests of the parties.",
                "Determine the Most Applicable Cause of Action": "Determine the most applicable cause of action for the current case and provide an explanation."
            }
        },
        "Current Case Information":
        {
            "Case Number": "{id}",
            "Fact Description": "{case_description}",
            "Preliminary Predicted Cause of Action": "{pre_accu}",
            "Preliminary Predicted Cause of Action Applicable Case": {
                "Case Description": "{similar_case_description}"
            },
            "Definition of Preliminary Predicted Cause of Action": "{crime_details}",
            "Given Candidate Causes of Action": "{candidate_ste}"
        },
        "Please output in json format, the output structure requirements": {
            "id": "<case_number>",
            "1. Case Analysis and Preliminary Result Reflection": {
                "Current Case Analysis": "<current_case_analysis>",
                "Case Similarity and Difference Analysis": "<case_similarity_and_difference_analysis>",
                "Reflect on Preliminary Prediction": "<reflect_on_preliminary_prediction>"
            },
            "2. Whether to Re-predict": "<yes/no>",
            "3. Re-prediction Steps": {
                "Cause of Action Screening": "<cause_of_action_screening/not_needed>",
                "Candidate Cause of Action Analysis": "<candidate_cause_of_action_analysis/not_needed>",
                "Determine the Most Applicable Cause of Action": "<determine_the_most_applicable_cause_of_action/not_needed>"
            },
            "4. Prediction Result": "<prediction_result>"
        }
    }



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
