# Arabic Machine Reading Comprehension
## Machine Reading Comprehension
Machine Reading Comprehension (MRC), or the ability to read and understand unstructured text and then answer questions about it remains a challenging task for computers. MRC is a growing field of research due to its potential in various enterprise applications, as well as the availability of MRC benchmarking datasets for latin languages (MSMARCO, SQuAD, NewsQA, etc.) except this dataset does not help arabic language thus lack of research and models in arabic language.
## What in this repo?
This repository now contains code and implementation for:
- **AraElectra-Arabic-SQuADv2-QA**: Ouestion Answering model based on [AraElectra](https://huggingface.co/aubmindlab/araelectra-base-discriminator) trained on Arabic-SQuADv2.0 
- **AraElectra-Arabic-SQuADv2-CLS**: Classification model to predict if the questions can be answered in case of SQuADv2.0 based on [AraElectra](https://huggingface.co/aubmindlab/araelectra-base-discriminator) trained on Arabic-SQuADv2.0 
- **Arabic-SQuADv2.0**: New Arabic dataset based on SQuADv2.0 [Read More...](#Dataset)
## Hosted Interface

AraELECTRA-Arabic-SQuADv2-QA/CLS powered Arabic Wikipedia QA system with Streamlit

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/zeyadahmed10/arabic-wikipedia-qa-streamlit/main/streamlit_app.py)
## Results
Model | Arabic-SQuADv2.0 (EM - F1) | 
|:----|:----:|
AraElectra-Arabic-SQuADv2-QA| **65.12** - **71.49** |
- New **state of the art** for arabic language on SQuADv2.0
- This results with the help of the classification model on the [last experiment](https://github.com/zeyadahmed10/Arabic-MRC/blob/main/AraElectraDecoupling-ASQuADv2.ipynb) 
  ### Reproduce Experiments
    - Install requirments 
    ```
    pip install requirments.txt
    ```
    - Run the desired notebook experiment to reproduce same results

## Models

 Model | HuggingFace Model Name | Size / Params|
 ---|:---:|:---:
 AraElectra-Arabic-SQuADv2-QA | [AraElectra-Arabic-SQuADv2-QA](https://huggingface.co/ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA) | 514MB/134M |
 AraElectra-Arabic-SQuADv2-CLS| [AraElectra-Arabic-SQuADv2-CLS](https://huggingface.co/ZeyadAhmed/AraElectra-Arabic-SQuADv2-CLS) |  516MB/135M  
 
 ## Dataset and Compute

For Dataset Source see the [Dataset Section](#Dataset)

Model | Hardware | num of examples (seq len = dynamic len) | Batch Size | Num of epochs 
 ---|:---:|:---:|:---:|:---:
AraElectra-Arabic-SQuADv2-QA | Tesla K80 | 34.5K | 8 | 4 
AraElectra-Arabic-SQuADv2-CLS | Telsa K80 | 76.8K | 8 | 8

## Dataset
Introducing the new Arabic-SQuADV2.0 based on the popular SQuADv2.0 with unanswered questions for more challenging task.
  ### Creation
  - A-NMT on Microsoft Azure Cognitive with state of the arts models on SQuADv2.0 train split
  - Data cleaning and drop uncontiguos span created by NMT
  - Preprocessing using the [Arabert Preprocessor](#Preprocessing)
  ### Size
  Train | Validation | Test 
  ---|:---:|:---:
 76.8K | 9.6K | 9.6K
  ### Structure 
  Arabic-SQuADv2.0 have same structure as SQuADv2.0 for consistency
  ![Data Strucutre](dataset.PNG)
 
