
# Deploying the HEARTS framework to aid model explicability for hate speech detection in the context of morphologically rich languages like Hindi

## Introduction

This repository explores the application of the HEARTS explainability framework to Hindi hate speech detection using IndicBERT. While HEARTS provides a structured approach for evaluating stereotype detection systems, it has primarily been developed and validated in high-resource language settings.

This project examines whether these evaluation assumptions transfer to a morphologically rich, low-resource context by combining baseline replication, model training, and explainability analysis using SHAP and LIME.

The findings highlight practical challenges in translating abstract explainability frameworks into reliable and measurable system behaviour in multilingual settings.

---

## Research Quesiton

Can the HEARTS framework (King et al., 2024) be reliably operationalised for hate speech detection in a morphologically rich, low-resource language such as Hindi, and do its evaluation assumptions remain valid in this setting?

## Methodology 

The project follows a structured pipeline to test the applicability of the HEARTS framework (King et al., 2024) in a new linguistic setting:

Replication of the HEARTS baseline model (ALBERT-based)
Training a Hindi hate speech classifier using IndicBERT
Application of explainability methods (SHAP, LIME) to analyse model behaviour
Comparative analysis of predictions across correct and misclassified examples

This setup enables evaluation of both model performance and the reliability of interpretability signals in a low-resource context.

## Repository Structure

### Core Notebooks

Replicating_HEARTS_Model.ipynb:
    Reproduces the HEARTS baseline model. King et al. (2024) used ALBERT.

Replicating_HEARTS_Explainability.ipynb:
    Replicates the explainability analysis from the HEARTS framework (King et al.,2024)using SHAP.
    
Exploratory_data_analysis_HateDay.ipynb:
    Analysis of class imbalance, label distribution, token characteristics, and word clouds for hate, non_hate speech.

Data_Preparation_HateDay.ipynb:
    Preprocessing pipeline for HateDay, including cleaning the dataset for model training.

IndicBERT_Model_Training.ipynb:
    Training pipeline for IndicBERT with class weights and a learning‑rate scheduler.

IndicBERT_Model_Explainability.ipynb
    SHAP and LIME explanations for correct and incorrect predictions.

Generating_SHAP_LIME_Plots.ipynb:
    Produces global heatmaps and example‑level explanation plots used in the poster.

### Additional Files  
indicbert_hate_model_v2, indicbert_hate_model_v2_final
    Trained model checkpoints.

shap_results.csv, lime_results.csv
    Explanation outputs.

full_results_albertv2.csv
    Model replication results for ALBERT, for comparison against the baseline.

hate_wordcloud.png, nonhate_wordcloud.png
    Word clouds from EDA.

hindi_hatespeech_cleaned.csv, sampled_data.csv
    Preprocessed datasets used locally.

---

## Explainability Approach

The project applies two complementary explainability methods:

SHAP: Global and local token-level attribution
LIME: Local perturbation-based explanations

These methods are used to analyse both correct and misclassified predictions, enabling comparison between expected and observed model behaviour.

---
## Key Findings

- The HEARTS framework can be partially reproduced in a Hindi context, but its evaluation assumptions do not fully transfer.
- Explainability methods (SHAP, LIME) indicate that the model often relies on identity markers rather than contextual understanding.
- Dataset limitations (size, diversity, and potential labelling bias) make evaluation signals unreliable.
- This creates ambiguity: it becomes unclear whether observed behaviour reflects genuine model limitations or insufficient measurement.

---

## Interpretation

These results suggest that evaluation and explainability frameworks depend heavily on the availability of reliable data and context-aware signals. In low-resource settings, both datasets and metrics may fail to capture the underlying behaviour of the model, limiting the interpretability of results.

This raises a broader concern: evaluation pipelines may give a misleading impression of model reliability when applied outside the conditions in which they were originally developed.

---

## Installation

Install all required packages using:

pip install -r requirements.txt

The requirements file lists all dependencies necessary to reproduce the results, including Transformers, HuggingFace datasets, SHAP, LIME, and visualisation libraries.

---

## Dataset Access

This project uses the HateDay Hindi hate‑speech dataset (Tonneau et al., 2025).
The dataset is not included in the repository due to licensing restrictions. You must request tthe authors for access.

To obtain the dataset:

1. Request access from the authors via HuggingFace:
   https://huggingface.co/datasets/Tonneau/hateday

2. Download the files and place them in a local directory such as:

data/hateday.csv

A Hindi stopword list used during preprocessing is sourced from(publically available):
https://www.kaggle.com/datasets/rsrishav/wordcloud-hindi-font

---

## Running the Project

To reproduce the entire workflow, run the notebooks in the following order:

1. Replicating_HEARTS_Model.ipynb
2. Replicating_HEARTS_Explainability.ipynb
3. Exploratory_data_analysis_HateDay.ipynb
4. Data_Preparation_HateDay.ipynb
5. IndicBERT_Model_Training.ipynb
6. IndicBERT_Model_Explainability.ipynb
7. Generating_SHAP_LIME_Plots.ipynb

All code cells are documented and reproducible, with fixed seeds where appropriate.

---

## Broader Implications

This work connects to SDG 16 (Peace, Justice, and Strong Institutions), with additional relevance to SDGs 5, 10, and 9.

The findings highlight the need for:

improved dataset diversity and annotation practices
context-aware explainability methods
clearer evaluation standards for deployment in multilingual settings

More broadly, the project suggests that responsible deployment requires not only better models, but also more robust and context-sensitive evaluation frameworks.

---

## Refrerences


    King, T., Wu, Z., Koshiyama, A., Kazim, E., & Treleaven, P. (2024).HEARTS: A holistic framework for explainable, sustainable and robust text stereotype detection. arXiv. https://arxiv.org/abs/2409.11579
    
    Tonneau, M., Liu, D., Malhotra, N., Hale, S. A., Fraiberger, S. P., Orozco-Olvera, V., & Röttger, P. (2025).HateDay: Insights from a global hate speech dataset representative of a day on Twitter. arXiv. https://arxiv.org/abs/2411.15462

    Kakwani, D., Kunchukuttan, A., Golla, S., N. C., G., Bhattacharyya, A., Khapra, M. M., & Kumar, P. (2020). IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages. In Findings of the Association for Computational Linguistics: EMNLP 2020.


---

