
# Deploying the HEARTS framework to aid model explicability for hate speech detection in the context of morphologically rich languages like Hindi

This repository contains the full workflow for applying the HEARTS explainability framework to the HateDay Hindi hate‑speech dataset using the IndicBERT model. 
I have included exploratory data analysis, dataset preparation, model training, model explainability using SHAP and LIME, replication of the HEARTS baseline, 
and generation of all figures used in the accompanying research poster. The project was undertaken as part of coursework for MSc AI for Sustainable Development at UCL.

---

## Repository Structure

### Core Notebooks

Replicating_HEARTS_Model.ipynb
    Reproduces the HEARTS baseline model. King et al. (2024) used ALBERT.

Replicating_HEARTS_Explainability.ipynb
    Replicates the explainability analysis from the HEARTS framework (King et al.,2024)using SHAP.
    
Exploratory_data_analysis_HateDay.ipynb
    Analysis of class imbalance, label distribution, token characteristics, and word clouds for hate, non_hate speech.

Data_Preparation_HateDay.ipynb
    Preprocessing pipeline for HateDay, including cleaning the dataset for model training.

IndicBERT_Model_Training.ipynb
    Training pipeline for IndicBERT with class weights and a learning‑rate scheduler.

IndicBERT_Model_Explainability.ipynb
    SHAP and LIME explanations for correct and incorrect predictions.

Generating_SHAP_LIME_Plots.ipynb
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

## Explainability Approach

The project applies two complementary explainability methods, following the HEARTS framework(King et al., 2024):

SHAP: Global and local token‑importance attribution.
LIME: Local perturbation‑based explanations of individual predictions.

Comparative plots are used to demonstrate how the model behaves on correct and misclassified examples. I found the model to be reliant on identity markers, somewhat failing to capture contextual cues in Hindi. This maybe due to the lack of diversity of data, limited number of samples in the training dataset, and possible labelling bias in the training dataset.


---

## Critical Reflection on SDGs

This work connects to several Sustainable Development Goals:

Primarly, it can potentially contribite to SDG 16 (Peace, Justice, and Strong Institutions)
It can also be mapped to SDG 5 (Gender Equality), SDG 10 (Reduced Inequalities), and SDG 9 (Industry, Innovation and Infrastructure).

I recommend participatory inclusion of stakeholders in data collection, annotation, and validation to overcome the challenges identified here.
In addition, Hindi hate speech detection models could benefit from more lexically aware XAI methods. 
For real world deployment, model explicability could benefit from policy discussions on data rights, and responsible inference. To an extent, this could ensure safe digital inclusion for disadvantaged sections of society, fostering SDG 5, 10. 16, and SDG 9. 


---

## Citations


    King, T., Wu, Z., Koshiyama, A., Kazim, E., & Treleaven, P. (2024).
HEARTS: A holistic framework for explainable, sustainable and robust text stereotype detection. 
arXiv. https://arxiv.org/abs/2409.11579
    
    Tonneau, M., Liu, D., Malhotra, N., Hale, S. A., Fraiberger, S. P., Orozco-Olvera, V., & Röttger, P. (2025).
HateDay: Insights from a global hate speech dataset representative of a day on Twitter. 
arXiv. https://arxiv.org/abs/2411.15462

    Kakwani, D., Kunchukuttan, A., Golla, S., N. C., G., Bhattacharyya, A., Khapra, M. M., & Kumar, P. (2020). 
IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages. 
In Findings of the Association for Computational Linguistics: EMNLP 2020.


---

