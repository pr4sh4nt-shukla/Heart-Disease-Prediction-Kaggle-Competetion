# 🫀 Heart Disease Prediction - Kaggle Playground Series S6E2

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E2-20BEFF)](https://www.kaggle.com/competitions/playground-series-s6e2)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)](https://lightgbm.readthedocs.io/)

## 🎯 Overview

This project is my first Kaggle competition submission for the **Playground Series Season 6 Episode 2** - Heart Disease Prediction. The goal is to predict whether a patient has heart disease (**Presence** or **Absence**) based on 13 clinical features using multiple machine learning models.

The project covers the full machine learning pipeline — from raw data exploration to model training, evaluation, and final submission — comparing 6 different models to identify the best performer.

## 🛠️ Tech Stack

- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
- **Analysis Type:** Binary Classification

## 📈 Pipeline Phases

### 1. Data Loading & Exploration
- Loaded 630,000 training samples with 13 clinical features
- Verified zero missing values across all columns
- Analyzed target class distribution: **55% Absence** vs **45% Presence**
- Visualized class balance using bar and pie charts

### 2. Data Preprocessing
- Encoded target variable: `Absence → 0`, `Presence → 1`
- Applied **80/20 train-test split** with stratification to preserve class balance
- Used `StandardScaler` for distance-based models (Logistic Regression, KNN)
- Tree-based models trained on unscaled features for optimal performance

### 3. Model Training & Evaluation
Six models were trained and compared across four key metrics:

**Distance-Based Models (scaled features):**
* **Logistic Regression** - Strong linear baseline with balanced class weights
* **KNN** - Distance-based classifier with 5 neighbors

**Tree-Based Models (unscaled features):**
* **Random Forest** - Ensemble with 200 trees, highest recall at 91.59%
* **XGBoost** - Gradient boosting with calculated `scale_pos_weight`
* **Gradient Boosting** - Sklearn implementation with learning rate 0.05
* **LightGBM** - Fast gradient boosting, best overall accuracy

### 4. Visualization & Analysis
* **Performance Comparison:** Bar charts across Accuracy, Precision, Recall, F1-Score
* **Radar Chart:** Top 3 model comparison on all metrics simultaneously
* **ROC Curves:** AUC scores for all 6 models
* **Confusion Matrices:** Per-model breakdown of true/false positives and negatives
* **Feature Importance:** Top 15 features ranked by Random Forest importance scores

## 🏆 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LightGBM** 🏆 | **0.8889** | **0.8826** | 0.8676 | 0.8750 |
| Gradient Boosting | 0.8885 | 0.8821 | 0.8674 | 0.8747 |
| XGBoost | 0.8879 | 0.8688 | 0.8832 | **0.8760** |
| Logistic Regression | 0.8839 | 0.8671 | 0.8752 | 0.8711 |
| Random Forest | 0.8731 | 0.8216 | **0.9159** | 0.8662 |
| KNN | 0.8718 | 0.8621 | 0.8501 | 0.8560 |

* **Best Model:** LightGBM with **88.89%** validation accuracy
* **Kaggle Public Score:** 0.88321
* **Average Accuracy Across All Models:** 88.24%
* **Key Insight:** Random Forest achieved the highest recall (91.59%), making it ideal for minimizing missed heart disease cases

## 📂 Repository Structure

* `Completed_Kaggle_Completion.ipynb`: Complete notebook with all code, training, and visualizations
* `submission.csv`: Final predictions submitted to Kaggle
* `requirements.txt`: Necessary libraries to reproduce the environment
* `README.md`: Project documentation

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Prashant Shukla**  
📧 Email: prashantshukla8851@gmail.com  
💼 LinkedIn: [Prashant Shukla](https://www.linkedin.com/in/prashant-shukla-58ba19373)  
🏆 Kaggle: [Prashant Shukla](https://www.kaggle.com/prashantshukla44)

**Project Link:** [https://github.com/pr4sh4nt-shukla/Heart-Disease-Prediction-Kaggle-Competetion](https://github.com/pr4sh4nt-shukla/Heart-Disease-Prediction-Kaggle-Competetion)

---
⭐ **If you found this project helpful, please consider giving it a star!** ⭐
