# Multi-Modal Machine Learning for Alzheimer's Disease Diagnosis

## Overview
This project explores the use of a late-fusion ensemble approach combining demographic, genetic, and imaging data for Alzheimer's disease diagnosis. The solution uses machine learning models for each modality and aggregates their predictions using a meta-learner.

## Dataset
The dataset used in this project is the Alzheimer's Disease Neuroimaging Initiative (ADNI) Merge dataset. To access this dataset kindly request for it from the ADNI website. Here is the link https://adni.loni.usc.edu/ .

Alot of data processing was done on the raw data. Kindly contact me if you'll need guidance with that.

## Key Features
- **Imaging Model**: Predicts likelihood of dementia using imaging biomarkers (FDG, AV45, PIB, FBB).
- **Demographics Model**: Evaluates the impact of demographic features like age and ethnicity.
- **Genetics Model**: Analyzes genetic biomarkers (e.g., APOE4, ABETA, TAU).
- **Meta-Learner**: Aggregates predictions from individual models for a robust final decision.

## Why This Project?
The diagnosis of Alzheimer's Disease is a complex challenge requiring insights from multiple data sources. By combining modalities, the model achieves higher accuracy than using any single data source.

## Technologies Used
- **Python**: Core programming language.
- **Scikit-learn**: Machine learning framework.
- **XGBoost**: For gradient boosting models.
- **Streamlit**: For creating an interactive web interface.
- **SMOTE**: To handle class imbalance in datasets.

## How to Use
(Clean and organize the datasets before the steps below. Check on and update the filepaths as required).
1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app using `streamlit run main.py`.
4. Navigate through the app to input data and view predictions.

## Acknowledgements
This project is for educational purposes and should not be used for clinical diagnosis. It highlights the importance of integrating multi-modal data in disease diagnosis and encourages further research in this domain.

## Contact
Feel free to reach out with questions or contributions:
- **Monicah Cherop**
- [cheronai37@gmail.com](mailto:cheronai37@gmail.com)
