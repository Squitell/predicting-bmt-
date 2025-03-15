# Medical Decision Support Application

## Project Report

### Introduction
This project focuses on developing a decision-support application to assist physicians in predicting the success rate of pediatric bone marrow transplants. The objective is to create a reliable and explainable machine learning model that ensures transparency in medical predictions.

### Objectives
The main objectives of this project are:

- **Developing** an explainable machine learning model to predict transplant success.
- **Ensuring transparency** through SHAP-based explainability.
- **Building** a user-friendly interface using Streamlit or Flask.
- **Implementing** professional software development practices, including GitHub and CI/CD automation.
- **Exploring prompt engineering** by documenting AI-generated prompts used in the workflow.

### Dataset Used
The dataset utilized in this project is available at:  
[Bone Marrow Transplant Children Dataset](https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children)

### Data Processing and Analysis
- **Handling Missing Values:** The dataset was analyzed for missing values and appropriate techniques were used to handle them.
- **Outliers Detection:** Any significant outliers were identified and addressed.
- **Class Imbalance:** The dataset showed an imbalance (approximately 60% survival and 40% non-survival). Techniques applied to handle this included:
  - Oversampling using SMOTE.
  - Undersampling.
  - Adjusting class weights.
- **Feature Correlation:** Highly correlated features were identified and managed accordingly.

### Machine Learning Models Tested
To ensure optimal performance, multiple machine learning models were evaluated, including:

- Random Forest Classifier
- XGBoost Classifier
- Support Vector Machine (SVM)
- LightGBM Classifier

**Evaluation Metrics Used:** The models were assessed based on ROC-AUC, accuracy, precision, recall, and F1-score.

### Final Deliverables
The final project submission includes:
- ✅ A well-structured and documented codebase.
- ✅ Detailed exploratory data analysis reports.
- ✅ A trained ML model integrated with SHAP-based explainability.
- ✅ An intuitive web-based interface.
- ✅ A functional CI/CD pipeline via GitHub Actions.
- ✅ A memory optimization module.
- ✅ Comprehensive documentation covering all aspects of the project.


### Conclusion
This project has provided valuable insights into the intersection of machine learning and medical decision support. By integrating SHAP explainability and best software development practices, we have developed a reliable tool that aids physicians in making informed transplant decisions.

**Thank you for reading!**
