# 🚄 Shinkansen Travel Experience Prediction – Hackathon Project 🚄

This project was developed as part of a **hackathon** to predict **passenger satisfaction** (`Overall_Experience`) on the Shinkansen Bullet Train using **machine learning**. The model analyzes survey responses and travel data to help optimize passenger experience.

---

## 🏆 **Hackathon Overview**
- **Hackathon Name:** Shinkansen Travel Experience  
- **Date:** Jan 17, 12:00 AM - Jan 19, 11:59 PM  
- **Team Members:** Robert Lupo 
- **Challenge Statement:** Build a predictive model to determine whether a passenger had a satisfactory experience based on their survey responses and travel history.

---

## 📊 **Project Goal**
- **Objective:** Predict whether a passenger is **satisfied (`1`)** or **unsatisfied (`0`)** with their overall experience.  
- **Impact:** By identifying key factors influencing satisfaction, train operators can **improve services**, reduce negative feedback, and **increase customer retention**.  

---

## 📁 **Data Description**
### **📜 Dataset Provided in Hackathon**
1. **Train Data** (Labeled):
   - `Traveldata_train.csv` – Passenger demographics & travel details.
   - `Surveydata_train.csv` – Customer satisfaction ratings (target: `Overall_Experience`).

2. **Test Data** (Unlabeled for submission):
   - `Traveldata_test.csv` – Same structure as training travel data.
   - `Surveydata_test.csv` – Same structure as training survey data.

### **📝 Key Features**
| Feature | Description |
|---------|------------|
| `ID` | Unique passenger identifier |
| `Gender` | Male / Female |
| `Age` | Age of the passenger (numeric/binned) |
| `Customer_Type` | Loyal or Disloyal Customer |
| `Travel_Class` | Business / Economy |
| `Travel_Distance` | Distance traveled (km/miles) |
| `DepartureDelay_in_Mins` | Delay at departure (minutes) |
| `ArrivalDelay_in_Mins` | Delay at arrival (minutes) |
| `Seat_Comfort`, `Cleanliness`, `Onboard_Wifi_Service`, etc. | Survey-based satisfaction ratings |
| **`Overall_Experience`** | 🎯 **Target Variable** (0 = Not Satisfied, 1 = Satisfied) |

---

## Repository Structure
shinkansen-travel-experience/
├── data/
│   ├── Traveldata_train.csv
│   ├── Surveydata_train.csv
│   ├── Traveldata_test.csv
│   └── Surveydata_test.csv
├── notebook/
│   ├── shinkansen-travel-experience.ipynb       # Main Jupyter notebook for EDA & modeling
├── outputs/
│   ├── submission.csv           # Final prediction output for Kaggle
├── README.md                    # Project overview & instructions
└── LICENSE                       # Open-source license

## 🚀 Getting Started

Follow these steps to set up the project and run the model.

### **1️⃣ Clone the Repository**
```
git clone https://github.com/your-username/shinkansen-travel-experience.git
cd shinkansen-travel-experience
```
### **2️⃣ Install Dependencies
```
pip install pandas numpy scikit-learn xgboost lightgbm catboost seaborn matplotlib
```
### **3️⃣ Obtain the Data
- **Ensure your CSV files are in the data/ directory, or update the paths in the notebook if they are stored elsewhere.**
- **If using a Kaggle dataset, download and place it in data/.**
### **4️⃣ Run the Jupyter Notebook
```
jupyter notebook
```
- **Open notebook/shinkansen_experience.ipynb and execute all cells.**
- **Follow the notebook workflow for data preprocessing, model training, evaluation, and predictions.**
### **5️⃣ Generate Predictions
- **Run the final model on the test data.**
- **Save the predictions to a CSV file:**
```
submission.to_csv('submission.csv', index=False)
```
- **Upload submission.csv to Kaggle or the hackathon platform for evaluation.**

## 🏗 Modeling Approach

### **1️⃣ Data Preprocessing**
- **Handling Missing Values**:
  - Imputed missing numerical values using **mean**.
  - Categorical features were filled using **mode (most frequent value)**.
  - Certain missing values were replaced with "Unknown" where appropriate.
  
- **Feature Engineering**:
  - **Binning `Age`**: Converted into categorical bins (`25`, `35`, `45`, `60`, `80`).
  - **Ordinal Encoding**: Converted ratings like "Poor" → `1`, "Excellent" → `5`.
  - **One-Hot Encoding**: Applied to nominal categorical variables (`Gender`, `Travel_Class`, `Customer_Type`).
  - **Scaling**: Used `StandardScaler` (Z-score normalization) for numerical features.

- **Data Splitting**:
  - **Train-Test Split**: `80%` for training, `20%` for testing.
  - **Cross-Validation**: Used **Stratified K-Fold (3-fold CV)** to ensure balanced class distribution.

---

### **2️⃣ Machine Learning Models Tested**
- **Baseline Model**:
  - **Logistic Regression**: Simple linear model for benchmarking.

- **Tree-Based Models**:
  - **Random Forest**: Ensemble learning with multiple decision trees.
  - **XGBoost**: Gradient boosting with advanced regularization.
  - **LightGBM**: Faster, memory-efficient boosting algorithm.
  - **CatBoost**: Handles categorical features natively, reducing preprocessing effort.

---

### **3️⃣ Hyperparameter Tuning**
- Used **`RandomizedSearchCV`** for quick optimization across multiple models.
- Parameters tuned:
  - **LightGBM**: `num_leaves`, `n_estimators`, `learning_rate`.
  - **CatBoost**: `depth`, `learning_rate`, `iterations`.
  - **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`.

- Optimized models were retrained on the full dataset before final predictions.

---

### **4️⃣ Model Evaluation**
- **Primary Metric**:  
  - **Accuracy**: Proportion of correct predictions.

- **Secondary Metrics** (for class imbalance handling):
  - **Precision, Recall, F1-score** (via `classification_report`).
  - **Confusion Matrix**: Evaluated misclassification patterns.

- **Feature Importance**:
  - Extracted feature importances from tree-based models.
  - Visualized top contributing features affecting `Overall_Experience`.

---

## 📊 Results & Key Findings

### **1️⃣ Best Performing Model**
- **🏆 CatBoost** achieved the highest accuracy of **95.76%**.
- **LightGBM** performed similarly with **95.64%**.
- **Random Forest** and **XGBoost** also performed well but were slightly less accurate.

---

### **2️⃣ Key Factors Influencing Passenger Satisfaction**
Using feature importance analysis from tree-based models, the **top drivers** of satisfaction were:

#### **🔝 Top 5 Most Important Features**
| Rank | Feature | Importance |
|------|---------|------------|
| 1️⃣  | **Seat Comfort** | 📈 Strongest predictor of satisfaction |
| 2️⃣  | **Cleanliness** | ✨ Highly correlated with satisfaction |
| 3️⃣  | **Onboard Service** | 🛎 Passengers who rated onboard service poorly were more likely to be dissatisfied |
| 4️⃣  | **Arrival Time Convenience** | ⏳ Directly impacted overall experience |
| 5️⃣  | **Travel Class** | 🏢 Business class passengers had consistently higher satisfaction |

---

### **3️⃣ Interesting Insights from Data Analysis**
✅ **Seat Comfort & Cleanliness** were **the biggest factors** affecting customer satisfaction.  
✅ **Onboard Service & Online Support** also had a significant impact.  
✅ **Business Class passengers had much higher satisfaction** than Economy travelers.  
✅ **Longer travel distances slightly reduced satisfaction**, possibly due to fatigue.  
✅ **Wi-Fi quality had a weaker impact** than expected, possibly because many travelers didn’t use it.

---

### **4️⃣ Confusion Matrix & Misclassifications**
- **Precision & Recall** for satisfied customers (`Overall_Experience = 1`) were both **above 94%**, meaning the model was highly effective at distinguishing between satisfied and unsatisfied passengers.
- **Common Misclassifications**:
  - Some **satisfied passengers were predicted as unsatisfied**, likely due to **delays** despite positive onboard ratings.
  - A few **unsatisfied passengers were predicted as satisfied**, possibly because they gave decent ratings in certain areas but had strong complaints in others.

---

### **5️⃣ Areas for Future Improvement**
🔹 **Further hyperparameter tuning** to improve model generalization.  
🔹 **Consider adding interaction terms** (e.g., combining "Seat Comfort" and "Travel Class").  
🔹 **Explore time-series factors**, such as peak travel seasons affecting satisfaction.  
🔹 **Investigate class imbalance strategies** (though the dataset appeared fairly balanced).  

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project as long as proper attribution is given.

See the [LICENSE](LICENSE) file for full details.

---

## 📩 Contact & Acknowledgments

### **👤 Author**
- **Robert Lupo**  
- 📧 Email: treylupo1197@gmail.com  
- 🔗 LinkedIn: [[robertlupo1997]](https://www.linkedin.com/in/robertlupo1997/)  
- 🏆 GitHub: [[robertlupo1997]](https://github.com/robertlupo1997)  

### **🙏 Acknowledgments**
- **Hackathon Organizers**: MIT Institute for Data, Systems, and Society (IDSS) for providing the challenge and dataset.  
- **Machine Learning Community**: Special thanks to Kaggle discussions, Stack Overflow, and open-source contributors for valuable insights on hyperparameter tuning, feature engineering, and modeling strategies.  
- **Python & ML Libraries**:  
  - **scikit-learn**: For model training and evaluation  
  - **XGBoost, LightGBM, CatBoost**: For advanced boosting techniques  
  - **Pandas & NumPy**: For efficient data processing  
  - **Seaborn & Matplotlib**: For visualizing insights  

### **💡 Want to Collaborate?**
🚀 If you're interested in working together or have feedback, feel free to reach out!  
Contributions, suggestions, and PRs are **always welcome**!  

> ⭐ If you found this project helpful, give it a **star** on GitHub!
