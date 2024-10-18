## Understanding the Problem: What Is Credit Risk?

Credit risk is the possibility of borrowers not repaying loans on time, which is monitored through **Days Past Due (DPD)**. Loans overdue beyond a certain threshold, such as **90 days**, become **Non-Performing Assets (NPAs)**, posing a heightened default risk. **Portfolio at Risk (PAR)** measures the **Outstanding Principal (OSP)** of overdue loans, helping banks manage loan portfolio health.

### Objective
The primary goal of this project is to create a predictive model that can:
- Analyze factors influencing credit risk and loan approval.
- Classify individuals into **four priority categories** based on their credit profiles to streamline loan approval and minimize NPAs.

---

## Data Sources

### 1. CIBIL Data (60 features):
- Aggregates credit history across multiple banks.
- **Key features:**
  - **Credit Score**: Ranges from 469 to 811, a key indicator of creditworthiness.
  - **Trade Lines**: Information on loans and credit accounts across banks.
  - **Payment History**: Tracks payment behavior over time (e.g., 6 and 12 months), including delinquency indicators like **Days Past Due (DPD)**.

### 2. Internal Bank Data (25 features):
- Provides a granular view of the customer’s relationship with the bank.
- **Key features:**
  - **Total/Active Trade Lines**: Number of credit accounts showing engagement with credit products.
  - **Assets**: Details on assets held with the bank, contributing to the financial profile.

---

## Handling Missing Data

![Handling Missing Data](<img width="582" alt="i1" src="https://github.com/user-attachments/assets/f5b3bd8b-2624-42c9-a32b-3ce8a0c4b0e7">)

One major challenge was **missing data**, common in financial datasets due to data entry errors or unreported information. For instance, the CIBIL dataset had 35k missing values for the delinquency column (`max_del`), accounting for 70% of the data. Instead of imputing, this column was removed to avoid bias. Columns with more than **10,000 missing values** (represented by `-99999`) were similarly removed. This ensured that 70-80% of the data was retained, maintaining model reliability.

---

## Feature Engineering and Selection

- **Chi-Square Test**: All categorical features were retained since p-values ≤ 0.05, indicating a statistically significant relationship with the the target variable.
- **Variance Inflation Factor (VIF)**: Removed numerical features with VIF > 6 to reduce multicollinearity, reducing the feature set from **72 to 39**.
- **ANOVA**: Applied to remaining numerical features, retaining those with p-values ≤ 0.05, resulting in **37 statistically significant predictors** for the approved .

---

## Model Selection: Multiclass Classification

After cleaning the data and selecting relevant features, the focus was on categorizing loan applicants into **four priority categories (P1 to P4)** to enhance loan approval decisions based on risk profiles and repayment likelihood. The **XGBoost** algorithm was selected as the base model and tuned to achieve:
- **Train accuracy**: 81%
- **Test accuracy**: 78%

<p float="left">
  <img src="<img width="230" alt="i3" src="https://github.com/user-attachments/assets/b31bde9b-32de-4645-bb3e-8f0414bd037e">  " width="49%" />
  <img width="254" alt="i4" src="https://github.com/user-attachments/assets/81029296-5824-45a5-8f13-3260511b250e">
</p>

---

## Performance Evaluation

- Beyond accuracy; Precision, recall, and F1 scores were calculated for each class to thoroughly assess model performance.
- This evaluation revealed class-specific issues, with P3 showing lower accuracy due to an **ambiguous decision boundary** within the credit score range.

---

## Addressing Class P3 Ambiguity

<p float="left">
  <img width="266" alt="i6" src="https://github.com/user-attachments/assets/bee49c7c-6fed-408b-9af9-5aded081e06c">
 " width="49%" />
  <img width="266" alt="i6" src="https://github.com/user-attachments/assets/bee49c7c-6fed-408b-9af9-5aded081e06c">" width="49%" />
</p>

- **P3 Credit Score Range**: Spans from **489 to 776**, much broader than other classes (P1, P2, P4), affecting classification accuracy.
- Re-evaluating how credit scores are utilized for P3 could significantly improve model accuracy.

