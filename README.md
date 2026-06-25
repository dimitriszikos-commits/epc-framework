# Empirical Phenotype Classification (EPC) Framework

**Supervised Missingness Classification and Context-Based Imputation for Clinical Data**
Missing data in clinical datasets frequently acts as a predictive clinical proxy rather than random noise. Traditional imputation inherently assumes data is Missing At Random (MAR) or Missing Completely At Random (MCAR), which can mask the severity of worsening patients by substituting unobserved physiological vectors with population averages. 

The **Empirical Phenotype Classification (EPC)** framework is a supervised data-engineering pipeline designed to systematically characterize the underlying mechanism of missingness prior to data imputation. By isolating true clinical phenotypes (Missing Not At Random - MNAR) from protocol-based omissions (MAR) and administrative noise (MCAR), EPC preserves clinical risk vectors and optimizes predictive modeling for highly interpretable algorithms like Logistic Regression.

## 🧠 The EPC Pipeline

The framework operates as a sequential, model-agnostic decision engine that evaluates structural missingness through four distinct phases

### 1. The Dynamic Noise Floor (Systemic Structure)
Uses **Symmetric Uncertainty (SU)** permutation testing (100 random shuffles of the target outcome) to establish a dataset-specific noise floor[cite: 2]. This computationally inexpensive filter catches and removes random background clerical errors (MCAR).

### 2. The Asymmetric Risk Triad (Clinical Severity)
Variables surviving the noise filter are evaluated against three mathematicized clinical criteria to isolate clinical risk:
*   **Information Value (IV > 0.05):** Ensures the missingness possesses sufficient binary separation power.
*   **Fisher's Exact Test (p < 0.05):** Ensures statistical confidence, even in small/sparse sample sizes.
*   **Positive Likelihood Ratio (LR+ > 1.5 or < 0.67):** Determines the asymmetric clinical direction (e.g., acute deterioration vs. visual triage of stability).

### 3. The k-Way Structural Scanner (Redundancy Mapping)
A combinatorial scanner that maps redundant, bundled missingness (e.g., routine metabolic laboratory panels missing simultaneously)[cite: 2]. Using Joint Symmetric Uncertainty (JSU) and Interaction Information Gain, it bundles variables into MAR Structural Blocks to prevent donor-pool collapse during imputation.

### 4. Directed Data Reconstruction
Rather than universally applying a single imputer, EPC routes features to the mathematically appropriate algorithm based on their diagnostic classification:
*   **MCAR (Pure Noise):** Unsupervised median or mode substitution.
*   **MAR (Structural Protocol):** Joint Multivariate Imputation by Chained Equations (MICE).
*   **MNAR (Clinical Phenotype):** Bypasses standard imputation and routes to outcome-stratified matching. Selective binary indicators are generated exclusively for these variables to prevent feature-space inflation.

---

## 🚀 Repository Structure

*   `data/`
    *   `synthetic/`: Scripts to generate the engineered synthetic ICU cohort ($n=10000$).
    *   `mimic/`: Query scripts for extracting the 15 core clinical features from the MIMIC-IV (v2.2) database. *(Note: Raw MIMIC-IV data must be obtained directly via PhysioNet)*
*   `src/`
    *   `epc_engine.py`: Core algorithm containing the Adjudication Triad, permutation testing, and k-way scanner.
    *   `imputation_router.py`: The context-based routing logic (Median vs. MICE vs. Stratified Match).
    *   `evaluation.py`: Model training and evaluation scripts comparing EPC to baseline MICE using XGBoost and Logistic Regression.
*   `notebooks/`
    *   `01_epc_demonstration.ipynb`: Step-by-step walkthrough of the EPC classification pipeline.

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/yourusername/epc-framework.git](https://github.com/yourusername/epc-framework.git)
cd epc-framework
pip install -r requirements.txt
