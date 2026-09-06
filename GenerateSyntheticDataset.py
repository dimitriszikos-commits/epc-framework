import pandas as pd
import numpy as np
from scipy.special import expit

def generate_bmc_synthetic_icu(n_samples=10000, random_seed=42, 
                               alpha=1.5, beta_bmi=2.0, beta_lac=-2.5, 
                               gamma_1=-0.08, gamma_2=-1.2):
    """
    Generates synthetic ICU data using a Latent Data-Generating Process (DGP).
    Incorporates an unobserved acute clinical gestalt and high continuous variance
    to simulate scenarios where missingness conveys behavioral risk that MICE cannot infer.
    """
    np.random.seed(random_seed)
    
    # ==========================================
    # 1. LATENT ACUITY & BASELINE COVARIATES
    # ==========================================
    # S represents the unobserved, true baseline physiological deterioration
    S = np.random.normal(0, 1, n_samples)
    
    # Acute Event represents a sudden deterioration seen by doctors but not recorded in baseline labs
    acute_event = np.random.normal(0, 1.5, n_samples)
    
    age = np.random.normal(65, 15, n_samples).clip(18, 95)
    comorbidities = np.random.binomial(1, p=expit((age - 50) / 15)) + \
                    np.random.binomial(1, p=expit((age - 45) / 10))
    median_income = np.random.normal(60000, 20000, n_samples).clip(15000, 200000)
    
    # ==========================================
    # 2. TRUE CLINICAL VALUES (Correlated with Latent S, but noisier)
    # ==========================================
    # Increased standard deviation prevents MICE from perfectly interpolating the missing values
    bmi_true = 28 + (1.5 * S) + np.random.normal(0, 8.0, n_samples)
    lactate_true = 2.0 + (1.2 * S) + np.random.normal(0, 1.5, n_samples).clip(0, None)
    creatinine_true = 1.0 + (0.4 * S) + (0.2 * comorbidities) + np.random.normal(0, 0.4, n_samples)
    wbc_true = 8.0 + (1.8 * S) + np.random.normal(0, 3.0, n_samples)
    severity_index_true = 50 + (10 * S) + np.random.normal(0, 5, n_samples)
    
    # ==========================================
    # 3. CLINICAL OUTCOMES (Decoupled from direct variables)
    # ==========================================
    # Mortality Y is driven by baseline S AND the unobserved acute event
    epsilon_1 = np.random.normal(0, 0.5, n_samples)
    logit_death = (alpha * S) + (0.8 * acute_event) + epsilon_1 - 2.2
    died = np.random.binomial(1, p=expit(logit_death))
    
    los_days = np.where(
        died == 1,
        np.random.gamma(shape=2.0, scale=3.0, size=n_samples),
        np.random.gamma(shape=4.0, scale=2.5, size=n_samples) + (S * 2)
    ).round().clip(1, 45)
    
    df = pd.DataFrame({
        'Age': age.round(1), 'Comorbidities': comorbidities,
        'Median_Income': median_income.round(0),
        'BMI': bmi_true.round(1), 'Severity_Index': severity_index_true.round(1),
        'Lactate': lactate_true.round(2), 'Creatinine': creatinine_true.round(2), 
        'WBC': wbc_true.round(1), 'LOS_Days': los_days, 'Died': died
    })

    # ==========================================
    # 4. PROBABILISTIC MISSINGNESS INJECTION
    # ==========================================
    
    # --- A. Outcome-Predictive Missingness ---
    # Perceived severity is driven by S, the acute event, and independent noise
    epsilon_2 = np.random.normal(0, 1.0, n_samples)
    perceived_severity = S + acute_event + epsilon_2
    
    # BMI: Stronger missingness correlation with acute deterioration (beta_bmi = 2.0)
    prob_missing_bmi = expit((beta_bmi * perceived_severity) - 2.5)
    df.loc[np.random.rand(n_samples) < prob_missing_bmi, 'BMI'] = np.nan
    
    # Lactate: Stronger missingness correlation with perceived stability (beta_lac = -2.5)
    prob_missing_lactate = expit((beta_lac * perceived_severity) - 2.0)
    df.loc[np.random.rand(n_samples) < prob_missing_lactate, 'Lactate'] = np.nan

    # --- B. Structurally Bundled Missingness ---
    epsilon_3 = np.random.normal(0, 0.5, n_samples)
    logit_skip_bundle = (gamma_1 * age) + (gamma_2 * comorbidities) + epsilon_3 + 3.0
    prob_skip_bundle = expit(logit_skip_bundle)
    
    bundle_mask = np.random.rand(n_samples) < prob_skip_bundle
    df.loc[bundle_mask, ['Creatinine', 'WBC']] = np.nan

    # --- C. Adversarial Negative Controls ---
    # Pure Random Noise
    df.loc[np.random.rand(n_samples) < 0.04, 'Severity_Index'] = np.nan
    
    # Covariate-Correlated Noise (Independent of S, acute event, and Y)
    prob_missing_income = expit(0.08 * (age - 60) - 1.5)
    df.loc[np.random.rand(n_samples) < prob_missing_income, 'Median_Income'] = np.nan

    return df

if __name__ == "__main__":
    print("Generating Tuned Latent DGP EPC Validation Cohort...")
    df_synthetic = generate_bmc_synthetic_icu()
    
    df_synthetic.to_csv("epc_synthetic_validation.csv", index=False)
    
    print(f"\nDataset 'epc_synthetic_validation.csv' created successfully.")
    print(f"Total Patients: {len(df_synthetic)}")
    print(f"Mortality Rate: {df_synthetic['Died'].mean():.2%}")
    
    print("\nMissingness Overview:")
    print((df_synthetic.isna().sum() / len(df_synthetic) * 100).round(1).astype(str) + '%')