# Empirical Phenotype Classification (EPC) Framework

The Empirical Phenotype Classification (EPC) framework is a predictive-model-agnostic preprocessing pipeline that categorizes missing data mechanisms in Electronic Health Record (EHR) datasets prior to imputation. It classifies missing data into prognostic vectors, protocol blocks, and clerical noise to prevent predictive models from overfitting to administrative workflows or losing true clinical signals.

## Overview
EPC translates clinical deductive reasoning into sequential statistical checks, classifying missingness into three distinct categories:
* **Random Non-Informative Missingness:** Filtered using a dynamic noise floor derived from Symmetric Uncertainty (SU) permutation testing. Resolved via simple median or mode substitution.
* **Outcome-Predictive Missingness:** Identified via a formal diagnostic audit requiring intersectional satisfaction of Information Value (IV > 0.05), Fisher's Exact Test with FDR correction (q < 0.05), and Positive Likelihood Ratio (LR+ > 1.5 or < 0.67). Retained as a binary predictive feature while the underlying continuous variable is neutrally underlaid.
* **Structurally Bundled Missingness:** Mapped using a $k$-way structural scanner (Interaction Information Gain) to preserve routine clinical protocols. Reconstructed using joint multivariate imputation (e.g., MICE) to maintain clinical covariance.

## Dependencies
* Python >= 3.10
* scikit-learn >= 1.3
* xgboost >= 2.0
* pandas
* numpy
* scipy

## Repository Structure
* `data/`: Scripts for generating the synthetic cohort and extracting the MIMIC-IV (v2.2) retrospective cohort.
* `epc_framework/`: Core modules for the SU permutation filter, the clinical criteria gatekeeper (IV, Fisher's, LR+), and the $k$-way structural scanner.
* `evaluation/`: Scripts for executing the dual-classifier evaluation (XGBoost and Logistic Regression) and calculating bootstrapped AUC and Sensitivity metrics.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/dzikos/epc-framework.git](https://github.com/dzikos/epc-framework.git)
cd epc-framework
pip install -r requirements.txt


graph LR
    %% Styling definitions for a minimalist, B&W academic look
    classDef default fill:#ffffff,stroke:#000000,stroke-width:1px,color:#000000,font-family:Times New Roman;
    classDef highlight fill:#f4f4f4,stroke:#000000,stroke-width:2px,color:#000000,font-family:Times New Roman;
    classDef endpoint fill:#e8e8e8,stroke:#000000,stroke-width:2px,color:#000000,font-family:Times New Roman,font-weight:bold;

    %% Left Section: Input
    subgraph INPUT [INPUT: Raw EHR Data]
        direction TB
        A[Data Matrix] --> B[Missing 'NaN' Cells]
    end

    %% Middle Section: EPC Adjudication
    subgraph EPC [EPC Framework Adjudication]
        direction TB
        C[1. Entropy Filter<br/>Isolates Random Clerical Noise]
        D[2. k-way Scanner<br/>Maps Structural Protocol Bundles]
        E[3. Clinical Gatekeeper<br/>Identifies Predictive Phenotypes]
    end

    %% Right Section: Output & Routing
    subgraph OUTPUT [OUTPUT: Reconstructed Dataset]
        direction TB
        F[Standard Imputation<br/>Median Substitution / MICE]
        G[Selective Feature Engineering<br/>Binary Indicators _missing = 1]
    end
    
    H([Predictive Model Deployment])

    %% Data Flow Connections
    B -->|Filter| C
    B -->|Scan| D
    B -->|Audit| E

    C -->|Clerical Noise| F
    D -->|Protocol Bundles| F
    E -->|Outcome Predictive| G

    F --> H
    G --> H

    %% Apply Classes
    class INPUT,EPC,OUTPUT default;
    class A,B,C,D,E,F,G highlight;
    class H endpoint;
