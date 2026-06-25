import pandas as pd
import numpy as np
import itertools
from scipy.stats import fisher_exact
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier 
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# MODULE 1: NORMALIZED ENTROPY (SYMMETRIC UNCERTAINTY)
# ==========================================

def calc_entropy(array):
    counts = pd.Series(array).value_counts(normalize=True)
    return -sum(counts * np.log2(counts + 1e-9))

def calc_su(x, y):
    ent_x = calc_entropy(x)
    ent_y = calc_entropy(y)
    if ent_x == 0 or ent_y == 0: return 0.0
    
    xy = pd.Series(list(zip(x, y)))
    ent_xy = calc_entropy(xy)
    ig = ent_x + ent_y - ent_xy
    return max(0.0, 2.0 * ig / (ent_x + ent_y))

def get_optimal_bins(df_obs, col_name, target_name, max_bins=5):
    if df_obs[col_name].dtype in ['object', 'category'] or df_obs[col_name].nunique() <= max_bins:
        return df_obs[col_name].astype(str)

    best_su = -1
    best_binned = None

    for q in range(2, max_bins + 1):
        try:
            binned = pd.qcut(df_obs[col_name].astype(float), q=q, duplicates='drop')
            if len(binned.unique()) < 2: continue
            su = calc_su(binned, df_obs[target_name].astype(int))
            if su > best_su:
                best_su = su
                best_binned = binned
        except ValueError:
            continue

    # FIX: Prevents UnboundLocalError if all pd.qcut attempts fail. 
    # Safely falls back to treating it as categorical noise.
    return best_binned if best_binned is not None else df_obs[col_name].astype(str)

def missingness_su(df, col_name, target_name):
    is_missing = df[col_name].isna().astype(int)
    target = df[target_name].astype(int)
    return calc_su(is_missing, target)

def baseline_su(df, col_name, target_name):
    df_obs = df[~df[col_name].isna()].copy()
    if len(df_obs) == 0: return 0.0
    binned_values = get_optimal_bins(df_obs, col_name, target_name)
    target_obs = df_obs[target_name].astype(int)
    return calc_su(binned_values, target_obs)

def joint_missingness_su(df, cols, target_name):
    joint_state = pd.Series([""] * len(df), index=df.index)
    for col in cols:
        joint_state += "_" + df[col].isna().astype(int).astype(str)
    return calc_su(joint_state, df[target_name].astype(int))

# ==========================================
# MODULE 2: ASYMMETRIC RISK (IV, LR+, FISHER'S)
# ==========================================

def calc_asymmetric_metrics(df, col_name, target_name):
    target = df[target_name].astype(int)
    is_missing = df[col_name].isna()
    
    missing_pos_raw = target[is_missing].sum()
    missing_neg_raw = (target == 0)[is_missing].sum()
    obs_pos_raw = target[~is_missing].sum()
    obs_neg_raw = (target == 0)[~is_missing].sum()
    
    # Return 1.0 for OR if the contingency table is empty
    if missing_pos_raw + missing_neg_raw == 0: 
        return 0.0, 1.0, 1.0, 1.0

    table = [[missing_pos_raw, missing_neg_raw], [obs_pos_raw, obs_neg_raw]]
    
    # Fisher's Exact Test returns both the Odds Ratio and the P-Value
    odds_ratio, p_value = fisher_exact(table)
    
    missing_pos = missing_pos_raw + 1
    missing_neg = missing_neg_raw + 1
    total_pos = target.sum() + 2
    total_neg = (target == 0).sum() + 2
    
    pct_pos_missing = missing_pos / total_pos
    pct_neg_missing = missing_neg / total_neg
    
    woe = np.log(pct_pos_missing / pct_neg_missing)
    iv = (pct_pos_missing - pct_neg_missing) * woe
    
    sens_void = missing_pos / total_pos
    spec_void = (obs_neg_raw + 1) / total_neg
    lr_plus = sens_void / (1 - spec_void + 1e-9)
    
    return iv, lr_plus, p_value, odds_ratio
# ==========================================
# MODULE 3: PERMUTATION ENGINE & STRUCTURAL SCANNER 
# ==========================================

def calculate_noise_floor(df, feature_cols, target_name, iterations=100, confidence_level=0.95):
    print(f"  -> Running Permutation Test ({iterations} iterations) to establish noise floor...")
    target_array = df[target_name].values
    noise_su_scores = []
    
    missing_vars = [col for col in feature_cols if df[col].isna().any()]
    if not missing_vars: return 0.0001 
    
    missing_masks = {col: df[col].isna().astype(int) for col in missing_vars}
    
    for _ in range(iterations):
        shuffled_target = np.random.permutation(target_array)
        for col in missing_vars:
            noise_su_scores.append(calc_su(missing_masks[col], shuffled_target))
            
    dynamic_threshold = np.percentile(noise_su_scores, confidence_level * 100)
    print(f"  -> 95% Confidence Noise Floor established at SU = {dynamic_threshold:.6f}")
    return max(dynamic_threshold, 1e-5)

def find_missingness_redundancies(df, feature_cols, target_name, dynamic_threshold, max_k=3):
    missing_vars = [col for col in feature_cols if df[col].isna().any()]
    individual_su = {col: missingness_su(df, col, target_name) for col in missing_vars}
    
    active_vars = [col for col in missing_vars if individual_su[col] > dynamic_threshold]
    redundant_variables_map = {} 
    
    interaction_threshold = -abs(dynamic_threshold / 2)
    
    # The k-Way Structural Scanner
    for k in range(2, max_k + 1):
        # PRUNING: Only check variables that haven't already been mapped at a lower level
        current_pool = [v for v in active_vars if v not in redundant_variables_map]
        
        # Stop searching if we don't have enough variables left to form a k-sized bundle
        if len(current_pool) < k: break 
        
        for combo in itertools.combinations(current_pool, k):
            joint_su = joint_missingness_su(df, combo, target_name)
            sum_individual_su = sum([individual_su[col] for col in combo])
            interaction_info = joint_su - sum_individual_su
            
            if interaction_info < interaction_threshold: 
                for var in combo:
                    if var not in redundant_variables_map:
                        trigger_string = " & ".join([c for c in combo if c != var])
                        redundant_variables_map[var] = {'Trigger_Combo': f"[{k}-Way] {trigger_string}"}
                        
    return redundant_variables_map, individual_su
# ==========================================
# MODULE 4: THE ADJUDICATION MATRIX (TRIAD)
# ==========================================
def run_epc_diagnostics(df, feature_cols, target_col, search_depth=3):
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    dynamic_threshold = calculate_noise_floor(df, feature_cols, target_col)
    known_redundancies, individual_su = find_missingness_redundancies(df, feature_cols, target_col, dynamic_threshold, max_k=search_depth)
    
    results = []
    su_dictionary = {}
    
    print("  -> Adjudicating Triad Diagnostics (SU, IV, LR+)...")
    for col in feature_cols:
        base_su = baseline_su(df, col, target_col)
        miss_su = individual_su.get(col, 0.0)
        
        # Unpack the 4 variables now, including odds_ratio
        iv, lr_plus, p_val, odds_ratio = calc_asymmetric_metrics(df, col, target_col)
        
        su_dictionary[col] = base_su if base_su > 0 else dynamic_threshold
        linked_str = ""
        
        # Determine boolean state flags
        is_global = miss_su > dynamic_threshold
        is_clinical = iv > 0.05 and (lr_plus > 1.5 or lr_plus < 0.67) and p_val < 0.05
        is_redundant = col in known_redundancies

        if is_clinical and is_global:
            classification = "Systemic Clinical Phenotype"
        elif is_clinical and not is_global:
            classification = "Extreme Localized Phenotype"
        elif not is_clinical and is_global:
            if is_redundant:
                classification = "Structural Block (Redundant)"
                linked_combo = known_redundancies[col]['Trigger_Combo']
                linked_str = str(linked_combo) 
            else:
                classification = "Pervasive Informative Missingness"
        else:
            classification = "Non-Informative / Pure Noise"
                
        results.append({
            'Variable': col,
            'Missing_SU': round(miss_su, 6), 'IV_Smoothed': round(iv, 4), 
            'LR_Plus': round(lr_plus, 2), 'Odds_Ratio': round(odds_ratio, 4), 'Fisher_P': round(p_val, 4),
            'Classification': classification,
            'Linked_Variables': linked_str
        })
    return pd.DataFrame(results), su_dictionary

# ==========================================
# MODULE 5: EPAI IMPUTER (ADAPTIVE DIRECTED ROUTING)
# ==========================================

class EPC_Imputer:
    def __init__(self, su_dict=None, cat_cols=None, diagnostic_report=None):
        self.su_dict = su_dict or {}
        self.cat_cols = cat_cols or []
        self.diagnostic_report = diagnostic_report
        
        # Routing Lists
        self.mcar_vars = []
        self.mar_vars = []
        self.mnar_vars = []
        
        # Sub-Imputers
        self.mcar_imputer_num = SimpleImputer(strategy='median')
        self.mcar_imputer_cat = SimpleImputer(strategy='most_frequent')
        self.mar_imputer = IterativeImputer(random_state=42, max_iter=10)
        
        self.donors = None
        self.y_donors = None
        self.telemetry = {'MCAR_Resolved': 0, 'MAR_Resolved': 0, 'MNAR_Resolved': 0}

    def fit(self, X_train, y_train=None):
        self.donors = X_train.copy()
        if y_train is not None:
            self.y_donors = y_train.copy()
            
        if self.diagnostic_report is not None:
            for _, row in self.diagnostic_report.iterrows():
                var = row['Variable']
                cls = row['Classification']
                
                if var not in X_train.columns: continue
                    
                # Manuscript Rule 1: Pure administrative noise (MCAR)
                if "Pure Noise" in cls or "MCAR" in cls or "Non-Informative" in cls:
                    self.mcar_vars.append(var)
                # Manuscript Rule 2: Protocol skips & Structural Blocks (MAR)
                elif "MAR" in cls or "Redundant" in cls or "Structural Block" in cls:
                    self.mar_vars.append(var)
                # Manuscript Rule 3: Systemic Phenotypes (MNAR)
                else:
                    self.mnar_vars.append(var)

        # --- FIT MCAR ENGINE (Unsupervised Baseline) ---
        mcar_num = [v for v in self.mcar_vars if v not in self.cat_cols]
        mcar_cat = [v for v in self.mcar_vars if v in self.cat_cols]
        
        if mcar_num: self.mcar_imputer_num.fit(X_train[mcar_num])
        if mcar_cat: self.mcar_imputer_cat.fit(X_train[mcar_cat])
        
        # --- FIT MAR ENGINE (Joint Dimensional Block) ---
        if self.mar_vars:
            # MICE natively evaluates variables jointly to preserve structural covariance
            self.mar_imputer.fit(X_train[self.mar_vars])
            
        return self

    def transform(self, X_test, y_test=None):
        X_imp = X_test.copy()
        self.telemetry = {'MCAR_Resolved': 0, 'MAR_Resolved': 0, 'MNAR_Resolved': 0}
        
        # 1. ROUTE MCAR -> Simple Unsupervised Substitution
        mcar_num = [v for v in self.mcar_vars if v not in self.cat_cols]
        mcar_cat = [v for v in self.mcar_vars if v in self.cat_cols]
        
        if mcar_num and X_imp[mcar_num].isna().any().any():
            self.telemetry['MCAR_Resolved'] += X_imp[mcar_num].isna().sum().sum()
            X_imp.loc[:, mcar_num] = self.mcar_imputer_num.transform(X_imp[mcar_num])
            
        if mcar_cat and X_imp[mcar_cat].isna().any().any():
            self.telemetry['MCAR_Resolved'] += X_imp[mcar_cat].isna().sum().sum()
            X_imp.loc[:, mcar_cat] = self.mcar_imputer_cat.transform(X_imp[mcar_cat])

        # 2. ROUTE MAR -> Joint Imputation (MICE)
        if self.mar_vars and X_imp[self.mar_vars].isna().any().any():
            self.telemetry['MAR_Resolved'] += X_imp[self.mar_vars].isna().sum().sum()
            X_imp.loc[:, self.mar_vars] = self.mar_imputer.transform(X_imp[self.mar_vars])

        # 3. ROUTE MNAR -> Outcome-Stratified Recovery (Preserving Asymmetric Risk)
        for var in self.mnar_vars:
            missing_mask = X_imp[var].isna()
            if not missing_mask.any(): continue
            
            self.telemetry['MNAR_Resolved'] += missing_mask.sum()
            
            for idx in X_imp[missing_mask].index:
                # Attempt Stratified Matching if labels are available (Training Phase)
                if self.y_donors is not None and y_test is not None:
                    target_outcome = y_test.loc[idx]
                    valid_donors = self.donors[(self.y_donors == target_outcome) & (self.donors[var].notna())]
                else:
                    # Fallback for Inference Phase (Relies on _missing flags added in Mod 6)
                    valid_donors = self.donors[self.donors[var].notna()]
                    
                if len(valid_donors) == 0:
                    valid_donors = self.donors[self.donors[var].notna()]
                    
                # Donor substitution (using median/mode of valid stratified donor pool)
                if len(valid_donors) > 0:
                    if var in self.cat_cols:
                        X_imp.loc[idx, var] = valid_donors[var].mode()[0]
                    else:
                        X_imp.loc[idx, var] = valid_donors[var].median()
                else:
                    X_imp.loc[idx, var] = 0 # Extreme fallback
                    
        print(f"      [Imputer Telemetry] MCAR (Simple): {self.telemetry['MCAR_Resolved']} cells | MAR (Joint MICE): {self.telemetry['MAR_Resolved']} cells | MNAR (Stratified): {self.telemetry['MNAR_Resolved']} cells")
        
        return X_imp


# ==========================================
# MODULE 6: DATASET GENERATORS
# ==========================================

def _encode_categoricals(X_train, X_test):
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    # Filter out columns that are 100% missing in the training set
    valid_cats = [c for c in cat_cols if X_train[c].notna().any()]
    
    if valid_cats:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[valid_cats] = cat_imputer.fit_transform(X_train[valid_cats])
        X_test[valid_cats] = cat_imputer.transform(X_test[valid_cats])
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[valid_cats] = encoder.fit_transform(X_train[valid_cats])
        X_test[valid_cats] = encoder.transform(X_test[valid_cats])
    return X_train, X_test, valid_cats

def prepare_raw(df_train, df_test, target_col):
    X_train, y_train = df_train.drop(columns=[target_col]).copy(), df_train[target_col]
    X_test, y_test = df_test.drop(columns=[target_col]).copy(), df_test[target_col]
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    for col in cat_cols:
        val_map = {val: float(i) for i, val in enumerate(X_train[col].dropna().unique())}
        X_train[col], X_test[col] = X_train[col].map(val_map), X_test[col].map(val_map)
    return X_train, y_train, X_test, y_test

def prepare_mice(df_train, df_test, target_col):
    X_train, y_train = df_train.drop(columns=[target_col]).copy(), df_train[target_col]
    X_test, y_test = df_test.drop(columns=[target_col]).copy(), df_test[target_col]
    X_train, X_test, cat_cols = _encode_categoricals(X_train, X_test)
    
    num_cols = X_train.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
    
    # Only pass numerical columns to MICE that have at least one valid observation
    valid_nums = [c for c in num_cols if X_train[c].notna().any()]
    
    if len(valid_nums) > 0:
        imputer = IterativeImputer(max_iter=10, random_state=42)
        X_train.loc[:, valid_nums] = imputer.fit_transform(X_train[valid_nums])
        X_test.loc[:, valid_nums] = imputer.transform(X_test[valid_nums])
    return X_train, y_train, X_test, y_test

def prepare_epc(df_train, df_test, target_col, report, su_dict):
    X_train, y_train = df_train.drop(columns=[target_col]).copy(), df_train[target_col]
    X_test, y_test = df_test.drop(columns=[target_col]).copy(), df_test[target_col]
    
    for _, row in report.iterrows():
        var_name, classification = row['Variable'], row['Classification']
        if var_name not in X_train.columns: continue
        
        # ENFORCEMENT: Only pass missingness flags to XGBoost if the Triad 
        # proved the void is clinically predictive (MNAR).
        if classification in ["Extreme Localized Phenotype", 
                              "Systemic Clinical Phenotype"]:
            
            X_train[f"{var_name}_missing"] = X_train[var_name].isna().astype(int)
            X_test[f"{var_name}_missing"] = X_test[var_name].isna().astype(int)
            
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    imputer = EPC_Imputer(su_dict=su_dict, cat_cols=cat_cols, diagnostic_report=report)
    
    X_train_imp = imputer.fit(X_train, y_train).transform(X_train, y_train)
    X_test_imp = imputer.transform(X_test)
    
    for c in X_train_imp.columns: 
        X_train[c], X_test[c] = X_train_imp[c], X_test_imp[c]
        
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = enc.transform(X_test[cat_cols])
                
    return X_train, y_train, X_test, y_test

def prepare_epc_raw(df_train, df_test, target_col, report):
    X_train, y_train = df_train.drop(columns=[target_col]).copy(), df_train[target_col]
    X_test, y_test = df_test.drop(columns=[target_col]).copy(), df_test[target_col]
    
    for _, row in report.iterrows():
        var_name, classification = row['Variable'], row['Classification']
        if var_name not in X_train.columns: continue
        
        # ENFORCEMENT: Only pass missingness flags to XGBoost if the Triad 
        # proved the void is clinically predictive (MNAR).
        if classification in ["Extreme Localized Phenotype", 
                              "Systemic Clinical Phenotype"]:
            X_train[f"{var_name}_missing"] = X_train[var_name].isna().astype(int)
            X_test[f"{var_name}_missing"] = X_test[var_name].isna().astype(int)
            
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    for col in cat_cols:
        val_map = {val: float(i) for i, val in enumerate(X_train[col].dropna().unique())}
        X_train[col], X_test[col] = X_train[col].map(val_map), X_test[col].map(val_map)
                
    return X_train, y_train, X_test, y_test

# ==========================================
# MODULE 7: EVALUATION & MAIN
# ==========================================

def bootstrap_metrics(y_true, y_pred_proba, y_pred, n_bootstraps=1000, random_state=42):
    """Calculates 95% CIs for AUC and Sensitivity."""
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = np.array(y_pred)
    rng = np.random.RandomState(random_state)
    
    aucs, sensitivities = [], []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred_proba), len(y_pred_proba))
        if len(np.unique(y_true[indices])) < 2: continue
        
        aucs.append(roc_auc_score(y_true[indices], y_pred_proba[indices]))
        
        pos_mask = y_true[indices] == 1
        if np.sum(pos_mask) > 0:
            sensitivities.append(np.sum((y_pred[indices] == 1) & pos_mask) / np.sum(pos_mask))
            
    auc_ci = (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)) if aucs else (0,0)
    sens_ci = (np.percentile(sensitivities, 2.5), np.percentile(sensitivities, 97.5)) if sensitivities else (0,0)
    
    return auc_ci, sens_ci


def evaluate_dual_models(X_train, y_train, X_test, y_test, name, has_nans=False):
    num_pos = sum(y_train)
    spw = (len(y_train) - num_pos) / num_pos if num_pos > 0 else 1.0
    
    # --- 1. EVALUATE XGBOOST ---
    xgb_model = GridSearchCV(XGBClassifier(scale_pos_weight=spw, random_state=42, eval_metric='logloss'), 
                         {'max_depth': [3, 4], 'learning_rate': [0.05, 0.1], 'n_estimators': [100]}, 
                         scoring='roc_auc', cv=3, n_jobs=-1, verbose=0).fit(X_train.astype(float), y_train)
    
    y_pred_proba_xgb = xgb_model.predict_proba(X_test.astype(float))[:, 1]
    y_pred_xgb = xgb_model.predict(X_test.astype(float))
    auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    sens_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)['1']['recall']
    
    auc_xgb_ci, sens_xgb_ci = bootstrap_metrics(y_test, y_pred_proba_xgb, y_pred_xgb)
    
    # --- 2. EVALUATE LOGISTIC REGRESSION ---
    X_tr_lr = X_train.astype(float).copy()
    X_te_lr = X_test.astype(float).copy()
    
    if has_nans:
        imputer = SimpleImputer(strategy='median')
        X_tr_lr = imputer.fit_transform(X_tr_lr)
        X_te_lr = imputer.transform(X_te_lr)
        
    scaler = StandardScaler()
    X_tr_lr = scaler.fit_transform(X_tr_lr)
    X_te_lr = scaler.transform(X_te_lr)
    
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_tr_lr, y_train)
    
    y_pred_proba_lr = lr_model.predict_proba(X_te_lr)[:, 1]
    y_pred_lr = lr_model.predict(X_te_lr)
    auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
    sens_lr = classification_report(y_test, y_pred_lr, output_dict=True)['1']['recall']
    
    auc_lr_ci, sens_lr_ci = bootstrap_metrics(y_test, y_pred_proba_lr, y_pred_lr)

    print(f"\n{'='*70}\n {name}\n{'='*70}")
    print(f"Features: {X_train.shape[1]}")
    print(f"  -> XGBoost   | AUC: {auc_xgb:.4f} [{auc_xgb_ci[0]:.4f}-{auc_xgb_ci[1]:.4f}] | Sens: {sens_xgb:.4f} [{sens_xgb_ci[0]:.4f}-{sens_xgb_ci[1]:.4f}]")
    print(f"  -> LogistReg | AUC: {auc_lr:.4f} [{auc_lr_ci[0]:.4f}-{auc_lr_ci[1]:.4f}] | Sens: {sens_lr:.4f} [{sens_lr_ci[0]:.4f}-{sens_lr_ci[1]:.4f}]")
    
    return (auc_xgb, auc_xgb_ci, sens_xgb, sens_xgb_ci), (auc_lr, auc_lr_ci, sens_lr, sens_lr_ci)

if __name__ == "__main__":
    print("\n" + "="*50 + "\n EPC METHODOLOGY VALIDATION \n" + "="*50)
    
    target_file = input("Enter dataset filename [default: nhanes_master_clean.csv]: ").strip() or "nhanes_master_clean.csv"
    target_col = input("Enter target column name [default: diabetes]: ").strip() or "diabetes"
    
    try:
        depth_input = input("Enter max depth for structural workflow discovery (e.g., 2, 3, 4) [default: 3]: ").strip()
        search_depth = int(depth_input) if depth_input.isdigit() else 3
        
        df = pd.read_csv(target_file)
        if target_col not in df.columns: raise KeyError(f"Target '{target_col}' missing.")
        
        # ==========================================
        # ANTI-LEAKAGE & COLLINEARITY PATCH
        # ==========================================
        leakage_cols = [c for c in df.columns if c.startswith('DIQ') or c.startswith('DID')]
        if target_col in leakage_cols: leakage_cols.remove(target_col)
        disease_covariates = [c for c in df.columns if c.startswith('MCQ')]
        df = df.drop(columns=leakage_cols + disease_covariates, errors='ignore')
        # ==========================================
            
        unique_vals = df[target_col].dropna().unique()
        if len(unique_vals) == 2: df[target_col] = (df[target_col] == unique_vals[0]).astype(int)
        if 'patient_id' in df.columns: df = df.set_index('patient_id')
        if 'SEQN' in df.columns: df = df.set_index('SEQN')
            
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target_col])
        features = [col for col in df.columns if col != target_col]
        
        print("\n[STAGE 1] Running Multi-Pillar EPC Diagnostics (Max Depth: {search_depth})...")
        report_df, su_dict = run_epc_diagnostics(df_train, features, target_col, search_depth=search_depth)
        
        print("\n=== DIAGNOSTIC AUDIT RESULTS ===")
        # Added 'Odds_Ratio' to the list of columns to print
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(report_df[['Variable', 'Missing_SU', 'IV_Smoothed', 'LR_Plus', 'Odds_Ratio', 'Classification', 'Linked_Variables']].to_string())

        print("\n[STAGE 2] Generating Datasets...")
        X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = prepare_raw(df_train, df_test, target_col)
        X_tr_mice, y_tr_mice, X_te_mice, y_te_mice = prepare_mice(df_train, df_test, target_col)
        X_tr_epc, y_tr_epc, X_te_epc, y_te_epc = prepare_epc(df_train, df_test, target_col, report_df, su_dict)
        X_tr_epc_raw, y_tr_epc_raw, X_te_epc_raw, y_te_epc_raw = prepare_epc_raw(df_train, df_test, target_col, report_df)

        print("\n[STAGE 3] Evaluating Dual-Classifier Models (XGBoost vs Logistic Regression)...")
        xgb_raw, lr_raw = evaluate_dual_models(X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, "Baseline: Raw Data (Naive Median for LR)", has_nans=True)
        xgb_mice, lr_mice = evaluate_dual_models(X_tr_mice, y_tr_mice, X_te_mice, y_te_mice, "Baseline: MICE Imputed", has_nans=False)
        xgb_epc, lr_epc = evaluate_dual_models(X_tr_epc, y_tr_epc, X_te_epc, y_te_epc, "Proposed: EPC Imputed", has_nans=False)
        xgb_epc_raw, lr_epc_raw = evaluate_dual_models(X_tr_epc_raw, y_tr_epc_raw, X_te_epc_raw, y_te_epc_raw, "Proposed: EPC Flags + Raw NaNs (Median for LR)", has_nans=True)

        print("\n" + "="*85 + "\n FINAL COMPARISON (XGBoost - Sparsity Aware) \n" + "="*85)
        print(f"Raw Data Baseline     -> AUC: {xgb_raw[0]:.4f} [{xgb_raw[1][0]:.4f}-{xgb_raw[1][1]:.4f}] | Sens: {xgb_raw[2]:.4f} [{xgb_raw[3][0]:.4f}-{xgb_raw[3][1]:.4f}]")
        print(f"MICE Baseline         -> AUC: {xgb_mice[0]:.4f} [{xgb_mice[1][0]:.4f}-{xgb_mice[1][1]:.4f}] | Sens: {xgb_mice[2]:.4f} [{xgb_mice[3][0]:.4f}-{xgb_mice[3][1]:.4f}]")
        print(f"EPC Imputed           -> AUC: {xgb_epc[0]:.4f} [{xgb_epc[1][0]:.4f}-{xgb_epc[1][1]:.4f}] | Sens: {xgb_epc[2]:.4f} [{xgb_epc[3][0]:.4f}-{xgb_epc[3][1]:.4f}]")
        print(f"EPC Flags + Raw NaNs  -> AUC: {xgb_epc_raw[0]:.4f} [{xgb_epc_raw[1][0]:.4f}-{xgb_epc_raw[1][1]:.4f}] | Sens: {xgb_epc_raw[2]:.4f} [{xgb_epc_raw[3][0]:.4f}-{xgb_epc_raw[3][1]:.4f}]")
        
        print("\n" + "="*85 + "\n FINAL COMPARISON (Logistic Regression - Parametric) \n" + "="*85)
        print(f"Naive Median Baseline -> AUC: {lr_raw[0]:.4f} [{lr_raw[1][0]:.4f}-{lr_raw[1][1]:.4f}] | Sens: {lr_raw[2]:.4f} [{lr_raw[3][0]:.4f}-{lr_raw[3][1]:.4f}]")
        print(f"MICE Baseline         -> AUC: {lr_mice[0]:.4f} [{lr_mice[1][0]:.4f}-{lr_mice[1][1]:.4f}] | Sens: {lr_mice[2]:.4f} [{lr_mice[3][0]:.4f}-{lr_mice[3][1]:.4f}]")
        print(f"EPC Imputed           -> AUC: {lr_epc[0]:.4f} [{lr_epc[1][0]:.4f}-{lr_epc[1][1]:.4f}] | Sens: {lr_epc[2]:.4f} [{lr_epc[3][0]:.4f}-{lr_epc[3][1]:.4f}]")
        print(f"EPC Flags + Naive Med -> AUC: {lr_epc_raw[0]:.4f} [{lr_epc_raw[1][0]:.4f}-{lr_epc_raw[1][1]:.4f}] | Sens: {lr_epc_raw[2]:.4f} [{lr_epc_raw[3][0]:.4f}-{lr_epc_raw[3][1]:.4f}]")
        print("="*85)
        
    except Exception as e: print(f"Error: {e}")
