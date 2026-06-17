# ----------
#### Section: Modules & Libraries 
# ----------
import matplotlib # improting core plotting lib, engine  
matplotlib.use('Agg') # Agg for no graphical windows, good for server 
import matplotlib.pyplot as plt # plotting interface, stearing wheel  
import seaborn as sns # higher level plotting library 
import pandas as pd # main data wrangling tool 
import numpy as np # for anything needing fast/vectorized 

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
#   StratifiedKFold, for CV, keeping proportion in each fold, esp for imbalanced data 
#   cross_validate, to get metrics per fold 
#   cross_val_predict, to get prediction for each sample when held out, to make ROC & confusion matrix 

from sklearn.ensemble import RandomForestClassifier # classifier, in here as feature selector 
from sklearn.svm import SVC # classifier
from xgboost import XGBClassifier # classifier 

from sklearn.decomposition import PCA # to reduce dimensions, linearly 
from umap import UMAP # to reduce dimension, non-linearly 

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# eval metrics: 
#   classification_report, to print precision, recall, F1 
#   confusion_matrix, to get hit/miss grid per class 
#   roc_curve & auc, gets per-class ROC curves and the AUC 

from sklearn.feature_selection import VarianceThreshold # drops genes w near-0 variance 
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
#   StandardScaler, normalizes features to 0 mean and unit var before feeding into classifiers 
#   LabelEncoder, converts subtype strings e.g., LumA into integers 
#   label_binarize, converts those int into binary matrix for multi-class ROC calc 

from sklearn.base import BaseEstimator, TransformerMixin
#   BaseEstimator, to handle setting + copying 
#   TransformerMixin, to handle fit then transform step
# for later to use inside custom class/function 

from imblearn.over_sampling import SMOTE # SMOTE to generate new minority samples
#   through interpolating, to handle class imbalance 
from imblearn.pipeline import Pipeline # VERY IMPORTANT 
#   to apply SMOTE only to training folds, prevents leak

# mc nemar's test 
from scipy.stats import chi2 # gets chi^2 distro, to check model diffs 
from itertools import combinations # to gen unique combos of models, instead of doing manually 

# importing but silencing non-critical warnings 
import warnings
warnings.filterwarnings('ignore') 

# ----------

#### custom scikit-learn transformers 
class TopNSelector(BaseEstimator, TransformerMixin): # custom class, inheriting from 2 parent classes 
    def __init__(self, n_features=1000, random_state=42): # keeping 1000 genes, setting seed for reproducibility 
        self.n_features = n_features
        self.random_state = random_state
        # instantiates internal RF model to compare feature importance score 
        self.rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        self.top_indices_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.rf.fit(X, y) # trains RF on the current fold's training data 
        importances = self.rf.feature_importances_ # extracts gini imp score 
        actual_n = min(self.n_features, X.shape[1]) # prevents errors if data has less features than req
        self.top_indices_ = np.argsort(importances)[::-1][:actual_n] # sorts feature imp in descending order 
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns[self.top_indices_].tolist() # retains gene symbol if input is a df 
        return self

    def transform(self, X): # slices out only selected top N features from data 
        return X.iloc[:, self.top_indices_] if isinstance(X, pd.DataFrame) else X[:, self.top_indices_]

#### in-fodl var thresholding > aparently did not work 
class InFoldVarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self._vt = None

    def fit(self, X, y=None):
        self._vt = VarianceThreshold(threshold=self.threshold) # initing thresholding 
        self._vt.fit(X) # calcs feature var on the training fold 
        return self

    def transform(self, X):
        return self._vt.transform(X) # drops low var features from split 

#### data loading and processing 
def load_and_preprocess(mrna_file, clin_file):
    print("1. Loading and cleaning data...")
    mrna = pd.read_csv(mrna_file, sep='\t').set_index('Hugo_Symbol').drop(columns=['Entrez_Gene_Id'], errors='ignore').astype(np.float32).T
    # read gene exp files, raw = gene, col = samples 
    mrna.index = mrna.index.str[:12] # truncates tcga barcodes, to match patient id, takes first 12 chars  

    # reads clinical data file, bypassing cBioPortal comments 
    clinical = pd.read_csv(clin_file, sep='\t', comment='#')
    id_col = 'Patient Identifier' if 'Patient Identifier' in clinical.columns else 'PATIENT_ID'
    clinical.set_index(id_col, inplace=True)

    # finds target col with cancner subtype info 
    subtype_col = next((col for col in ['Subtype', 'SUBTYPE', 'BRCA_Subtype'] if col in clinical.columns), None)
    df = mrna.join(clinical[[subtype_col]], how='inner').dropna(subset=[subtype_col])
    # inner-join mRNA metrics and clinical labels based on mathcing patient id, dropping samples missing subtype label 

    X, y = df.drop(columns=[subtype_col]), df[subtype_col]
    X.columns = [str(col) for col in X.columns] # enforces str formatting on col labels 

    # global var filtering, drops all genes w var < 0.01
    vt = VarianceThreshold(threshold=0.01)
    X = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()], index=X.index)

    # standardizes text subtyping str (e.g., LumA) into machine readbale num (like 0, 1, or 2,..)
    le = LabelEncoder()
    return X, le.fit_transform(y), le.classes_

#### diagnostics and visualization 
def plot_diagnostics(y_true, y_probas, classes, name):
    # converts int class to binary flag matrix for multi-class roc calculation 
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    plt.figure(figsize=(8, 6))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probas[:, i]) # calcs false +ve and true +ve 
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {auc(fpr, tpr):.2f})') # plots perclass roc 
    plt.plot([0, 1], [0, 1], 'k--') # draws diagonal 50% random chance baseline 
    plt.title(f'ROC Curves: {name}')
    plt.legend(loc="lower right")
    plt.savefig(f'ROC_{name.replace(" ", "_")}.png') # exports graph 
    plt.close()

    # evals hard predictions by picking class hodling best probability score 
    y_pred = np.argmax(y_probas, axis=1)
    cm = confusion_matrix(y_true, y_pred) # counts classification hits and misses 
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    # overlays raw integer counts 
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f'CM_{name.replace(" ", "_")}.png')
    plt.close()

#### model statistical sig diff test, mc nemar's 
def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true) # boolean array showin if model A correct 
    correct_b = (pred_b == y_true) # if model B correct 
    b = np.sum(correct_a & ~correct_b) # contingency table where A right, B not 
    c = np.sum(~correct_a & correct_b) # if B right, A not 
    if (b + c) == 0:
        return 0.0, 1.0 # returns 0 diff and p=1, if identical model 
    stat = (abs(b - c) - 1) ** 2 / (b + c) # mc nemar's test statistic w edwards continuity correction 
    p_value = chi2.sf(stat, df=1) # p-value from 1 df chi-sq distribution 
    return stat, p_value

def run_significance_tests(all_preds, y_true, pipeline_names):
    print("\n" + "=" * 70)
    print("PAIRWISE MCNEMAR'S TEST (Bonferroni-corrected, alpha=0.05)")
    print("=" * 70)
    pairs = list(combinations(range(len(pipeline_names)), 2)) # gens unique pairs of models 
    alpha = 0.05
    corrected_alpha = alpha / len(pairs) # strict bonferroni correction to penalize multiple testing 
    print(f"  Number of pairs: {len(pairs)}  |  Corrected alpha: {corrected_alpha:.4f}\n")
    print(f"  {'Pipeline A':<30} {'Pipeline B':<30} {'Chi2':>8} {'p-value':>10} {'Sig?':>6}")
    print(f"  {'-'*30} {'-'*30} {'-'*8} {'-'*10} {'-'*6}")

    results = []
    for i, j in pairs:
        stat, p_val = mcnemar_test(y_true, all_preds[i], all_preds[j])
        significant = "YES" if p_val < corrected_alpha else "no" # checks against corrected alpha 
        print(f"  {pipeline_names[i]:<30} {pipeline_names[j]:<30} {stat:>8.3f} {p_val:>10.4f} {significant:>6}")
        results.append((pipeline_names[i], pipeline_names[j], stat, p_val, significant))

    # formats results into structure to export 
    results_df = pd.DataFrame(results, columns=['Pipeline_A', 'Pipeline_B', 'McNemar_Chi2', 'p_value', 'Significant'])
    results_df['Bonferroni_alpha'] = corrected_alpha
    results_df.to_csv('significance_tests.csv', index=False)
    print(f"\n  Full results saved to significance_tests.csv")

    # makes internal sequare synmetric matrix for p-value for plotting 
    n = len(pipeline_names)
    p_matrix = np.ones((n, n))
    for i, j in pairs:
        _, p_val = mcnemar_test(y_true, all_preds[i], all_preds[j])
        p_matrix[i, j] = p_val
        p_matrix[j, i] = p_val

    short_names = [name.split('.')[1].strip() if '.' in name else name for name in pipeline_names]
    plt.figure(figsize=(8, 6))
    # heatmap to clip between 0 - 0.05 to show sig vs non-sig diffs 
    sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=short_names, yticklabels=short_names,
                vmin=0, vmax=0.05, linewidths=0.5)
    plt.title(f"McNemar p-values (corrected α = {corrected_alpha:.4f})\nGreen = not significant | Red = significant difference")
    plt.tight_layout()
    plt.savefig('Significance_Heatmap.png')
    plt.close()
    print("  p-value heatmap saved to Significance_Heatmap.png")

#### sensitivity check, data leakage test 
def run_vt_leakage_sensitivity_check(X, y):
    print("\n SENSITIVITY CHECK: Global vs In-fold VarianceThreshold")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # instantiate 10-fold CV strategy  
    xgb_params = {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6,
                  'random_state': 42, 'eval_metric': 'mlogloss'}
    
    # for global variance cut 
    pipe_standard = Pipeline([
        ('selector', TopNSelector(n_features=1000)),
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(**xgb_params))
    ])

    # supposed to isolate low var filtering inside each fold  
    pipe_strict = Pipeline([
        ('vt', InFoldVarianceThreshold(threshold=0.01)),
        ('selector', TopNSelector(n_features=1000)),
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(**xgb_params))
    ])

    for label, pipe in [("Global VT (standard)", pipe_standard),
                        ("In-fold VT (strict)",  pipe_strict)]:
        results = cross_validate(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        fold_accs = results['test_score']
        # shows mean accuracy, var, sd, 95% ci 
        print(f"  {label}: mean={fold_accs.mean():.4f}  std={fold_accs.std():.4f}  "
              f"95% CI=[{fold_accs.mean()-1.96*fold_accs.std():.4f}, "
              f"{fold_accs.mean()+1.96*fold_accs.std():.4f}]")

#### for comparative study benchmarking  
def run_comparative_study(X, y, classes):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    xgb_params = {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6,
                  'random_state': 42, 'eval_metric': 'mlogloss'}
    svm_params = {'C': 20, 'gamma': 'scale', 'kernel': 'rbf',
                  'probability': True, 'random_state': 42}
    # dict container for teh 6 comparative pipelines 
    pipelines = {
        "1. PCA-SVM (Baseline)": Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=100, random_state=42)),
            ('svm', SVC(**svm_params))
        ]),
        "2. UMAP-SVM (Non-linear)": Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('umap', UMAP(n_components=15, random_state=42)),
            ('svm', SVC(**svm_params))
        ]),
        "3. RF-SVM (Info Recovery)": Pipeline([
            ('selector', TopNSelector(n_features=1000)),
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('svm', SVC(**svm_params))
        ]),
        "4. Hybrid RF-PCA-SVM": Pipeline([
            ('selector', TopNSelector(n_features=1000)),
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=100, random_state=42)),
            ('svm', SVC(**svm_params))
        ]),
        "5. RF-XGBoost (Best)": Pipeline([
            ('selector', TopNSelector(n_features=1000)),
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(**xgb_params))
        ]),
        "6. Full-genome XGBoost (Control)": Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(**xgb_params))
        ]),
    }

    all_preds = []
    pipeline_names = list(pipelines.keys())
    fold_summary_rows = []

    for name, pipe in pipelines.items():
        print(f"\nEvaluating: {name}...")

        # execs 10-fold CV for stable metric collection 
        cv_results = cross_validate(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        fold_accs = cv_results['test_score']
        mean_acc  = fold_accs.mean()
        std_acc   = fold_accs.std()
        ci_lo     = mean_acc - 1.96 * std_acc
        ci_hi     = mean_acc + 1.96 * std_acc
        print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}  "
              f"(approx. 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
        # to save data per fold to verify model performance consistency 
        fold_summary_rows.append({
            'Pipeline': name,
            'Mean_Accuracy': mean_acc,
            'Std_Accuracy':  std_acc,
            'CI_Lower_95':   ci_lo,
            'CI_Upper_95':   ci_hi,
            **{f'Fold_{i+1}': acc for i, acc in enumerate(fold_accs)}
        })

        # generates clean, out-of-fold predict probabilities to build diags w/o overestimaitng model performance 
        y_probas = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba', n_jobs=1)
        y_pred = np.argmax(y_probas, axis=1)

        all_preds.append(y_pred)
        
        # prints precision, recall, and f1
        print(classification_report(y, y_pred, target_names=classes))
        plot_diagnostics(y, y_probas, classes, name)
        plt.close('all')

    # compiles summary records and exports to csv 
    fold_df = pd.DataFrame(fold_summary_rows)
    fold_df.to_csv('fold_accuracy_summary.csv', index=False)
    print("\n  Per-fold accuracy summary saved to fold_accuracy_summary.csv")

    # pipeline accuracy bar chart horizontal
    short_names = [n.split('.')[1].strip() if '.' in n else n for n in pipeline_names]
    bar_colors = [
        '#808080',    # gray
        '#8B0000',    # dark red (worst)
        '#4682B4',    # steel blue
        '#4682B4',    # steel blue
        '#2E7D32',    # green
        '#00008B',    # dark blue
    ]
    plt.figure(figsize=(10, 5))
    plt.barh(short_names, fold_df['Mean_Accuracy'], xerr=fold_df['Std_Accuracy'],
             capsize=5, color=bar_colors, alpha=0.85)
    plt.xlabel('Accuracy (mean ± std)')
    plt.title('10-Fold CV Accuracy with Variance (all pipelines)')
    plt.tight_layout()
    plt.savefig('Pipeline_Accuracy_CI.png')
    plt.close()
    print("  Accuracy ± std bar chart saved to Pipeline_Accuracy_CI.png")

    # mc nemars' test on accumulated prediction arrays 
    run_significance_tests(all_preds, y, pipeline_names)

    # Feature importance stability check: capture top 20 genes per fold, then curate
    print("\nGenerating Feature Importance Stability Check (per-fold top 20)...")
    cv_fi = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    gene_counts = {}        # tracks total fold selection freqs per gene 
    gene_importances = {}   # records raw Gini importance values across splits 

    for fold_idx, (train_idx, _) in enumerate(cv_fi.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        # fits selector indepenndently on current training fold to avoid leakage 
        selector = TopNSelector(n_features=1000).fit(X_train, y_train)
        fold_imps = selector.rf.feature_importances_
        top20_idx = np.argsort(fold_imps)[::-1][:20] # extracts 20 best-score features 
        top20_genes = X_train.columns[top20_idx].tolist()
        top20_imps  = fold_imps[top20_idx]
        for gene, imp in zip(top20_genes, top20_imps):
            gene_counts[gene] = gene_counts.get(gene, 0) + 1
            gene_importances.setdefault(gene, []).append(imp)
        print(f"  Fold {fold_idx + 1:>2}/10 done  ({len(top20_genes)} genes captured)")

    # Build curated list: sort by fold frequency, break ties by mean importance
    stability_rows = [
        {
            'Gene':            gene,
            'Fold_Count':      count,
            'Mean_Importance': np.mean(gene_importances[gene]),
            'Std_Importance':  np.std(gene_importances[gene]),
        }
        for gene, count in gene_counts.items()
    ] # sorts descending, most stable features on top 
    stability_df = (
        pd.DataFrame(stability_rows)
        .sort_values(['Fold_Count', 'Mean_Importance'], ascending=[False, False])
        .reset_index(drop=True)
    )
    stability_df.to_csv('feature_importance_stability.csv', index=False)
    print(f"  Full stability table ({len(stability_df)} genes) saved to "
          f"feature_importance_stability.csv")

    # plot the curated top-20 by fold frequency
    top20_stable = stability_df.head(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Fold_Count', y='Gene', data=top20_stable, palette='viridis')
    plt.axvline(x=5, color='red', linestyle='--', linewidth=1,
                label='50 % threshold (5 folds)')
    plt.legend(loc='lower right')
    plt.title('Top 20 Genes by Stability Across 10 CV Folds\n'
              '(frequency of appearing in per-fold top 20)')
    plt.xlabel('Number of Folds Gene Appeared in Top 20 (out of 10)')
    plt.xlim(0, 10)
    plt.tight_layout()
    plt.savefig('Feature_Importance_Stability.png')
    plt.close()
    print("  Stability bar chart saved to Feature_Importance_Stability.png")

#### umap hyperparameter sweep combinding n_components x n_neighbors 
def run_umap_sweep(X, y):
    print("\n UMAP HYPERPARAMETER SWEEP (n_components × n_neighbors)")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    svm_params = {'C': 20, 'gamma': 'scale', 'kernel': 'rbf',
                  'probability': True, 'random_state': 42}
    
    # parameter grid space definitions 
    n_components_grid = [5, 10, 15, 20, 30]
    n_neighbors_grid  = [5, 15, 30]

    sweep_rows = []
    for n_comp in n_components_grid:
        for n_neigh in n_neighbors_grid:
            # reconstructs clean pipeline exec units for each hyper-param combo 
            pipe = Pipeline([
                ('smote',  SMOTE(random_state=42)),
                ('scaler', StandardScaler()),
                ('umap',   UMAP(n_components=n_comp, n_neighbors=n_neigh, random_state=42)),
                ('svm',    SVC(**svm_params))
            ])
            results   = cross_validate(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=1)
            fold_accs = results['test_score']
            mean_acc  = fold_accs.mean()
            std_acc   = fold_accs.std()
            print(f"  n_components={n_comp:>2}, n_neighbors={n_neigh:>2}: "
                  f"{mean_acc:.4f} ± {std_acc:.4f}")
            sweep_rows.append({'n_components': n_comp, 'n_neighbors': n_neigh,
                                'mean_accuracy': mean_acc, 'std_accuracy': std_acc})
    
    # svaes grid eval matrix data 
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv('umap_sweep_results.csv', index=False)
    print("\n  Sweep results saved to umap_sweep_results.csv")

    # identifies and prints optimal hyper param combo found in sweep 
    best_row = sweep_df.loc[sweep_df['mean_accuracy'].idxmax()]
    print(f"\n  Best UMAP config: n_components={int(best_row['n_components'])}, "
          f"n_neighbors={int(best_row['n_neighbors'])}  "
          f"→ accuracy={best_row['mean_accuracy']:.4f} ± {best_row['std_accuracy']:.4f}")

#### program execution block 
if __name__ == "__main__":
    X, y, classes = load_and_preprocess('data_mrna_seq_v2_rsem.txt', 'data_clinical_patient.txt')
    # to run file loading, sample matching, and target encoding steps 
    run_comparative_study(X, y, classes)
    # runs main pipeline comparison, roc auc, feature stability
    run_vt_leakage_sensitivity_check(X, y)
    # runs sensitivity analysis to check data leakage 
    run_umap_sweep(X, y)
    # doing the UMAP hyperparameter grid search 

