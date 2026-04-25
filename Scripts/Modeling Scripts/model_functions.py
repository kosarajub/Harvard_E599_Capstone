# Import Packages
import os
import re
import sys
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_val_score, GroupKFold

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV

from sklearn.feature_selection import VarianceThreshold

def remove_low_variance_cols(X_data):
    categorical_cols = X_data.select_dtypes(include=['object', 'category']).columns.tolist()

    continuous_cols = [c for c in X_data.columns if c not in categorical_cols]
    
    # Apply variance threshold only to continuous columns
    X_continuous = X_data[continuous_cols]
    selector     = VarianceThreshold(threshold=0.01)
    selector.fit(X_continuous)
    
    low_variance_cols = X_continuous.columns[~selector.get_support()].tolist()
    print(f"Low variance columns removed: {len(low_variance_cols)}")
    
    kept_continuous = X_continuous.columns[selector.get_support()].tolist()
    
    # Reassemble — keep all encoded columns, keep only high-variance continuous columns
    X_data = X_data[kept_continuous + categorical_cols]
    print(f"Features after variance filtering: {len(X_data.columns)}")
    return X_data

def remove_uncorrelated_cols(X_data, y_data):
    categorical_cols = X_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Separate continuous and encoded columns
    continuous_cols = [c for c in X_data.columns if c not in categorical_cols]
    
    # Apply correlation filter only to continuous columns
    X_continuous = X_data[continuous_cols]
    correlations = X_continuous.corrwith(y_data).abs().sort_values(ascending=False)
    
    # Compute Pearson correlation of every feature against target
    CORR_THRESHOLD   = 0.1
    strong_corr_cols = correlations[correlations >= CORR_THRESHOLD].index.tolist()
    weak_corr_cols   = correlations[correlations <  CORR_THRESHOLD].index.tolist()
    
    print(f"Continuous features kept   : {len(strong_corr_cols)}")
    print(f"Continuous features removed: {len(weak_corr_cols)}")
    print(f"\nTop 20 correlated features:")
    print(correlations.head(20))
    
    # Reassemble — keep all encoded columns, keep only high-correlation continuous columns
    X_data = X_data[strong_corr_cols + categorical_cols]
    print(f"\nTotal features after correlation filtering: {len(X_data.columns)}")

    return X_data

def tabulate_results(results):
    rows = []
    for label, metrics in results.items():
        rows.append({
            'Experiment'      : label,
            'Test R²'         : round(metrics['test_r2'], 4),
            'Test RMSE'       : round(metrics['test_rmse'], 2),
            'Train R²'        : round(metrics['train_r2'], 4),
            'Tr4ain RMSE'     : round(metrics['train_rmse'], 2),
            'CV R² Mean'      : round(metrics['cv_r2_mean'], 4),
            'CV R² Std'       : round(metrics['cv_r2_std'], 4),
            'Group CV R² Mean': round(metrics['group_cv_r2_mean'], 4),
            'Group CV R² Std' : round(metrics['group_cv_r2_std'], 4),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Group CV R² Mean', ascending=False).reset_index(drop=True)

    styled = (
        df.style
        .highlight_max(
            subset=['Test R²', 'Group CV R² Mean'],
            color='lightgreen'
        )
        .highlight_min(
            subset=['Test RMSE', 'Group CV R² Std'],
            color='lightgreen'
        )
    )
    display(styled)

def residual_analysis(y_pred, residuals):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].scatter(y_pred, residuals, alpha=0.4)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted AGB')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    
    axes[1].hist(residuals, bins=30)
    axes[1].set_xlabel('Residual')
    axes[1].set_title('Residual Distribution')
    
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('QQ Plot')
    
    plt.tight_layout()
    plt.show()

def cross_validate(model_func, X_data, y_data, cv=10, scoring='r2'):
    cv_scores = cross_val_score(model_func,
                                X_data, y_data,
                                cv=cv,
                                scoring=scoring)
    print("\n Cross-validation ---")
    print(f"CV R² mean: {cv_scores.mean():.4f}")
    print(f"CV R² std : {cv_scores.std():.4f}")
    print(f"CV scores : {cv_scores.round(3)}")

    return cv_scores.mean(), cv_scores.std(), cv_scores

def cross_validate_by_groups(model_func, X_data, y_data, groups, n_splits=10, scoring='r2'):
    # Cap n_splits to the number of unique groups
    n_unique_groups = len(groups.unique())

    # Example:
    # n_unique_groups = 60
    #  GroupKFold(n_splits=60) => 1 group held out per fold  (leave-one-group-out)
    #  GroupKFold(n_splits=10) => 6 groups held out per fold
    #  GroupKFold(n_splits=6)  => 10 groups held out per fold
    #  GroupKFold(n_splits=2)  => 30 groups held out per fold
    n_splits = min(n_splits, n_unique_groups) # num folds <= group count

    # GroupKFold splits data into K folds ensuring all rows belonging
    # to the same group are entirely in one fold.
    # Example:
    #   There have 5 plots and n_splits=5.
    #   GroupKFold assigns each plot to exactly one fold:
    #    Fold 1 — test: Plot A    train: Plots B, C, D, E
    #    Fold 2 — test: Plot B    train: Plots A, C, D, E
    #    Fold 3 — test: Plot C    train: Plots A, B, D, E
    #    Fold 4 — test: Plot D    train: Plots A, B, C, E
    #    Fold 5 — test: Plot E    train: Plots A, B, C, D
    gkf       = GroupKFold(n_splits=n_splits)
    cv_scores = cross_val_score(model_func,
                                X_data, y_data,
                                cv=gkf.split(X_data, y_data, groups),
                                scoring=scoring)

    print("\nGrouped Cross-validation ---")
    print(f"Grouped CV R² mean: {cv_scores.mean():.4f}")
    print(f"Grouped CV R² std : {cv_scores.std():.4f}")
    print(f"Grouped CV scores : {cv_scores.round(3)}")

    return cv_scores.mean(), cv_scores.std(), cv_scores

def split_data_by_groups(X_var, y_var, groups):
    X_local = X_var.copy(deep=True)
    y_local = y_var.copy(deep=True)

    # Let us say there are 5 unique plots and 15 trees in total, aka, 3 trees per plot.
    #  Plot A — rows 0, 1, 2
    #  Plot B — rows 3, 4, 5
    #  Plot C — rows 6, 7, 8
    #  Plot D — rows 9, 10, 11
    #  Plot E — rows 12, 13, 14
    #
    # GroupShuffleSplit
    # -----------------
    # This is a cross-validation iterator that generates train/test indices
    # to split data into subsets, ensuring specific groups do not overlap
    # between sets.
    # 
    # n_splits=1 => One fixed split. 80-20 split in this case.
    # OUTPUT of GroupShuffleSplit(n_splits=1..)
    #   train_idx = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14] # plots A,B,D,E
    #   test_idx  = [6, 7, 8]                                 # plot C only
    #
    # next(gss.split())
    # -----------------
    # gss.split() is a generator that would produce n_splits pairs.
    # Since n_splits=1 there is only one pair to produce.
    # The next() call retrieves that one pair and returns it.
    # OUTPUT of next(gss.split())
    #    train_idx = [0,1,2,3,4,5,9,10,11,12,13,14]
    #    test_idx  = [6,7,8]
    #
    # Interpolation (The "Cheat" Split)
    # ---------------------------------
    #  If the training set has 8 trees from Plot_A and the test set has the
    #  remaining 2 trees from Plot_A, the  model is interpolating. It already
    #  knows the specific "vibe" of Plot_A.
    #
    #  Result:
    #   - High R2 that is false.
    #   - The model is just identifying trees it already "knows" by proxy.
    #
    # Extrapolation (The "True" Split)
    # ---------------------------------
    #  If you put every single tree from Plot_A into the test set and train the model
    #  only on trees from Plot_B, C, and D, the model is extrapolating.
    #
    #  The "New Site": In this context, Plot_A is the "new site."
    #  It is a geographic location the model has never seen.
    #
    #  Result:
    #    - A lower, but realistic R2.
    #    - This tells you how well your model will perform if you take it to a
    #      completely different part of Panama or Brazil.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_idx, test_idx = next(gss.split(X_local, y_local, groups=groups))
    
    X_train = X_local.iloc[train_idx]
    X_test  = X_local.iloc[test_idx]
    y_train = y_local.iloc[train_idx]
    y_test  = y_local.iloc[test_idx]

    # Verify no plot appears in both train and test
    #train_plots = set(X_var.iloc[train_idx]['group']) #plot_id
    #test_plots  = set(X_var.iloc[test_idx]['group'])

    train_plots = set(groups[train_idx]) #plot_id
    test_plots  = set(groups[test_idx])
    
    overlap     = train_plots & test_plots
    #print(f"Train plots : {len(train_plots)}")
    #print(f"Test plots  : {len(test_plots)}")
    #print(f"Overlapping plots: {len(overlap)}")  # must be 0
    assert not len(overlap)
    
    return X_train, X_test, y_train, y_test

def split_data_ver2(X_var, y_var):
    # The train_test_split() splits randomly by rows — it
    # does not know about any groups (e.g., plot ids, etc).
    # 
    # When the data is split with this method, trees from the same plot
    # can end up in both train and test. The model trains on 58 trees
    # from plot X and is tested on the remaining 7 trees from the same plot.
    # This is pseudo-replication.
    #
    # pseudo-replication
    # ------------------
    # Pseudo-replication occurs when individual data points are treated as
    # independent observations in a statistical model, even though they are
    # actually clustered or correlated due to a shared environment.

    # When is pseudo-replication NOT useful?
    #  - Pseudo-replication is a problem specifically when your data has a
    #    hierarchical structure, e.g., trees nested within plots,
    #    students nested within schools, etc.
    #  - In those cases observations within the same group are correlated and
    #   random splitting leaks information.
    # E.g.
    #  - When trees from the same plot appear in both train and test,
    #    the model is evaluated on data it has effectively already seen,
    #    i.e., same EMIT pixel, same site conditions, same species mix.
    #  - The R² will be inflated.
    #  - You report a number that does not reflect real world performance on new sites.

    # When is pseudo-replication useful?
    #  - Pseudo-replication is acceptable when your dataset has no meaningful
    #    grouping structure, aka, the observations are independent of each other.

    X_train, X_test, y_train, y_test = train_test_split(X_var,
                                                        y_var,
                                                        test_size=0.2,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def show_importances(results):
    importances = results["importances"].sort_values(ascending=False)
    
    # Feature importances
    N = 4
    print(f"\nTop {N} feature importances:")
    for feat, imp in importances.head(N).items():
        bar = '█' * int(imp * 50)
        print(f"  {feat:45s} {imp:.4f}  {bar}")

def handle_null_data(X_data):
    null_rows = X_data[X_data.isnull().any(axis=1)]
    total_nulls = X_data.isnull().sum().sum()
    
    print(f"Total NULL count           : {total_nulls}")
    print(f"Rows with at least one NULL: {len(null_rows)}")
    print(f"Total rows                 : {len(X_data)}")
    print(f"Percentage                 : {len(null_rows)/len(X_data)*100:.1f}%")
    
    # NULL count per column for only the affected rows
    null_col_counts = null_rows.isnull().sum().sort_values(ascending=False)
    
    print("\nNULL count per column in affected rows:")
    print(null_col_counts[null_col_counts > 0])

    vapor_bands = null_rows.isnull().sum()
    vapor_bands = vapor_bands[vapor_bands > 0].index.tolist()
    
    print(f"Dropping {len(vapor_bands)} columns:")
    print(vapor_bands)
    
    X_data = X_data.drop(columns=vapor_bands)
    print(f"\nNULL count after dropping: {X_data.isnull().sum().sum()}")

    return X_data

def linear_reg_groups(X_var, y_var, groups, label):
    #groups = X_var['group']
    X_train, X_test, y_train, y_test = split_data_by_groups(X_var, y_var, groups)
    return linear_reg(X_train, X_test, y_train, y_test, groups, label)

def linear_reg_ver2(X_train, X_test, y_train, y_test, groups, label):
    return linear_reg(X_train, X_test, y_train, y_test, groups, label)

def linear_reg(X_train, X_test, y_train, y_test, groups, label):
    X_var = pd.concat([X_train, X_test], axis=0)
    y_var = pd.concat([y_train, y_test], axis=0)
    
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred       = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train)

    test_r2     = r2_score(y_test, y_pred)
    test_rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

    train_r2    = r2_score(y_train, y_train_pred)
    train_rmse  = np.sqrt(mean_squared_error(y_train, y_train_pred))

    residuals   = y_test - y_pred

    print(f"\n--- {label} ---")
    print(f"Test R²     : {test_r2:.4f}")
    print(f"Test RMSE   : {test_rmse:.2f} kg")
    print(f"Train R²    : {train_r2:.4f}")
    print(f"Train RMSE  : {train_rmse:.2f} kg")
    print(f"Num Features: {X_var.shape[1]}")

    cv_r2_mean, cv_r2_std, cv_scores = cross_validate(LinearRegression(),
                                                      X_var,
                                                      y_var,
                                                      cv=10,
                                                      scoring='r2')
    
    group_cv_r2_mean = group_cv_r2_std = group_cv_scores = None
    if groups is not None:
        group_cv_r2_mean, group_cv_r2_std, group_cv_scores = \
            cross_validate_by_groups(LinearRegression(),
                                     X_var,
                                     y_var,
                                     groups,
                                     n_splits=10,
                                     scoring='r2')

    datum = {"test_r2": test_r2,
             "test_rmse": test_rmse,
             "train_r2": train_r2,
             "train_rmse": train_rmse,
             "y_pred": y_pred,
             "residuals": residuals,
             "cv_r2_mean": cv_r2_mean,
             "cv_r2_std": cv_r2_std,
             "cv_scores": cv_scores,
             "group_cv_r2_mean": group_cv_r2_mean,
             "group_cv_r2_std": group_cv_r2_std,
             "group_cv_scores": group_cv_scores,
             "model": model}

    return datum

def randomForest_groups(X_var, y_var, groups, label):
    X_train, X_test, y_train, y_test = split_data_by_groups(X_var, y_var, groups)

    return random_forest(X_train, X_test, y_train, y_test, groups, label)

def randomForest_ver2(X_train, X_test, y_train, y_test, groups, label):
    return random_forest(X_train, X_test, y_train, y_test, groups, label)

def random_forest(X_train, X_test, y_train, y_test, groups, label):
    X_var = pd.concat([X_train, X_test], axis=0)
    y_var = pd.concat([y_train, y_test], axis=0)
    
    # NOTE: Random forest does not need scaling.
    rf = RandomForestRegressor(
        n_estimators    = 500,
        max_features    = 'sqrt',
        min_samples_leaf= 2,
        random_state    = 42,
        n_jobs          = -1
    )
    
    rf.fit(X_train, y_train)
    
    # Test set performance
    y_pred       = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)
    
    test_r2    = r2_score(y_test, y_pred)
    test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred))

    train_r2   = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    residuals  = y_test - y_pred
    
    print(f"EXPERIMENT  : {label}")
    print(f"Test R²     : {test_r2:.4f}")
    print(f"Test RMSE   : {test_rmse:.2f} kg")
    print(f"Train R²    : {train_r2:.4f}")
    print(f"Train RMSE  : {train_rmse:.2f} kg")
    print(f"Num Features: {X_var.shape[1]}")

    cv_r2_mean, cv_r2_std, cv_scores = cross_validate(rf,
                                                      X_var,
                                                      y_var,
                                                      cv=10,
                                                      scoring='r2')
    
    group_cv_r2_mean = group_cv_r2_std = group_cv_scores = None
    if groups is not None:
        group_cv_r2_mean, group_cv_r2_std, group_cv_scores = \
            cross_validate_by_groups(LinearRegression(),
                                     X_var,
                                     y_var,
                                     groups,
                                     n_splits=10,
                                     scoring='r2')

    importances = pd.Series(rf.feature_importances_, index=X_var.columns)

    datum = {"test_r2": test_r2,
             "test_rmse": test_rmse,
             "train_r2": train_r2,
             "train_rmse": train_rmse,
             "y_pred": y_pred,             
             "residuals": residuals,
             "cv_r2_mean": cv_r2_mean,
             "cv_r2_std": cv_r2_std,
             "cv_scores": cv_scores,
             "group_cv_r2_mean": group_cv_r2_mean,
             "group_cv_r2_std": group_cv_r2_std,
             "group_cv_scores": group_cv_scores,
             "importances": importances,
             "model": rf}

    return datum

def select_best_model(experiments):
    max_val = max(v["cv_r2_mean"] for v in experiments.values())
    
    rf_best_labels = [
        label
        for label, vals in experiments.items()
        if vals["cv_r2_mean"] == max_val
    ]
    
    print(rf_best_labels)
    return rf_best_labels[0]