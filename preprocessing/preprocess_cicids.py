import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from scipy.stats import levene, kruskal

# Set output directory
out_dir = r'c:\Users\nates\Desktop\Major_Project\data\cicids'
os.makedirs(out_dir, exist_ok=True)

# 1. Load the dataset
df = pd.read_csv(r'c:\Users\nates\Desktop\Major_Project\data\cicids\cicids_dataset.csv')

# 2. Handle missing/infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# 3. Feature identification
cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
label_col = ' Label'  # or whatever your target column is
if label_col in num_features:
    num_features.remove(label_col)

# 4. Correlation analysis (remove highly correlated features)
corr_matrix = df[num_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
df.drop(columns=to_drop, inplace=True)
num_features = [f for f in num_features if f not in to_drop]

# 5. Variance threshold (optional, after correlation)
selector = VarianceThreshold(threshold=0.0001)
selector.fit(df[num_features])
low_var_cols = [col for col, var in zip(num_features, selector.variances_) if var < 0.0001]
df.drop(columns=low_var_cols, inplace=True)
num_features = [f for f in num_features if f not in low_var_cols]



# 6. Statistical tests (Levene & Kruskal-Wallis)
statistically_significant = []
target = df[label_col]
for col in num_features:
    groups = [df[col][target == t] for t in target.unique()]
    try:
        p_levene = levene(*groups).pvalue
        p_kruskal = kruskal(*groups).pvalue
        if p_levene < 0.05 and p_kruskal < 0.05:
            statistically_significant.append(col)
    except Exception:
        continue

# Remove label_col from cat_features if present (add this block)
if label_col in cat_features:
    cat_features.remove(label_col)

df = df[statistically_significant + cat_features + [label_col]]
num_features = statistically_significant



# 7. Visualization (set to False to skip)
if False:
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[num_features].corr(), annot=False)
    plt.title('Feature Correlation Heatmap')
    plt.show()

    for col in num_features:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution: {col}')
        plt.show()
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot: {col}')
        plt.show()

# 8. Categorical feature encoding
le_cat = {}
for cat in cat_features:
    if cat == label_col: continue
    le = LabelEncoder()
    df[cat] = le.fit_transform(df[cat])
    le_cat[cat] = dict(zip(le.classes_, le.transform(le.classes_)))

# 9. Save selected features
selected_features = num_features + [c for c in cat_features if c != label_col]
with open(os.path.join(out_dir, 'selected_features.json'), 'w') as f:
    json.dump(selected_features, f, indent=2)
with open(os.path.join(out_dir, 'categorical_encoders.pkl'), 'wb') as f:
    pickle.dump(le_cat, f)

# 10. Scaling/Normalization
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])
with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# ...existing code...

# 11. Target encoding & mapping
label_encoder = LabelEncoder()
if label_col in cat_features:
    cat_features.remove(label_col)
df[label_col] = label_encoder.fit_transform(df[label_col])
label_mapping = {str(k): int(v) for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
with open(os.path.join(out_dir, 'label_mapping.json'), 'w') as f:
    json.dump(label_mapping, f, indent=2)

# ...rest of your code...
# ...existing code...

# 12. Train/test split
X = df[selected_features]
y = df[label_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 13. Save the preprocessed data (SMOTE removed)
train_res = pd.DataFrame(X_train, columns=selected_features)
train_res[label_col] = y_train
test = pd.DataFrame(X_test, columns=selected_features)
test[label_col] = y_test

train_res.to_csv(os.path.join(out_dir, 'train_preprocessed.csv'), index=False)
test.to_csv(os.path.join(out_dir, 'test_preprocessed.csv'), index=False)

print(f"Preprocessing complete. Files saved to {out_dir}")
# ...rest of your code...