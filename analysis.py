import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np

# Data Loading
path = './pl_data'
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df_team = pd.read_csv(filename)
    df_team.columns = df_team.columns.str.strip()
    li.append(df_team)
df = pd.concat(li, axis=0, ignore_index=True)

df['ShotsOutsideBox'] = df['Shots'] - df['PenaltyAreaShots']

features = ['PenaltyAreaShots', 'ShortKeyPasses', 'ShortAccPasses',
            'UnsuccessfulPasses', 'SquadValue', 'Tackles', 'Interceptions',
            'ShotsOutsideBox']
df = df.dropna(subset=features + ['Points'])

df['SeasonStart'] = df['Season'].str[:4].astype(int)
df = df.sort_values('SeasonStart', ascending=True).reset_index(drop=True)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df[features + ['Points']].corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, fmt=".2f")
plt.title("Data Sense-Check: Feature Correlation Grid")
plt.show()

print("CORRELATION MATRIX (all features + points)")
print(corr.round(3))
print("\nRanked absolute correlation with Points:")
corr_with_points = corr['Points'].drop('Points').abs().sort_values(ascending=False)
for feature, r_val in corr_with_points.items():
    sign = '+' if corr.loc[feature, 'Points'] > 0 else '-'
    print(f"   {feature}: {r_val:.3f} (sign = {sign})")

# Unsupervided Clustering
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(df[features])

# Dynamic PCA 
pca_full = PCA(n_components=0.9, random_state=42)
X_pca_full = pca_full.fit_transform(X_scaled_full)
df['PCA1'], df['PCA2'] = X_pca_full[:, 0], X_pca_full[:, 1]

print(f"\nPCA explained variance ratio: {pca_full.explained_variance_ratio_.sum():.3f} total with {pca_full.n_components_} components")

# Determine optimal K from only 3 or 4 clusters
inertias = []
sil_scores = []
K_range = range(3, 5)   
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_pca_full)
    inertias.append(kmeans_temp.inertia_)
    sil_scores.append(silhouette_score(X_pca_full, labels_temp))
best_k = K_range[np.argmax(sil_scores)]
print(f"Optimal k for clustering (3 or 4): {best_k} (silhouette score: {max(sil_scores):.3f})")

# Final clustering
kmeans_full = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['TacticalCluster'] = kmeans_full.fit_predict(X_pca_full)

print(" CLUSTER & PCA SUMMARY (full data – visualisation only)")
cluster_counts = df['TacticalCluster'].value_counts().sort_index()
print("\nCluster sizes (number of team‑seasons):")
for cl, cnt in cluster_counts.items():
    print(f"  Cluster {cl}: {cnt} ({cnt/len(df)*100:.1f}%)")
print("\nMean Points per cluster:")
for cl in sorted(df['TacticalCluster'].unique()):
    mean_pts = df.loc[df['TacticalCluster']==cl, 'Points'].mean()
    print(f"  Cluster {cl}: {mean_pts:.1f}")
print("\nCluster centroids in PCA space:")
for i, centroid in enumerate(kmeans_full.cluster_centers_):
    print(f"  Cluster {i}: PC1={centroid[0]:.2f}, PC2={centroid[1]:.2f}")
print("\nMean feature values per cluster (standardised scale):")
cluster_means = pd.DataFrame(X_scaled_full, columns=features).copy()
cluster_means['Cluster'] = df['TacticalCluster']
print(cluster_means.groupby('Cluster').mean().round(3))

# PCA Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='TacticalCluster',
                style='Team', palette='viridis', s=120)
plt.title(f"Tactical Archetypes (k={best_k})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='TacticalCluster', y='Points', data=df,
               hue='TacticalCluster', palette='muted', legend=False)
plt.title("Success by Style: Points Distribution per Tactical Cluster")
plt.show()

# Classificication
split_idx = int(0.75 * len(df))
X_train_clf = df.iloc[:split_idx][features]
X_test_clf  = df.iloc[split_idx:][features]

print(f"\nChronological split: training on seasons {df.iloc[0]['Season']} to {df.iloc[split_idx-1]['Season']}, "
      f"testing on {df.iloc[split_idx]['Season']} to {df.iloc[-1]['Season']}")

# Scale, PCA, KMeans 
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train_clf)

pca_train = PCA(n_components=0.9, random_state=42)
X_train_pca = pca_train.fit_transform(X_train_scaled)

# Find best k on training 
inertias_train = []
sil_scores_train = []
for k in range(3, 5):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_train_pca)
    inertias_train.append(kmeans_temp.inertia_)
    sil_scores_train.append(silhouette_score(X_train_pca, labels_temp))
best_k_train = range(3,5)[np.argmax(sil_scores_train)]
print(f"Optimal k for training clustering (3 or 4): {best_k_train}")

kmeans_train = KMeans(n_clusters=best_k_train, random_state=42, n_init=10)
train_clusters = kmeans_train.fit_predict(X_train_pca)

# Transform test data
X_test_scaled = scaler_train.transform(X_test_clf)
X_test_pca = pca_train.transform(X_test_scaled)
test_clusters = kmeans_train.predict(X_test_pca)

# L1 logistic regression 
logreg_l1 = LogisticRegression(solver='saga', max_iter=5000, random_state=42, l1_ratio=1)
                            
logreg_l1.fit(X_train_scaled, train_clusters)

y_pred_clf = logreg_l1.predict(X_test_scaled)
acc = accuracy_score(test_clusters, y_pred_clf)

print("\nL1 Logistic Regression ")
print(f"Test Accuracy (future seasons): {acc:.3f}")
print("\nClassification Report:")
print(classification_report(test_clusters, y_pred_clf, zero_division=0))

cm = confusion_matrix(test_clusters, y_pred_clf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(set(train_clusters)),
            yticklabels=sorted(set(train_clusters)))
plt.title("Confusion Matrix: L1 Logistic Regression (Chronological Split)")
plt.xlabel("Predicted Cluster")
plt.ylabel("True Cluster")
plt.tight_layout()
plt.show()

# Coefficient heatmap 
coef_df = pd.DataFrame(
    logreg_l1.coef_,
    columns=features,
    index=[f'Cluster {c}' for c in logreg_l1.classes_]
)
plt.figure(figsize=(8, 4))
sns.heatmap(coef_df, annot=True, cmap='coolwarm', center=0)
plt.title("L1 Logistic Regression Coefficients (Chronological Split)")
plt.tight_layout()
plt.show()

# Regression, 3 Ridge
X_train_reg = df.iloc[:split_idx][features]
X_test_reg  = df.iloc[split_idx:][features]
y_train_reg = df.iloc[:split_idx]['Points']
y_test_reg  = df.iloc[split_idx:]['Points']

tactical_features = [f for f in features if f != 'SquadValue']
all_features = features

def tune_ridge(X_train, y_train):
    param_grid = {'alpha': [0.01, 0.1, 1, 5, 10, 50, 100]}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    return grid.best_params_['alpha']

# SquadValue only
scaler_sv = StandardScaler()
X_train_sv = scaler_sv.fit_transform(X_train_reg[['SquadValue']])
X_test_sv = scaler_sv.transform(X_test_reg[['SquadValue']])
best_alpha_sv = tune_ridge(X_train_sv, y_train_reg)
ridge_sv = Ridge(alpha=best_alpha_sv).fit(X_train_sv, y_train_reg)
y_pred_sv = ridge_sv.predict(X_test_sv)

# Tactical only
scaler_tac = StandardScaler()
X_train_tac = scaler_tac.fit_transform(X_train_reg[tactical_features])
X_test_tac = scaler_tac.transform(X_test_reg[tactical_features])
best_alpha_tac = tune_ridge(X_train_tac, y_train_reg)
ridge_tac = Ridge(alpha=best_alpha_tac).fit(X_train_tac, y_train_reg)
y_pred_tac = ridge_tac.predict(X_test_tac)

# Tactical + SquadValue
scaler_all = StandardScaler()
X_train_all = scaler_all.fit_transform(X_train_reg[all_features])
X_test_all = scaler_all.transform(X_test_reg[all_features])
best_alpha_all = tune_ridge(X_train_all, y_train_reg)
ridge_all = Ridge(alpha=best_alpha_all).fit(X_train_all, y_train_reg)
y_pred_all = ridge_all.predict(X_test_all)

print("\nBaseline Point Prediction")
print(f"SquadValue only (Ridge, α={best_alpha_sv:.2f})       -> RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_sv)):.2f}, R²: {r2_score(y_test_reg, y_pred_sv):.3f}")
print(f"Tactical features only (Ridge, α={best_alpha_tac:.2f}) -> RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_tac)):.2f}, R²: {r2_score(y_test_reg, y_pred_tac):.3f}")
print(f"Tactical + SquadValue (Ridge, α={best_alpha_all:.2f}) -> RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_all)):.2f}, R²: {r2_score(y_test_reg, y_pred_all):.3f}")

# Coefficient interpretation
print(" WHICH FEATURES DRIVE POINTS? (Ridge regression coefficients)")
coef_ridge = pd.Series(ridge_all.coef_, index=all_features)
coef_ridge_sorted = coef_ridge.abs().sort_values(ascending=False)
print("Top 3 most influential features (absolute coefficient):")
for i, (feature, coef_abs) in enumerate(coef_ridge_sorted.head(3).items(), 1):
    sign = "+" if coef_ridge[feature] > 0 else '-'
    print(f"   {i}. {feature}: coefficient = {coef_ridge[feature]:.3f} (→ {sign} impact)")
print("\nAll features:")
for feature, coef_val in coef_ridge.sort_values(key=abs, ascending=False).items():
    print(f"   {feature}: {coef_val:+.3f}")

# Scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(y_test_reg, y_pred_sv, alpha=0.6)
axes[0].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
axes[0].set_xlabel("Actual Points")
axes[0].set_ylabel("Predicted Points")
axes[0].set_title(f"SquadValue only (R²={r2_score(y_test_reg, y_pred_sv):.2f})")

axes[1].scatter(y_test_reg, y_pred_tac, alpha=0.6)
axes[1].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
axes[1].set_xlabel("Actual Points")
axes[1].set_ylabel("Predicted Points")
axes[1].set_title(f"Tactical only (R²={r2_score(y_test_reg, y_pred_tac):.2f})")

axes[2].scatter(y_test_reg, y_pred_all, alpha=0.6)
axes[2].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
axes[2].set_xlabel("Actual Points")
axes[2].set_ylabel("Predicted Points")
axes[2].set_title(f"Tactical + SquadValue (R²={r2_score(y_test_reg, y_pred_all):.2f})")

plt.tight_layout()
plt.show()