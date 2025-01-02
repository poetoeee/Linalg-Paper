import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE  # SMOTE untuk pemerataan kelas
from xgboost import XGBClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt

# Membaca dataset dari URL (Kaggle)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Memeriksa data
print("Data Head:\n", df.head())
print("\nJumlah data hilang di setiap kolom:\n", df.isnull().sum())

# Mengisi nilai yang hilang dengan median (jika ada)
df.fillna(df.median(), inplace=True)

# Memisahkan fitur dan target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Membagi dataset menjadi training dan testing set (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Menampilkan ukuran data training dan testing
print("\nShape data training:", X_train.shape)
print("Shape data testing:", X_test.shape)

# Menggunakan SMOTE untuk pemerataan kelas pada data training
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Penerapan PCA untuk mereduksi dimensi (6 komponen)
pca = PCA(n_components=6)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Menampilkan variansi kumulatif yang dijelaskan oleh PCA
print("\nVariansi yang dijelaskan oleh setiap komponen PCA:", pca.explained_variance_ratio_)
print("Cumulative variance:", pca.explained_variance_ratio_.cumsum())

# Visualisasi Scree Plot untuk PCA
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', label='Cumulative Variance')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Scree Plot - PCA')
plt.legend()
plt.grid()
plt.show()


# Pelatihan dan Evaluasi Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_pca, y_train)
y_pred_rf = rf.predict(X_test_pca)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='macro')
recall_rf = recall_score(y_test, y_pred_rf, average='macro')
f1_rf = f1_score(y_test, y_pred_rf, average='macro')

print("\nRandom Forest Evaluation:")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Precision (macro): {precision_rf:.2f}")
print(f"Recall (macro): {recall_rf:.2f}")
print(f"F1-Score (macro): {f1_rf:.2f}")

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf.classes_)
disp_rf.plot(cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.show()


# Train Random Forest without PCA
rf_no_pca = RandomForestClassifier(random_state=42)
rf_no_pca.fit(X_train, y_train)

# Evaluation without PCA
y_pred_no_pca = rf_no_pca.predict(X_test)
accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)
precision_no_pca = precision_score(y_test, y_pred_no_pca, average='macro')
recall_no_pca = recall_score(y_test, y_pred_no_pca, average='macro')
f1_no_pca = f1_score(y_test, y_pred_no_pca, average='macro')

print("\nRandom Forest Without PCA Evaluation:")
print(f"Accuracy: {accuracy_no_pca:.2f}")
print(f"Precision (macro): {precision_no_pca:.2f}")
print(f"Recall (macro): {recall_no_pca:.2f}")
print(f"F1-Score (macro): {f1_no_pca:.2f}")

# cm_rf = confusion_matrix(y_test, y_pred_no_pca)
# disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf.classes_)
# disp_rf.plot(cmap='Blues')
# plt.title("Confusion Matrix - Random Forest")
# plt.show()


# Eksperimen dengan XGBoost
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_pca, y_train)
y_pred_xgb = xgb.predict(X_test_pca)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb, average='macro')
recall_xgb = recall_score(y_test, y_pred_xgb, average='macro')
f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')

print("\nXGBoost Evaluation:")
print(f"Accuracy: {accuracy_xgb:.2f}")
print(f"Precision (macro): {precision_xgb:.2f}")
print(f"Recall (macro): {recall_xgb:.2f}")
print(f"F1-Score (macro): {f1_xgb:.2f}")

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=xgb.classes_)
disp_xgb.plot(cmap='Blues')
plt.title("Confusion Matrix - XGBoost")
plt.show()

# Eksperimen dengan LightGBM
lgbm = lgb.LGBMClassifier(random_state=42)
lgbm.fit(X_train_pca, y_train)
y_pred_lgbm = lgbm.predict(X_test_pca)

accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
precision_lgbm = precision_score(y_test, y_pred_lgbm, average='macro')
recall_lgbm = recall_score(y_test, y_pred_lgbm, average='macro')
f1_lgbm = f1_score(y_test, y_pred_lgbm, average='macro')

print("\nLightGBM Evaluation:")
print(f"Accuracy: {accuracy_lgbm:.2f}")
print(f"Precision (macro): {precision_lgbm:.2f}")
print(f"Recall (macro): {recall_lgbm:.2f}")
print(f"F1-Score (macro): {f1_lgbm:.2f}")

cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
disp_lgbm = ConfusionMatrixDisplay(confusion_matrix=cm_lgbm, display_labels=lgbm.classes_)
disp_lgbm.plot(cmap='Blues')
plt.title("Confusion Matrix - LightGBM")
plt.show()