import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE  # SMOTE untuk pemerataan kelas
from xgboost import XGBClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
import time

# Membaca dataset dari URL (Kaggle)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Memeriksa data
df.fillna(df.median(), inplace=True)

# Memisahkan fitur dan target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Membagi dataset menjadi training dan testing set (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Menggunakan SMOTE untuk pemerataan kelas pada data training
# smote = SMOTE(random_state=42)
# X_train_smote, y_train = smote.fit_resample(X_train, y_train)

# Penerapan PCA untuk mereduksi dimensi (6 komponen)
pca = PCA(n_components=6)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Menyimpan metrik untuk grafik
results_pca = {'Random Forest': [], 'XGBoost': [], 'LightGBM': []}
results_no_pca = {'Random Forest': [], 'XGBoost': [], 'LightGBM': []}

# Waktu komputasi untuk dengan PCA dan tanpa PCA
time_pca = {'Random Forest': 0, 'XGBoost': 0, 'LightGBM': 0}
time_no_pca = {'Random Forest': 0, 'XGBoost': 0, 'LightGBM': 0}

def evaluate_model_with_pca(model, X_train_pca, y_train, X_test_pca, y_test):
    start_time = time.time()
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    end_time = time.time()
    elapsed_time = end_time - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1, elapsed_time

def evaluate_model_without_pca(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1, elapsed_time

# List of models to evaluate
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
}

# Loop to evaluate models with and without PCA
for model_name, model in models.items():
    # With PCA
    accuracy, precision, recall, f1, elapsed_time = evaluate_model_with_pca(model, X_train_pca, y_train, X_test_pca, y_test)
    results_pca[model_name].append(accuracy)
    results_pca[model_name].append(precision)
    results_pca[model_name].append(recall)
    results_pca[model_name].append(f1)
    time_pca[model_name] = elapsed_time

    # Without PCA
    accuracy, precision, recall, f1, elapsed_time = evaluate_model_without_pca(model, X_train, y_train, X_test, y_test)
    results_no_pca[model_name].append(accuracy)
    results_no_pca[model_name].append(precision)
    results_no_pca[model_name].append(recall)
    results_no_pca[model_name].append(f1)
    time_no_pca[model_name] = elapsed_time

# Plotting the comparison of metrics
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = range(len(labels))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, results_pca['Random Forest'], label='Random Forest with PCA', marker='o')
ax.plot(x, results_no_pca['Random Forest'], label='Random Forest without PCA', marker='x')
ax.plot(x, results_pca['XGBoost'], label='XGBoost with PCA', marker='o')
ax.plot(x, results_no_pca['XGBoost'], label='XGBoost without PCA', marker='x')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Performance Metrics (PCA vs No PCA)')
ax.legend()
plt.grid(True)
plt.show()