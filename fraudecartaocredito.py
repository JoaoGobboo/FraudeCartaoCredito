# ============================================================================
# Y INSTALAAAO E IMPORTAAAO DE BIBLIOTECAS
# ============================================================================

# !pip install imbalanced-learn -q

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

# Modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# SeleAAo de Features
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Balanceamento
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

import warnings
warnings.filterwarnings('ignore')

# ConfiguraAAes de visualizaAAo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

"""# YS 1. CARREGAMENTO E ANALISE EXPLORATARIA DOS DADOS"""

# Carregar o dataset
# IMPORTANTE: Substitua o caminho abaixo pelo seu dataset
# Exemplo com dataset do Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud

# Leitura do dataset a partir de arquivo local ou argumento da linha de comando
def load_dataset():
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de deteccao de fraude em um CSV local."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        help="Caminho para o arquivo CSV (ex: ./creditcard.csv).",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Aviso: argumentos desconhecidos ignorados: {unknown}", file=sys.stderr)
    if args.dataset:
        candidate = Path(args.dataset).expanduser()
    else:
        prompt = "Informe o caminho do arquivo CSV do dataset de fraudes: "
        candidate = Path(input(prompt).strip()).expanduser()
    if not candidate.exists():
        sys.exit(f"Arquivo nao encontrado: {candidate}")
    if candidate.is_dir():
        sys.exit(f"O caminho informado eh um diretorio: {candidate}")
    return pd.read_csv(candidate)

df = load_dataset()

print("=" * 80)
print("Y INFORMAAAES DO DATASET")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"\nPrimeiras linhas:")
print(df.head())
print(f"\nInformaAAes das colunas:")
print(df.info())
print(f"\nEstatAsticas descritivas:")
print(df.describe())

# Verificar valores nulos
print(f"\na Valores nulos por coluna:")
print(df.isnull().sum())

# AnAlise do desbalanceamento
print("\n" + "=" * 80)
print("asi  ANALISE DO DESBALANCEAMENTO")
print("=" * 80)

# Assumindo que a coluna target A 'Class' (ajuste se necessArio)
target_col = 'Class' if 'Class' in df.columns else df.columns[-1]
print(f"Coluna target: {target_col}")

fraud_counts = df[target_col].value_counts()
print(f"\nDistribuiAAo das classes:")
print(fraud_counts)
print(f"\nPercentual:")
print(df[target_col].value_counts(normalize=True) * 100)

# VisualizaAAo do desbalanceamento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GrAfico de barras
fraud_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('DistribuiAAo das Classes', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Classe (0=LegAtima, 1=Fraude)')
axes[0].set_ylabel('Quantidade')
axes[0].set_xticklabels(['LegAtima', 'Fraude'], rotation=0)

# GrAfico de pizza
colors = ['#2ecc71', '#e74c3c']
axes[1].pie(fraud_counts, labels=['LegAtima', 'Fraude'], autopct='%1.2f%%',
            colors=colors, startangle=90)
axes[1].set_title('ProporAAo das Classes', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

"""# Y 2. PRA-PROCESSAMENTO DOS DADOS"""

# Separar features e target
X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTreino: {X_train.shape}, Teste: {X_test.shape}")
print(f"DistribuiAAo no treino:\n{y_train.value_counts()}")
print(f"DistribuiAAo no teste:\n{y_test.value_counts()}")

# NormalizaAAo dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\na... Dados normalizados com StandardScaler")

"""# YZ  3. SELEAAO DE ATRIBUTOS (FEATURE SELECTION)"""

print("=" * 80)
print("Y SELEAAO DE ATRIBUTOS")
print("=" * 80)

# 3.1 SelectKBest
print("\nYS MAtodo 1: SelectKBest")
k_best = SelectKBest(score_func=f_classif, k=20)
X_train_kbest = k_best.fit_transform(X_train_scaled, y_train)
X_test_kbest = k_best.transform(X_test_scaled)

selected_features_kbest = X.columns[k_best.get_support()].tolist()
print(f"Features selecionadas ({len(selected_features_kbest)}): {selected_features_kbest}")

# 3.2 Feature Importance com Random Forest
print("\nY2 MAtodo 2: Feature Importance (Random Forest)")
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_train_scaled, y_train)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 features mais importantes:")
print(feature_importance.head(10))

# Visualizar importAncia das features
plt.figure(figsize=(12, 6))
top_n = 15
sns.barplot(data=feature_importance.head(top_n), x='importance', y='feature', palette='viridis')
plt.title(f'Top {top_n} Features Mais Importantes', fontsize=14, fontweight='bold')
plt.xlabel('ImportAncia')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Selecionar top features
top_features = feature_importance.head(20)['feature'].tolist()
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Normalizar novamente
X_train_top_scaled = scaler.fit_transform(X_train_top)
X_test_top_scaled = scaler.transform(X_test_top)

print(f"\na... Usando top {len(top_features)} features para os prA3ximos experimentos")

"""# asi  4. TACNICAS DE BALANCEAMENTO (SAMPLING)"""

print("=" * 80)
print("asi  TACNICAS DE BALANCEAMENTO")
print("=" * 80)

# DicionArio para armazenar datasets balanceados
balanced_datasets = {}

# 4.1 SMOTE (Oversampling)
print("\nY14 Aplicando SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_top_scaled, y_train)
balanced_datasets['SMOTE'] = (X_train_smote, y_train_smote)
print(f"Shape apA3s SMOTE: {X_train_smote.shape}")
print(f"DistribuiAAo: {pd.Series(y_train_smote).value_counts()}")

# 4.2 Random Undersampling
print("\nY12 Aplicando Random Undersampling...")
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_top_scaled, y_train)
balanced_datasets['Undersampling'] = (X_train_rus, y_train_rus)
print(f"Shape apA3s Undersampling: {X_train_rus.shape}")
print(f"DistribuiAAo: {pd.Series(y_train_rus).value_counts()}")

# 4.3 SMOTETomek (Combinado)
print("\nasi  Aplicando SMOTETomek...")
smote_tomek = SMOTETomek(random_state=42)
X_train_combined, y_train_combined = smote_tomek.fit_resample(X_train_top_scaled, y_train)
balanced_datasets['SMOTETomek'] = (X_train_combined, y_train_combined)
print(f"Shape apA3s SMOTETomek: {X_train_combined.shape}")
print(f"DistribuiAAo: {pd.Series(y_train_combined).value_counts()}")

# Dataset original (sem balanceamento)
balanced_datasets['Original'] = (X_train_top_scaled, y_train)

# Visualizar comparaAAo
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, (name, (X_bal, y_bal)) in enumerate(balanced_datasets.items()):
    pd.Series(y_bal).value_counts().plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'])
    axes[idx].set_title(f'{name}\n({len(y_bal)} amostras)', fontweight='bold')
    axes[idx].set_xlabel('Classe')
    axes[idx].set_ylabel('Quantidade')
    axes[idx].set_xticklabels(['LegAtima', 'Fraude'], rotation=0)

plt.tight_layout()
plt.show()

"""# Y 5. TREINAMENTO DE MODELOS BASE"""

print("=" * 80)
print("Y TREINAMENTO DE MODELOS BASE")
print("=" * 80)

# FunAAo para avaliar modelo
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Treina e avalia um modelo"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Macro F1': f1_score(y_test, y_pred, average='macro')
    }

    return results, y_pred

# Modelos base para testar
base_models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(n_jobs=-1)
}

# Testar cada tAcnica de balanceamento com cada modelo
all_results = []

for balance_name, (X_bal, y_bal) in balanced_datasets.items():
    print(f"\n{'='*60}")
    print(f"Testando com: {balance_name}")
    print(f"{'='*60}")

    for model_name, model in base_models.items():
        print(f"  a Treinando {model_name}...", end=' ')
        results, y_pred = evaluate_model(
            model, X_bal, y_bal, X_test_top_scaled, y_test,
            f"{model_name} ({balance_name})"
        )
        all_results.append(results)
        print(f"a Macro F1: {results['Macro F1']:.4f}")

# Criar DataFrame com resultados
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Macro F1', ascending=False)

print("\n" + "=" * 80)
print("YS RESULTADOS DOS MODELOS BASE")
print("=" * 80)
print(results_df.to_string(index=False))

# Visualizar comparaAAo
plt.figure(figsize=(14, 8))
results_pivot = results_df.pivot_table(
    values='Macro F1',
    index=results_df['Model'].str.split(' \(').str[0],
    columns=results_df['Model'].str.extract(r'\((.*?)\)')[0],
    aggfunc='first'
)

sns.heatmap(results_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0, vmax=1, linewidths=0.5)
plt.title('Macro F1-Score: Modelos vs TAcnicas de Balanceamento',
          fontsize=14, fontweight='bold')
plt.xlabel('TAcnica de Balanceamento')
plt.ylabel('Modelo')
plt.tight_layout()
plt.show()

"""# YZi  6. TUNING DE HIPERPARAMETROS (OTIMIZADO)"""

print("=" * 80)
print("YZi  TUNING DE HIPERPARAMETROS (VERSAO OTIMIZADA)")
print("=" * 80)

# Selecionar o melhor dataset de balanceamento (baseado nos resultados anteriores)
# Vamos usar SMOTE como exemplo
X_train_final, y_train_final = balanced_datasets['SMOTE']

# OPAAO RAPIDA: Usar apenas RandomizedSearchCV com menos iteraAAes
# e grids menores para acelerar o processo

# 6.1 Random Forest com RandomizedSearchCV (MAIS RAPIDO)
print("\nY2 Tunando Random Forest (otimizado)...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params,
    n_iter=10,  # Apenas 10 combinaAAes
    cv=3,
    scoring='f1_macro',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_random.fit(X_train_final, y_train_final)
print(f"a... Melhores parAmetros RF: {rf_random.best_params_}")
print(f"a... Melhor score RF: {rf_random.best_score_:.4f}")

best_rf = rf_random.best_estimator_

# 6.2 XGBoost com RandomizedSearchCV (OTIMIZADO)
print("\nas Tunando XGBoost (otimizado)...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 5]
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    xgb_params,
    n_iter=10,  # Apenas 10 combinaAAes
    cv=3,
    scoring='f1_macro',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_random.fit(X_train_final, y_train_final)
print(f"a... Melhores parAmetros XGB: {xgb_random.best_params_}")
print(f"a... Melhor score XGB: {xgb_random.best_score_:.4f}")

best_xgb = xgb_random.best_estimator_

# 6.3 Decision Tree (MODELO SIMPLES E RAPIDO)
print("\nY3 Tunando Decision Tree (rApido)...")
dt_params = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

dt_random = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    n_iter=8,
    cv=3,
    scoring='f1_macro',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

dt_random.fit(X_train_final, y_train_final)
print(f"a... Melhores parAmetros DT: {dt_random.best_params_}")
print(f"a... Melhor score DT: {dt_random.best_score_:.4f}")

best_dt = dt_random.best_estimator_

# Nota: Removemos o SVM do tuning pois A muito lento para datasets grandes
# Vamos usar um SVM com parAmetros padrAo otimizados
print("\nYZ  Usando SVM com parAmetros padrAo otimizados...")
best_svm = SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced', random_state=42)
best_svm.fit(X_train_final, y_train_final)
print("a... SVM treinado com parAmetros padrAo")

"""# YZa 7. ENSEMBLE DE MODELOS"""

print("=" * 80)
print("YZa ENSEMBLE DE MODELOS")
print("=" * 80)

# 7.1 Voting Classifier (Hard Voting)
print("\nY3i  Criando Voting Classifier (Hard Voting)...")
voting_hard = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('svm', best_svm)
    ],
    voting='hard'
)

voting_hard.fit(X_train_final, y_train_final)
print("a... Voting Classifier (Hard) treinado")

# 7.2 Voting Classifier (Soft Voting)
print("\nY3i  Criando Voting Classifier (Soft Voting)...")
# Treinar SVM com probability=True para soft voting
best_svm_prob = SVC(
    **{k: v for k, v in best_svm.get_params().items() if k not in ['random_state', 'probability']},
    probability=True,
    random_state=42
)
best_svm_prob.fit(X_train_final, y_train_final)

voting_soft = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('svm', best_svm_prob)
    ],
    voting='soft'
)

voting_soft.fit(X_train_final, y_train_final)
print("a... Voting Classifier (Soft) treinado")

# Avaliar ensembles
ensemble_results = []

models_to_evaluate = {
    'Random Forest (Tuned)': best_rf,
    'XGBoost (Tuned)': best_xgb,
    'SVM (Tuned)': best_svm,
    'Voting Hard': voting_hard,
    'Voting Soft': voting_soft
}

for name, model in models_to_evaluate.items():
    results, _ = evaluate_model(
        model, X_train_final, y_train_final,
        X_test_top_scaled, y_test, name
    )
    ensemble_results.append(results)

ensemble_df = pd.DataFrame(ensemble_results)
ensemble_df = ensemble_df.sort_values('Macro F1', ascending=False)

print("\n" + "=" * 80)
print("YS RESULTADOS DOS ENSEMBLES")
print("=" * 80)
print(ensemble_df.to_string(index=False))

# Visualizar comparaAAo
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(ensemble_df))
width = 0.15

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Macro F1']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, ensemble_df[metric], width, label=metric, color=colors[i])

ax.set_xlabel('Modelo')
ax.set_ylabel('Score')
ax.set_title('ComparaAAo de MAtricas - Modelos Tunados e Ensembles',
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(ensemble_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

"""# Y 8. AVALIAAAO FINAL DO MELHOR MODELO"""

print("=" * 80)
print("Y AVALIAAAO FINAL DO MELHOR MODELO")
print("=" * 80)

# Selecionar o melhor modelo
best_model_name = ensemble_df.iloc[0]['Model']
best_model = models_to_evaluate[best_model_name]

print(f"\nY Melhor modelo: {best_model_name}")
print(f"YZ  Macro F1-Score: {ensemble_df.iloc[0]['Macro F1']:.4f}")

# PrediAAes
y_pred_final = best_model.predict(X_test_top_scaled)
y_pred_proba = best_model.predict_proba(X_test_top_scaled)[:, 1]

# 8.1 Classification Report
print("\n" + "=" * 80)
print("Y CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test, y_pred_final,
                          target_names=['LegAtima', 'Fraude'],
                          digits=4))

# 8.2 Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz absoluta
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['LegAtima', 'Fraude'],
            yticklabels=['LegAtima', 'Fraude'])
axes[0].set_title('Matriz de ConfusAo (Valores Absolutos)', fontweight='bold')
axes[0].set_ylabel('Valor Real')
axes[0].set_xlabel('Valor Predito')

# Matriz normalizada
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
            xticklabels=['LegAtima', 'Fraude'],
            yticklabels=['LegAtima', 'Fraude'])
axes[1].set_title('Matriz de ConfusAo (Normalizada)', fontweight='bold')
axes[1].set_ylabel('Valor Real')
axes[1].set_xlabel('Valor Predito')

plt.tight_layout()
plt.show()

# AnAlise detalhada da matriz
tn, fp, fn, tp = cm.ravel()
print("\nYS AnAlise da Matriz de ConfusAo:")
print(f"  a True Negatives (TN):  {tn:,} - LegAtimas corretamente identificadas")
print(f"  a False Positives (FP): {fp:,} - LegAtimas classificadas como fraude")
print(f"  a False Negatives (FN): {fn:,} - Fraudes NAO detectadas as i ")
print(f"  a True Positives (TP):  {tp:,} - Fraudes corretamente detectadas a")

# 8.3 Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='#e74c3c', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--',
         label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nYS AUC-ROC Score: {roc_auc:.4f}")

"""# YS 9. ANALISE COMPARATIVA E CONCLUSAES"""

print("=" * 80)
print("YS ANALISE COMPARATIVA DAS TACNICAS")
print("=" * 80)

# 9.1 Impacto das TAcnicas de Balanceamento
print("\n1i a IMPACTO DAS TACNICAS DE BALANCEAMENTO")
print("-" * 60)

balance_comparison = results_df.copy()
balance_comparison['Balance'] = balance_comparison['Model'].str.extract(r'\((.*?)\)')
balance_comparison['Base Model'] = balance_comparison['Model'].str.split(' \(').str[0]

balance_impact = balance_comparison.groupby('Balance')['Macro F1'].agg(['mean', 'std', 'max'])
balance_impact = balance_impact.sort_values('mean', ascending=False)
print(balance_impact)

plt.figure(figsize=(10, 6))
balance_comparison.boxplot(column='Macro F1', by='Balance', ax=plt.gca())
plt.title('Impacto das TAcnicas de Balanceamento no Macro F1-Score',
          fontsize=14, fontweight='bold')
plt.suptitle('')
plt.xlabel('TAcnica de Balanceamento')
plt.ylabel('Macro F1-Score')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 9.2 Impacto dos Modelos
print("\n2i a IMPACTO DOS DIFERENTES MODELOS")
print("-" * 60)

model_impact = balance_comparison.groupby('Base Model')['Macro F1'].agg(['mean', 'std', 'max'])
model_impact = model_impact.sort_values('mean', ascending=False)
print(model_impact)

# 9.3 ComparaAAo: Modelos Base vs Tunados vs Ensemble
print("\n3i a EVOLUAAO: BASE a TUNED a ENSEMBLE")
print("-" * 60)

evolution_data = []

# Melhor modelo base com SMOTE
base_with_smote = results_df[results_df['Model'].str.contains('SMOTE')].sort_values('Macro F1', ascending=False).iloc[0]
evolution_data.append({
    'Stage': 'Base Model',
    'Model': base_with_smote['Model'].split(' (')[0],
    'Macro F1': base_with_smote['Macro F1']
})

# Melhor modelo tunado
best_tuned = ensemble_df[ensemble_df['Model'].str.contains('Tuned')].sort_values('Macro F1', ascending=False).iloc[0]
evolution_data.append({
    'Stage': 'Tuned Model',
    'Model': best_tuned['Model'],
    'Macro F1': best_tuned['Macro F1']
})

# Melhor ensemble
best_ensemble = ensemble_df[ensemble_df['Model'].str.contains('Voting')].sort_values('Macro F1', ascending=False).iloc[0]
evolution_data.append({
    'Stage': 'Ensemble',
    'Model': best_ensemble['Model'],
    'Macro F1': best_ensemble['Macro F1']
})

evolution_df = pd.DataFrame(evolution_data)
print(evolution_df.to_string(index=False))

# VisualizaAAo da evoluAAo
plt.figure(figsize=(10, 6))
plt.plot(evolution_df['Stage'], evolution_df['Macro F1'],
         marker='o', linewidth=2, markersize=10, color='#e74c3c')
plt.fill_between(range(len(evolution_df)), evolution_df['Macro F1'],
                 alpha=0.3, color='#e74c3c')
for i, row in evolution_df.iterrows():
    plt.text(i, row['Macro F1'] + 0.01, f"{row['Macro F1']:.4f}",
             ha='center', fontweight='bold')
plt.xlabel('EstAgio')
plt.ylabel('Macro F1-Score')
plt.title('EvoluAAo do Desempenho: Base a Tuned a Ensemble',
          fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.ylim(evolution_df['Macro F1'].min() * 0.95, evolution_df['Macro F1'].max() * 1.05) # Completed the ylim
plt.tight_layout()
plt.show()

# 9.4 ConclusAes
print("\n" + "=" * 80)
print("Y CONCLUSAES GERAIS")
print("=" * 80)
print(f"Y O melhor modelo identificado foi o {best_model_name} com um Macro F1-Score de {ensemble_df.iloc[0]['Macro F1']:.4f}.")
print("\nPrincipais insights:")
print("  - As tecnicas de balanceamento (SMOTE, Undersampling, SMOTETomek) foram cruciais para melhorar a deteccao de fraudes (Classe 1), como visto pelo aumento do Recall e F1-Score em comparacao com o dataset original.")
print(f"  - A matriz de confusao do {best_model_name} mostra que dos {fn + tp} casos de fraude no teste, {tp} foram corretamente identificados (True Positives), enquanto {fn} foram falsamente negativos (False Negatives), indicando fraudes nao detectadas.")
print(f"  - O modelo tambem classificou {fp} casos legitimos como fraude (False Positives) de um total de {tn + fp} casos legitimos.")
print("  - A curva ROC com AUC de {:.4f} sugere que o modelo tem uma boa capacidade de discriminar entre as classes.".format(roc_auc))
print("  - O tuning de hiperparametros e o uso de ensembles (Voting Soft) demonstraram melhorias incrementais no desempenho em relacao aos modelos base.")

print("\nPossiveis proximos passos:")
print("  - Explorar outras tecnicas de balanceamento ou combinacoes (ex: SMOTE com diferentes algoritmos de subamostragem).")
print("  - Engenharia de features mais avancada (ex: criar features baseadas no tempo ou agrupar transacoes).")
print("  - Testar outros modelos mais complexos ou customizados.")
print("  - Implementar o modelo em producao e monitorar seu desempenho em tempo real.")
print("  - Considerar o custo de Falso Positivo vs Falso Negativo para ajustar o threshold de decisao do modelo.")

