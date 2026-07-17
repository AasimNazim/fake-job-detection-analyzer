import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve,
    classification_report
)
import os
from cleaned_data_v2 import build_features

# Create a directory to save visualizations for the presentation
output_dir = "presentation_visualizations"
os.makedirs(output_dir, exist_ok=True)

print("Loading data and model...")
# 1. Load the dataset
df = pd.read_csv("fake_job_postings.csv")

# 2. Build features (same as training pipeline)
X, y = build_features(df)

# 3. Split data to get the exact test set used during training (random_state=42)
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Load the trained model
model = joblib.load("job_fraud_model_v2.pkl")

print("Generating predictions...")
# 5. Get predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Calculate Metrics
print("\n--- Model Performance Metrics ---")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC AUC": roc_auc
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# VISUALIZATIONS FOR PRESENTATION
# ==========================================
print("\nGenerating visualizations...")
sns.set_theme(style="whitegrid", context="talk")

# Plot 1: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real (0)', 'Fake (1)'], 
            yticklabels=['Real (0)', 'Fake (1)'])
plt.title('Confusion Matrix', fontsize=18)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
plt.close()

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
plt.close()

# Plot 3: Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='purple', lw=2, label='PR curve')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=18)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=300)
plt.close()

# Plot 4: Bar Chart of Key Metrics
plt.figure(figsize=(10, 6))
metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
ax = sns.barplot(x='Metric', y='Score', data=metrics_df, palette='viridis')
plt.title('Model Performance Metrics Overview', fontsize=18)
plt.ylim(0, 1.1)
# Add value labels on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.3f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metrics_bar_chart.png"), dpi=300)
plt.close()

print(f"\nAll done! Visualizations have been saved to the '{output_dir}' directory.")
print("You can use these high-resolution images in your PowerPoint presentation.")
