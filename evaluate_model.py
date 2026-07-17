import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print('Starting evaluation...')
try:
    print('Loading test data...')
    # Load test data
    test_df = pd.read_csv('test_fake_jobs.csv')
    text_col = 'combined_text'
    cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    num_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'text_length', 'word_count']

    X_test = test_df[[text_col] + cat_cols + num_cols]
    y_test = test_df['fraudulent']
    print('Test data loaded, shape:', X_test.shape)
    print('Class distribution:', y_test.value_counts())

    print('Loading model...')
    # Load model
    model = joblib.load('fake_job_detector_model.pkl')
    print('Model loaded')

    print('Predicting...')
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print('Prediction done')

    print('Classification Report:')
    print(classification_report(y_test, y_pred, digits=3))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_prob))
except Exception as e:
    print('Error:', str(e))
    import traceback
    traceback.print_exc()

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print('Classification Report:')
print(classification_report(y_test, y_pred, digits=3))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_prob))