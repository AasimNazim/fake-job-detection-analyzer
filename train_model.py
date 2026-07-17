import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from cleaned_data_v2 import build_features


df = pd.read_csv("fake_job_postings.csv")


X, y = build_features(df)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


text_features = "combined_text"


categorical_features = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]


numeric_features = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "text_length",
    "word_count",
    "suspicious_keyword_count",
    "has_suspicious_location",
    "missing_profile",
    "missing_requirements",
    "missing_benefits"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                max_features=40000,
                ngram_range=(1, 2),
                min_df=3,
                sublinear_tf=True,
                strip_accents="unicode"
            ),
            text_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
            ]),
            numeric_features
        )
    ]
)


base_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=2.0,
    solver="liblinear"
)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", CalibratedClassifierCV(base_model, method="sigmoid", cv=3))
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\\nClassification Report:\\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, "job_fraud_model_v2.pkl")
print("\\nSaved model as job_fraud_model_v2.pkl")