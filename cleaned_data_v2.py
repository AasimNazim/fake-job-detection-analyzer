import pandas as pd
import numpy as np
import re


SUSPICIOUS_KEYWORDS = [
    "work from home",
    "quick money",
    "easy money",
    "earn money",
    "earn extra",
    "no experience",
    "no interview",
    "urgent hiring",
    "immediate start",
    "data entry",
    "payment weekly",
    "limited time",
    "click here",
    "apply now",
    "investment",
    "wire transfer",
    "whatsapp",
    "telegram",
    "commission only",
    "crypto",
    "send your cv",
    "text me",
    "inbox me"
]


SUSPICIOUS_LOCATIONS = [
    "unknown",
    "remote",
    "worldwide"
]


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def count_suspicious_keywords(text):
    text = safe_text(text)
    count = 0
    for word in SUSPICIOUS_KEYWORDS:
        if word in text:
            count += 1
    return count


def has_suspicious_location(location):
    location = safe_text(location)
    for item in SUSPICIOUS_LOCATIONS:
        if item in location:
            return 1
    return 0


def clean_dataframe(df):
    df = df.copy()


    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "location"
    ]


    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.lower().str.strip()


    cat_columns = [
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function"
    ]


    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()


    binary_columns = [
        "telecommuting",
        "has_company_logo",
        "has_questions"
    ]


    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)


    df["combined_text"] = (
        df["title"] + " " +
        df["company_profile"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"] + " " +
        df["location"]
    ).str.replace(r"\\s+", " ", regex=True).str.strip()


    df["text_length"] = df["combined_text"].apply(len)
    df["word_count"] = df["combined_text"].apply(lambda x: len(x.split()))
    df["suspicious_keyword_count"] = df["combined_text"].apply(count_suspicious_keywords)
    df["has_suspicious_location"] = df["location"].apply(has_suspicious_location)
    df["missing_profile"] = (df["company_profile"] == "").astype(int)
    df["missing_requirements"] = (df["requirements"] == "").astype(int)
    df["missing_benefits"] = (df["benefits"] == "").astype(int)


    return df


def build_features(df):
    df = clean_dataframe(df)


    feature_columns = [
        "combined_text",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
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


    X = df[feature_columns].copy()
    y = df["fraudulent"].copy() if "fraudulent" in df.columns else None
    return X, y


if __name__ == "__main__":
    df = pd.read_csv("fake_job_postings.csv")
    df = clean_dataframe(df)
    print(df.head())
    print(df.shape)
    df.to_csv("processed_fake_jobs_v2.csv", index=False)
    print("Saved processed_fake_jobs_v2.csv")