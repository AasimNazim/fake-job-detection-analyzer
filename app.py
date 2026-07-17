import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

from cleaned_data_v2 import build_features

# Configure the Streamlit page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="AI Job Fraud Detector", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("job_fraud_model_v2.pkl")

model = load_model()

# ==========================================
# Custom CSS for Modern UI
# ==========================================
st.markdown("""
    <style>
    /* Main Background and text colors */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* Styled Containers / Cards */
    .st-emotion-cache-1wivap2 {
        background-color: #161A25;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
        border: 1px solid #2A3042;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    /* Input fields focus and styles */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
        background-color: #1E2332 !important;
        color: #FFFFFF !important;
        border: 1px solid #333A4D !important;
        border-radius: 8px !important;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus, .stSelectbox>div>div>div:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(37, 99, 235, 0.3);
        border: none;
        color: white;
    }
    
    /* Header Section */
    .header-container {
        text-align: center;
        padding: 40px 0;
        background: radial-gradient(circle at top, #1A2033 0%, #0E1117 100%);
        border-bottom: 1px solid #2A3042;
        margin-bottom: 40px;
        border-radius: 0 0 24px 24px;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #60a5fa, #3b82f6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #94a3b8;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Metrics Box */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Custom divider */
    hr {
        border-color: #2A3042;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# Header Section
# ==========================================
st.markdown("""
    <div class="header-container">
        <div class="main-title">AI Job Fraud Detector</div>
        <div class="sub-title">Detect Fake and Fraudulent Job Postings using Artificial Intelligence. Protect your career with our advanced risk analysis engine.</div>
    </div>
""", unsafe_allow_html=True)


# ==========================================
# Main Layout: Input Form vs Results
# ==========================================
form_col, result_col = st.columns([1.2, 1], gap="large")

with form_col:
    st.markdown("### Job Details Input")
    st.write("Fill in the job posting details below for AI analysis.")
    
    with st.container():
        title = st.text_input("Job Title*", placeholder="e.g. Senior Software Engineer")
        description = st.text_area("Job Description*", height=150, placeholder="Paste the full job description here...")
        
        col_req, col_ben = st.columns(2)
        with col_req:
            requirements = st.text_area("Requirements", height=100, placeholder="Required skills and qualifications...")
        with col_ben:
            benefits = st.text_area("Benefits", height=100, placeholder="Company benefits, perks, etc...")
            
        location = st.text_input("Location", placeholder="e.g. New York, NY or Remote")
        
        st.markdown("##### Additional Information")
        chk_col1, chk_col2, chk_col3 = st.columns(3)
        with chk_col1:
            has_logo = st.checkbox("Company has logo", value=True)
        with chk_col2:
            telecommuting = st.checkbox("Work from home")
        with chk_col3:
            has_questions = st.checkbox("Screening questions")
            
        employment_type = st.selectbox(
            "Employment Type",
            ["unknown", "full-time", "part-time", "contract", "temporary", "other"]
        )
        
        analyze_btn = st.button("Analyze Job Posting")

with result_col:
    st.markdown("### AI Analysis Dashboard")
    
    if analyze_btn:
        if not description.strip() or not title.strip():
            st.error("Please enter at least the Job Title and Job Description to run the analysis.")
        else:
            with st.spinner("AI is analyzing the job posting..."):
                # Data Preparation
                input_df = pd.DataFrame([{
                    "title": title,
                    "company_profile": "",
                    "description": description,
                    "requirements": requirements,
                    "benefits": benefits,
                    "location": location,
                    "employment_type": employment_type,
                    "required_experience": "unknown",
                    "required_education": "unknown",
                    "industry": "unknown",
                    "function": "unknown",
                    "telecommuting": int(telecommuting),
                    "has_company_logo": int(has_logo),
                    "has_questions": int(has_questions)
                }])
                
                # Feature Extraction
                X_input, _ = build_features(input_df)
                
                # Model Prediction
                prediction = model.predict(X_input)[0]
                prob_fake = model.predict_proba(X_input)[0][1]
                prob_real = 1 - prob_fake
                
                # Result logic
                if prob_fake >= 0.80:
                    status_color = "#ef4444" # Red
                    status_text = "High Risk - Likely Fake"
                    bg_color = "rgba(239, 68, 68, 0.1)"
                elif prob_fake >= 0.40:
                    status_color = "#eab308" # Yellow
                    status_text = "Suspicious - Needs Review"
                    bg_color = "rgba(234, 179, 8, 0.1)"
                else:
                    status_color = "#22c55e" # Green
                    status_text = "Safe - Likely Real"
                    bg_color = "rgba(34, 197, 94, 0.1)"
                
                # Final Verdict Banner
                st.markdown(f"""
                <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px;">
                    <h2 style="color: {status_color}; margin: 0;">{status_text}</h2>
                    <p style="margin: 5px 0 0 0; color: #FAFAFA; font-size: 1.1rem;">AI Confidence Score: <b>{max(prob_fake, prob_real)*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualizations
                # Risk Meter (Gauge Chart)
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_fake * 100,
                    title = {'text': "Fraud Risk Score", 'font': {'size': 18, 'color': '#FAFAFA'}},
                    number = {'suffix': "%", 'font': {'color': '#FAFAFA'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': status_color},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "#2A3042",
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(34, 197, 94, 0.2)"},
                            {'range': [40, 80], 'color': "rgba(234, 179, 8, 0.2)"},
                            {'range': [80, 100], 'color': "rgba(239, 68, 68, 0.2)"}],
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FAFAFA"})
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Explanation Section
                st.markdown("#### AI Insights & Indicators")
                indicators = []
                if prob_fake >= 0.5:
                    if not has_logo: indicators.append("- Missing Company Logo (Common in fake posts)")
                    if not has_questions: indicators.append("- No Screening Questions provided")
                    if employment_type == "unknown": indicators.append("- Employment type is not clearly specified")
                    if "dollar" in description.lower() or "money" in description.lower(): indicators.append("- Job description mentions money/dollars heavily")
                    if not indicators:
                        indicators.append("- The language patterns in the description match known fraudulent jobs.")
                else:
                    if has_logo: indicators.append("- Contains Company Logo")
                    if has_questions: indicators.append("- Includes Screening Questions")
                    if employment_type != "unknown": indicators.append(f"- Clear employment type ({employment_type})")
                    if not indicators:
                        indicators.append("- Description aligns well with legitimate industry standards.")
                        
                for ind in indicators:
                    st.write(ind)
                    
    else:
        # Initial Dashboard State before clicking
        st.info("Enter job details on the left and click **Analyze Job Posting** to view the AI analysis.")
        st.markdown("""
            <div style="text-align: center; padding: 40px; color: #64748b;">
                <p>Awaiting data...</p>
            </div>
        """, unsafe_allow_html=True)
