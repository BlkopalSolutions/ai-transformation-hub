import streamlit as st
import pandas as pd
import spacy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from transformers import pipeline

# Set page config as the first Streamlit command
st.set_page_config(page_title="AI Transformation Hub", layout="wide")

# Access code check
VALID_CODE = "UNIQUE_CODE_HERE"  # Replace with a unique code per client
access_code = st.text_input("Enter Access Code", type="password")
if access_code != VALID_CODE:
    st.error("Invalid access code. Contact support or purchase at [your payment link].")
    st.stop()

# Load SpaCy model with fallback
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run 'python3 -m spacy download en_core_web_sm' and restart the app.")
    nlp = None

# Load Hugging Face model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Custom CSS for styling (customizable per client)
st.markdown("""
<style>
.main {background-color: #f5f5f5;}
.stButton>button {background-color: #4CAF50; color: white;}
.stTextInput>div>input {border: 2px solid #4CAF50;}
h1 {color: #2E7D32;}
</style>
""", unsafe_allow_html=True)

# Main app interface (customizable title)
st.title("AI Transformation Hub")  # Replace with client-specific title, e.g., "ClientName AI Hub"
st.markdown("### Your AI co-pilot for business transformation")

# Sidebar navigation
phase = st.sidebar.selectbox(
    "Select Phase",
    ["Business Discovery", "AI Solution Matching", "Implementation Blueprint", 
     "ROI Projection", "Performance Optimization"]
)

# Phase 1: Business Discovery with Hugging Face NLP
if phase == "Business Discovery":
    st.header("Phase 1: Business Discovery")
    st.markdown("#### Map your business with advanced AI insights")
    text_input = st.text_area("Describe your business operations or paste a document")
    if st.button("Analyze with NLP"):
        if nlp:
            doc = nlp(text_input)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            sentiment = sentiment_analyzer(text_input)[0]
            st.write("Key Insights:")
            st.write(f"- Entities Detected: {entities}")
            st.write(f"- Sentiment: {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
            if "customer" in text_input.lower():
                st.success("Opportunity: Automate customer support!")
            if "sales" in text_input.lower():
                st.success("Opportunity: Predictive sales modeling!")
        else:
            st.warning("NLP analysis unavailable due to missing SpaCy model.")

# Phase 2: AI Solution Matching
elif phase == "AI Solution Matching":
    st.header("Phase 2: AI Solution Matching")
    st.markdown("#### Tailored AI recommendations")
    industry = st.selectbox("Industry", ["Retail", "Healthcare", "Finance", "Other"])
    budget = st.slider("Monthly Budget ($)", 0, 5000, 1000)
    if st.button("Get Recommendations"):
        st.markdown(f"### Recommendations for {industry} (${budget}):")
        st.write("- Chatbot: Rasa (open-source)")
        st.write("- Automation: PyAutoGUI (free)")
        if budget > 2000:
            st.write("- Predictive Models: Scikit-learn + Cloud Hosting")

# Phase 3: Implementation Blueprint
elif phase == "Implementation Blueprint":
    st.header("Phase 3: Implementation Blueprint")
    st.markdown("#### Your deployment roadmap")
    ai_tool = st.selectbox("Selected AI Tool", ["Chatbot", "Automation", "Predictive Model"])
    if st.button("Generate Blueprint"):
        st.markdown("### Roadmap")
        st.write("- Week 1: Data preparation")
        st.write("- Week 2: Tool setup")
        st.write("- Week 4: Pilot launch")
        roadmap = "Week 1: Data prep\nWeek 2: Tool setup\nWeek 4: Pilot launch"
        st.download_button("Download Roadmap", roadmap, file_name="roadmap.txt")

# Phase 4: ROI Projection with Expanded Scikit-learn
elif phase == "ROI Projection":
    st.header("Phase 4: ROI Projection")
    st.markdown("#### Quantify your AI value")
    cost = st.number_input("Estimated Cost ($)", 0, 10000, 1000)
    hours_saved = st.number_input("Hours Saved per Month", 0, 1000, 50)
    if st.button("Calculate ROI"):
        X_lr = np.array([[cost], [cost*1.2], [cost*0.8]]).reshape(-1, 1)
        y_lr = np.array([hours_saved*20, hours_saved*24, hours_saved*16])
        lr_model = LinearRegression().fit(X_lr, y_lr)
        lr_roi = (lr_model.predict([[cost]])[0] - cost) / cost * 100
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_lr, y_lr)
        rf_roi = (rf_model.predict([[cost]])[0] - cost) / cost * 100
        st.write(f"Linear Regression ROI: {lr_roi:.2f}%")
        st.write(f"Random Forest ROI: {rf_roi:.2f}%")
        st.bar_chart({"Scenario": ["Best", "Expected", "Worst"], 
                      "ROI": [rf_roi*1.2, rf_roi, rf_roi*0.8]})

# Phase 5: Performance Optimization
elif phase == "Performance Optimization":
    st.header("Phase 5: Performance Optimization")
    st.markdown("#### Keep your AI thriving")
    metric = st.slider("Current Performance (%)", 0, 100, 75)
    if st.button("Optimize"):
        st.write(f"Current Performance: {metric}%")
        st.write("Recommendations: Retrain model, scale data inputs.")
        st.line_chart({"Time": [1, 2, 3, 4], 
                       "Performance": [metric, metric+5, metric+10, metric+15]})