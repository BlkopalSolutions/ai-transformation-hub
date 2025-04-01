import streamlit as st

# Set page config
st.set_page_config(page_title="AI Transformation Hub", layout="wide")

# Title and description
st.title("AI Transformation Hub")
st.write("Your business’s AI co-pilot. Navigate the five-phase AI adoption journey below.")

# Sidebar for navigation
phase = st.sidebar.selectbox(
    "Select Phase",
    ["Business Discovery", "AI Solution Matching", "Implementation Blueprint", 
     "ROI Projection", "Performance Optimization"]
)

# Placeholder for phase content
if phase == "Business Discovery":
    st.header("Phase 1: Business Discovery")
    st.write("Map your business landscape to identify AI opportunities.")
elif phase == "AI Solution Matching":
    st.header("Phase 2: AI Solution Matching")
    st.write("Get tailored AI tool recommendations.")
elif phase == "Implementation Blueprint":
    st.header("Phase 3: Implementation Blueprint")
    st.write("Receive a detailed deployment roadmap.")
elif phase == "ROI Projection":
    st.header("Phase 4: ROI Projection")
    st.write("Quantify the value of your AI investment.")
elif phase == "Performance Optimization":
    st.header("Phase 5: Performance Optimization")
    st.write("Optimize your AI solutions over time.")