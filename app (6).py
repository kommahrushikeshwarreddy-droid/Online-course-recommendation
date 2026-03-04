import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load model and preprocessing objects ---
model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
data = pickle.load(open("data.pkl", "rb"))


# --- Page config ---
st.set_page_config(page_title="Online Course Recommendation", layout="centered")

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎓 Online Course Recommendation </h1>", unsafe_allow_html=True)
st.write("##### Get personalized course suggestions based on your preferences.")

# --- User Inputs ---
st.markdown("### 📋 Enter Your Details")
col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input("👤 Enter User ID", min_value=1, step=1)
    difficulty = st.selectbox("📈 Preferred Difficulty", ["Beginner", "Intermediate", "Advanced"])
    max_price = st.number_input("💰 Maximum Course Price", min_value=0.0, value=500.0)

with col2:
    min_rating = st.slider("⭐ Minimum Rating", 0.0, 5.0, 3.0, 0.5)
    certification = st.selectbox("🎓 Certification Offered", ["Yes", "No"])
    material = st.selectbox("📘 Study Material Available", ["Yes", "No"])

# --- Recommendation Function ---
def recommend_courses(user_id, difficulty, max_price, min_rating, certification, material, top_n=5):
    user_data = data.copy()
    user_data = user_data[(user_data['course_price'] <= max_price) & 
                          (user_data['rating'] >= min_rating)]
    for col, val in zip(
        ['difficulty_level', 'certification_offered', 'study_material_available'],
        [difficulty, certification, material]
    ):
        le = encoders[col]
        if val in le.classes_:
            encoded_val = le.transform([val])[0]
        else:
            encoded_val = 0
        user_data = user_data[user_data[col] == encoded_val]
    X_scaled = scaler.transform(user_data[features])
    user_data['predicted_interest'] = model.predict_proba(X_scaled)[:, 1]
    for col in ['course_name', 'instructor', 'difficulty_level']:
        user_data[col] = encoders[col].inverse_transform(user_data[col])
    return user_data.sort_values(by='predicted_interest', ascending=False).head(top_n).reset_index(drop=True)

# --- Show Recommendations ---
if st.button("🔍 Show Recommendations"):
    with st.spinner("Analyzing your preferences..."):
        recommendations = recommend_courses(user_id, difficulty, max_price, min_rating, certification, material, top_n=5)
        if recommendations.empty:
            st.warning("⚠️ No matching courses found for your input criteria.")
        else:
            st.success(f"🎯 Top {len(recommendations)} Recommended Courses:")
            for idx, row in recommendations.iterrows():
                interest_level = (
                    "🔥 Highly Recommended" if row['predicted_interest'] >= 0.75 else
                    "👍 Recommended" if row['predicted_interest'] >= 0.5 else
                    "⚪ Consider Later"
                )
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.85); padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <h4>{row['course_name']} — {row['instructor']}</h4>
                    <p>📈 Difficulty: {row['difficulty_level']}  ⭐ Rating: {row['rating']}  💬 Feedback Score: {row['feedback_score']}  💡 {interest_level}</p>
                </div>
                """, unsafe_allow_html=True)

