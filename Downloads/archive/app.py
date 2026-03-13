import streamlit as st
st.set_page_config(
    page_title="Autism Screening AI Tool",
    page_icon="🧠",
    layout="centered"
)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("autism_screening.csv")

# Select features (A1–A10)
X = data[['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
          'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score']]

# Target column
y = data["Class/ASD"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# App Title
st.title("Autism Screening Tool")

st.write(
"This tool uses Machine Learning to analyze behavioral responses "
"and provide an early autism screening suggestion."
)

st.write("Please answer the following questions:")
st.warning("⚠️ This tool is only for screening purposes and not a medical diagnosis. Please consult a healthcare professional.")

# Questions
a1 = st.selectbox("Does the person look at you when you call their name?", ["No","Yes"])
a2 = st.selectbox("Does the person make eye contact during interaction?", ["No","Yes"])
a3 = st.selectbox("Does the person respond to social interaction?", ["No","Yes"])
a4 = st.selectbox("Does the person use gestures like pointing?", ["No","Yes"])
a5 = st.selectbox("Does the person show interest in other people?", ["No","Yes"])
a6 = st.selectbox("Does the person imitate actions or sounds?", ["No","Yes"])
a7 = st.selectbox("Does the person understand facial expressions?", ["No","Yes"])
a8 = st.selectbox("Does the person engage in pretend play?", ["No","Yes"])
a9 = st.selectbox("Does the person react emotionally to others?", ["No","Yes"])
a10 = st.selectbox("Does the person enjoy social activities?", ["No","Yes"])

# Convert Yes/No to 1/0
answers = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
answers = [1 if ans=="Yes" else 0 for ans in answers]

# Prediction
if st.button("Predict"):
    
    sample = [answers]
    
    prediction = model.predict(sample)
    probability = model.predict_proba(sample)
    confidence = round(probability[0].max()*100,2)
    st.write("Prediction Confidence:", confidence, "%")

    if prediction[0] == 1:
        st.error("⚠ The screening indicates possible Autism Spectrum indicators. Please consult a healthcare professional.")
    else:
        st.success("✅ No significant Autism indicators detected based on this screening.")
import matplotlib.pyplot as plt

scores = answers

fig, ax = plt.subplots()

ax.bar(range(1,11), scores)

ax.set_xlabel("Question Number")
ax.set_ylabel("Response (0 = No, 1 = Yes)")
ax.set_title("Behavioral Response Pattern")

st.pyplot(fig)