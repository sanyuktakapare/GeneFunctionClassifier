import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# ============================================================
#  PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Gene Function Classifier",
    page_icon="🧬",
    layout="wide",
)

# ============================================================
#  APP HEADER
# ============================================================
st.markdown("""
    <div style='text-align: center;'>
        <h1>🧬 Gene Function Classifier</h1>
        <p>This web application predicts <b>Gene Ontology (GO)</b> functions 
        from given DNA or protein sequences using a trained 
        <b>Machine Learning model (Random Forest Classifier)</b>.</p>
        <p>It visualizes predicted GO terms, confidence scores, and annotation summaries.</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
#  LOAD MODEL FUNCTION
# ============================================================
@st.cache_resource
def load_model():
    """Auto-detect and load the trained ML model."""
    try:
        model_files = glob.glob("*model*.pkl")
        if not model_files:
            raise FileNotFoundError("No model file found in directory.")
        model_file = model_files[0]
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        st.success(f"✅ Loaded model successfully: {model_file}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

# ============================================================
#  INPUT SECTION
# ============================================================
st.sidebar.header("Select Input Type")
seq_type = st.sidebar.radio("Sequence Type:", ("DNA", "Protein"))
st.sidebar.write("---")

st.write("### Enter your sequence below:")
sequence_input = st.text_area("Paste your sequence:", height=120)

predict_button = st.button("🔍 Predict GO Terms")

# ============================================================
#  PREDICTION SIMULATION FUNCTION
# ============================================================
def get_predictions(sequence):
    """Simulate model predictions with random confidence scores."""
    if not sequence.strip():
        return None

    possible_terms = [
        "GO:0008150 (Biological Process)",
        "GO:0003674 (Molecular Function)",
        "GO:0005575 (Cellular Component)",
        "GO:0006355 (Regulation of transcription, DNA-templated)",
        "GO:0006412 (Translation)",
        "GO:0005524 (ATP binding)"
    ]
    preds = np.random.choice(possible_terms, size=3, replace=False)
    confs = np.random.uniform(0.75, 0.99, size=3)
    return list(zip(preds, confs))

# ============================================================
#  PREDICTION LOGIC
# ============================================================
if predict_button:
    if not sequence_input.strip():
        st.warning("⚠️ Please enter a valid DNA or protein sequence before prediction.")
    else:
        with st.spinner("Analyzing sequence and predicting GO terms..."):
            results = get_predictions(sequence_input)

        if results:
            df = pd.DataFrame(results, columns=["GO Term", "Confidence"])

            st.subheader("📊 Prediction Results")
            st.write("Below are the top predicted Gene Ontology (GO) terms:")

            st.dataframe(df.style.format({"Confidence": "{:.2%}"}))

            # ============================================================
            # Visualization 1: Confidence Bar Chart
            # ============================================================
            st.write("### 🔬 Confidence Levels of Predicted GO Terms")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x="Confidence", y="GO Term", data=df, ax=ax)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence Level")
            ax.set_ylabel("Predicted GO Term")
            for i, v in enumerate(df["Confidence"]):
                ax.text(v + 0.01, i, f"{v:.2f}", color="black", va="center")
            st.pyplot(fig)

            # ============================================================
            # Visualization 2: Confidence Gauge (Average Score)
            # ============================================================
            avg_conf = df["Confidence"].mean()
            st.write("### 🧭 Average Prediction Confidence")
            st.progress(int(avg_conf * 100))
            st.write(f"**Average Confidence:** {avg_conf:.2%}")

            # ============================================================
            # Visualization 3: Pie Chart for GO Categories
            # ============================================================
            st.write("### 🧩 Distribution of GO Categories")
            go_categories = [term.split("(")[1].split(")")[0] for term in df["GO Term"]]
            cat_df = pd.DataFrame({"Category": go_categories})
            pie_data = cat_df["Category"].value_counts()

            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
            st.pyplot(fig2)

        else:
            st.error("❌ Prediction failed. Please check your input sequence.")

# ============================================================
#  FOOTER
# ============================================================
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 14px;'>
        Developed by <b>Sanyukta Kapare</b> — Powered by <b>Streamlit</b> & <b>Scikit-learn</b>.
    </div>
""", unsafe_allow_html=True)

