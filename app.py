"""
Credit Card Fraud Detection - Interactive Web Application
Built with Gradio for deployment to Hugging Face Spaces.
"""

import os
import numpy as np
import pandas as pd
import joblib
import gradio as gr

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fraud_detection_model.joblib")


def load_model_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. Run 'python train.py' to generate it."
        )
    return joblib.load(MODEL_PATH)


# Load artifacts
artifacts = load_model_artifacts()
model = artifacts["model"]
label_encoders = artifacts["label_encoders"]

CATEGORIES = [
    "shopping_net",
    "grocery_pos",
    "misc_net",
    "shopping_pos",
    "gas_transport",
    "travel",
    "misc_pos",
    "grocery_net",
    "kids_pets",
    "personal_care",
    "entertainment",
    "health_fitness",
    "food_dining",
    "home",
]

DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def predict_fraud(
    amt: float,
    category: str,
    hour: int,
    day_name: str,
    age: int,
    gender: str,
    city_pop: int,
    distance_km: float,
):
    """
    Predicts transaction fraud probability and generates risk analysis.
    """
    # 1. Feature Preprocessing
    day_idx = DAYS_OF_WEEK.index(day_name)
    is_night = 1 if (hour >= 22 or hour <= 4) else 0
    log_amt = np.log1p(amt)

    # Encode category
    cat_le = label_encoders.get("category")
    if cat_le and category in cat_le.classes_:
        cat_encoded = cat_le.transform([category])[0]
    else:
        cat_encoded = 0

    # Encode gender (0 for Female, 1 for Male)
    gender_code = 1 if gender.upper().startswith("M") else 0

    # Assemble feature row in exact training column order
    input_df = pd.DataFrame([{
        "category": cat_encoded,
        "amt": float(amt),
        "gender": gender_code,
        "city_pop": int(city_pop),
        "trans_hour": int(hour),
        "trans_day_of_week": int(day_idx),
        "is_night": int(is_night),
        "age": int(age),
        "distance_km": float(distance_km),
        "log_amt": float(log_amt),
    }])

    # 2. Model Inference
    prob = model.predict_proba(input_df)[0, 1]
    prob_pct = prob * 100

    # 3. Decision Logic & Badging
    if prob >= 0.50:
        verdict = "🔴 FRAUD DETECTED (HIGH RISK)"
        color = "#d9534f"
        recommendation = "🚨 Immediate Action Required: Transaction blocked. Send SMS authorization prompt to cardholder."
    elif prob >= 0.20:
        verdict = "🟡 SUSPICIOUS TRANSACTION (MEDIUM RISK)"
        color = "#f0ad4e"
        recommendation = "⚠️ Secondary Verification: Transaction flagged for two-factor authentication or manual analyst review."
    else:
        verdict = "🟢 LEGITIMATE TRANSACTION (LOW RISK)"
        color = "#5cb85c"
        recommendation = "✅ Approved: Transaction exhibits normal purchasing behavior and has been processed."

    # 4. Explainability & Risk Indicators
    reasons = []
    if is_night == 1:
        reasons.append(f"• 🌙 **Late-Night Spike:** Transaction occurred at {hour}:00 (high-risk window between 10 PM and 4 AM).")
    if amt > 300:
        reasons.append(f"• 💰 **High Transaction Value:** Amount (${amt:.2f}) significantly exceeds median normal purchase ($47.35).")
    if category in ["shopping_net", "misc_net", "grocery_pos"]:
        reasons.append(f"• 🛒 **Elevated Risk Category:** '{category}' has historically higher fraud occurrence rates.")
    if distance_km > 100:
        reasons.append(f"• 📍 **Abnormal Proximity:** Merchant is located {distance_km:.1f} km away from cardholder home address.")

    if not reasons:
        reasons.append("• Standard daytime transaction within normal range and local distance.")

    explanation_md = "\n".join(reasons)

    # 5. Formatted HTML Banner
    banner_html = f"""
    <div style="background-color: {color}; color: white; padding: 18px; border-radius: 10px; text-align: center; font-family: sans-serif;">
        <h2 style="margin: 0; font-size: 24px;">{verdict}</h2>
        <h3 style="margin: 8px 0 0 0; font-weight: normal; font-size: 18px;">Fraud Probability: <b>{prob_pct:.2f}%</b></h3>
    </div>
    """

    prob_dict = {
        "Legitimate": float(1.0 - prob),
        "Fraudulent": float(prob)
    }

    return banner_html, prob_dict, recommendation, explanation_md


# --- Gradio UI Layout ---
with gr.Blocks(title="Credit Card Fraud Detector") as demo:
    gr.Markdown(
        """
        # 💳 Credit Card Fraud Detection System
        ### Real-Time Machine Learning Prediction for Imbalanced Financial Transactions
        *Powered by HistGradientBoosting trained on 1.2+ million transactions using cost-sensitive learning.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Enter Transaction Details")
            amt_input = gr.Number(
                label="Transaction Amount ($)",
                value=85.00,
                minimum=0.01,
                info="Total transaction amount in USD"
            )
            category_input = gr.Dropdown(
                label="Merchant Category",
                choices=CATEGORIES,
                value="shopping_net",
                info="Type of merchant or service"
            )
            with gr.Row():
                hour_input = gr.Slider(
                    label="Transaction Hour (0 - 23)",
                    minimum=0,
                    maximum=23,
                    step=1,
                    value=23,
                    info="24-hour format (e.g., 23 = 11 PM)"
                )
                day_input = gr.Dropdown(
                    label="Day of Week",
                    choices=DAYS_OF_WEEK,
                    value="Friday"
                )
            with gr.Row():
                age_input = gr.Slider(
                    label="Cardholder Age",
                    minimum=18,
                    maximum=100,
                    step=1,
                    value=42
                )
                gender_input = gr.Radio(
                    label="Cardholder Gender",
                    choices=["Female", "Male"],
                    value="Female"
                )
            with gr.Row():
                distance_input = gr.Slider(
                    label="Distance to Merchant (km)",
                    minimum=1.0,
                    maximum=200.0,
                    step=1.0,
                    value=85.0,
                    info="Calculated Haversine distance"
                )
                city_pop_input = gr.Number(
                    label="Cardholder City Population",
                    value=75000,
                    info="Population of resident city"
                )

            predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Real-Time Fraud Assessment")
            output_banner = gr.HTML(
                label="Risk Verdict",
                value="<div style='text-align: center; padding: 20px; color: #888;'>Click 'Analyze Transaction' to score.</div>"
            )
            output_probs = gr.Label(label="Class Probability Distribution", num_top_classes=2)
            output_action = gr.Textbox(label="Bank Automated Action Protocol", interactive=False)
            output_explanation = gr.Markdown("### Key Risk Indicators:\n*Awaiting transaction input...*")

    predict_btn.click(
        fn=predict_fraud,
        inputs=[
            amt_input,
            category_input,
            hour_input,
            day_input,
            age_input,
            gender_input,
            city_pop_input,
            distance_input,
        ],
        outputs=[output_banner, output_probs, output_action, output_explanation],
    )

    gr.Markdown("### 💡 Try Example Scenarios")
    gr.Examples(
        examples=[
            [850.00, "shopping_net", 23, "Friday", 45, "Female", 50000, 120.0],
            [24.50, "food_dining", 13, "Wednesday", 31, "Male", 250000, 4.5],
            [1250.00, "misc_net", 2, "Saturday", 28, "Male", 15000, 95.0],
            [68.20, "grocery_pos", 17, "Sunday", 54, "Female", 80000, 8.2],
            [9.75, "gas_transport", 8, "Tuesday", 39, "Male", 120000, 3.1],
        ],
        inputs=[
            amt_input,
            category_input,
            hour_input,
            day_input,
            age_input,
            gender_input,
            city_pop_input,
            distance_input,
        ],
    )

if __name__ == "__main__":
    demo.launch()
