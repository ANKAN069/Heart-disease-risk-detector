"""
app.py - Main Flask application entry point
Heart Disease Risk Detector
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import sqlite3
import os
import pickle
import numpy as np
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "heart_detector_secret_key_2026"  # Change this in production

# ─── Database Paths ───────────────────────────────────────────────────────────
USER_DB   = os.path.join("database", "user.db")
RECORD_DB = os.path.join("database", "record.db")
MODEL_PATH = os.path.join("models", "heart_model.pkl")

# ─── Average values used when user leaves a field blank ──────────────────────
AVERAGE_VALUES = {
    "age":      54,
    "sex":       1,
    "cp":        1,
    "trestbps": 131,
    "chol":     246,
    "fbs":       0,
    "restecg":   1,
    "thalach":  150,
    "exang":     0,
    "oldpeak":   1.0,
    "slope":     1,
    "ca":        0,
    "thal":      2
}

# ─── Helper: get DB connection ────────────────────────────────────────────────
def get_user_db():
    conn = sqlite3.connect(USER_DB)
    conn.row_factory = sqlite3.Row
    return conn

def get_record_db():
    conn = sqlite3.connect(RECORD_DB)
    conn.row_factory = sqlite3.Row
    return conn


# ─── Load the trained model ───────────────────────────────────────────────────
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()


# ─── Determine risk level from prediction probability ────────────────────────
def get_risk_level(probability):
    """
    Returns risk level label and color class based on predicted probability.
    Thresholds are set for UX clarity.
    """
    if probability < 0.25:
        return "Low Risk", "low"
    elif probability < 0.50:
        return "Moderate Risk", "moderate"
    elif probability < 0.75:
        return "High Risk", "high"
    else:
        return "Critical Risk", "critical"


# ─── Identify critical health factors from user input ────────────────────────
def get_critical_factors(data):
    """
    Compares user's input values against known clinical thresholds.
    Returns a list of warning messages for abnormal readings.
    """
    factors = []

    if data["chol"] > 240:
        factors.append("High Cholesterol (>240 mg/dl) — increases arterial blockage risk.")
    elif data["chol"] < 150:
        factors.append("Low Cholesterol (<150 mg/dl) — may indicate underlying illness.")

    if data["trestbps"] > 140:
        factors.append("High Resting Blood Pressure (>140 mmHg) — a key heart disease indicator.")
    elif data["trestbps"] < 90:
        factors.append("Low Resting Blood Pressure (<90 mmHg) — monitor for hypotension.")

    if data["thalach"] < 100:
        factors.append("Low Maximum Heart Rate (<100 bpm) — may indicate poor cardiac output.")

    if data["fbs"] == 1:
        factors.append("Fasting Blood Sugar >120 mg/dl — risk factor for diabetic heart disease.")

    if data["exang"] == 1:
        factors.append("Exercise-Induced Angina — chest pain during exercise is a serious warning sign.")

    if data["oldpeak"] > 2.0:
        factors.append(f"High ST Depression ({data['oldpeak']}) — abnormal heart stress response.")

    if data["ca"] >= 2:
        factors.append(f"{data['ca']} major vessels blocked — significantly elevated coronary risk.")

    if data["cp"] == 0:
        factors.append("Typical Angina (Chest Pain Type 0) — classic heart disease symptom.")

    if data["thal"] == 1:
        factors.append("Fixed Thalassemia Defect — permanent reduction in blood flow to heart.")
    elif data["thal"] == 2:
        factors.append("Reversible Thalassemia Defect — stress-induced reduced blood flow.")

    return factors if factors else ["No specific critical factors detected based on the inputs provided."]


# ─── Home / Prediction page ───────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    user = None
    if "user_id" in session:
        conn = get_user_db()
        user = conn.execute("SELECT name, username FROM users WHERE user_id = ?",(session["user_id"],)).fetchone()
        conn.close()
    return render_template("index.html", user=user)


# ─── Prediction API endpoint ──────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives form data, fills blanks with averages,
    runs model prediction, and returns JSON result.
    """
    form = request.json
    assumed_fields = []

    # Build input vector; substitute average for any missing fields
    input_data = {}
    for field, avg in AVERAGE_VALUES.items():
        raw = form.get(field, "").strip()
        if raw == "" or raw is None:
            input_data[field] = avg
            assumed_fields.append(field)
        else:
            input_data[field] = float(raw)

    # Create numpy array in correct feature order
    features = np.array([[
        input_data["age"],
        input_data["sex"],
        input_data["cp"],
        input_data["trestbps"],
        input_data["chol"],
        input_data["fbs"],
        input_data["restecg"],
        input_data["thalach"],
        input_data["exang"],
        input_data["oldpeak"],
        input_data["slope"],
        input_data["ca"],
        input_data["thal"]
    ]])

    if model is None:
        return jsonify({"error": "Model not found. Please train the model first."}), 500

    # Get prediction probability
    proba = model.predict_proba(features)[0]
    prob_yes = float(proba[1])    # probability of heart disease
    prob_no  = float(proba[0])    # probability of no heart disease

    risk_label, risk_class = get_risk_level(prob_yes)
    critical_factors = get_critical_factors(input_data)

    # If user is logged in, save record to record.db
    if "user_id" in session:
        try:
            conn = get_record_db()
            conn.execute("""
                INSERT INTO records
                  (user_id, age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal,
                   prob_yes, prob_no, risk_label, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                session["user_id"],
                input_data["age"], input_data["sex"], input_data["cp"],
                input_data["trestbps"], input_data["chol"], input_data["fbs"],
                input_data["restecg"], input_data["thalach"], input_data["exang"],
                input_data["oldpeak"], input_data["slope"], input_data["ca"],
                input_data["thal"], prob_yes, prob_no, risk_label,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Record save error: {e}")

    return jsonify({
        "risk_label":       risk_label,
        "risk_class":       risk_class,
        "prob_yes":         round(prob_yes * 100, 1),
        "prob_no":          round(prob_no  * 100, 1),
        "critical_factors": critical_factors,
        "assumed_fields":   assumed_fields
    })


# ─── Login ────────────────────────────────────────────────────────────────────
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    conn = get_user_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ?", (data["username"],)
    ).fetchone()
    conn.close()

    if user and check_password_hash(user["password"], data["password"]):
        session["user_id"] = user["user_id"]
        return jsonify({"success": True, "name": user["name"]})
    return jsonify({"success": False, "message": "Invalid username or password."}), 401


# ─── Logout ───────────────────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ─── Signup page ──────────────────────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.json
        conn = get_user_db()

        # Check if username or email already exists
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = ? OR email = ?",
            (data["username"], data["email"])
        ).fetchone()

        if existing:
            conn.close()
            return jsonify({"success": False, "message": "Username or email already taken."}), 400

        # Save new user with hashed password
        conn.execute("""
            INSERT INTO users (name, phone, email, username, password)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data["name"], data["phone"], data["email"],
            data["username"], generate_password_hash(data["password"])
        ))
        conn.commit()

        # Auto-login after signup
        user = conn.execute(
            "SELECT user_id, name FROM users WHERE username = ?",
            (data["username"],)
        ).fetchone()
        conn.close()

        session["user_id"] = user["user_id"]
        return jsonify({"success": True, "name": user["name"]})

    return render_template("signup.html")


# ─── Past Records API ─────────────────────────────────────────────────────────
@app.route("/records")
def records():
    """Returns the logged-in user's past prediction records as JSON."""
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_record_db()
    rows = conn.execute("""
        SELECT risk_label, prob_yes, prob_no, timestamp
        FROM records
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 20
    """, (session["user_id"],)).fetchall()
    conn.close()

    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    app.run(debug=True)
