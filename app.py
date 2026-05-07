from flask import Flask, render_template, request, Response, session, redirect, url_for
import pickle
import numpy as np
import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime

app = Flask(__name__)

# ✅ FIX 1: Secret key from environment variable
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

model = pickle.load(open("heartt_model.pkl", "rb"))


# ================= HOME =================
@app.route("/")
def home():
    
    return render_template(
        "index.html",
        prediction_text=session.get("result"),
        risk=session.get("risk"),
        reasons=session.get("reasons"),
        advice=session.get("advice"),
        raw_data=session.get("raw_data")
    )


# ================= INPUT PARSING (SAFE) =================
def parse_form(form):
    """
    ✅ FIX 2: Centralized safe input parsing with validation.
    Returns (data_dict, error_string). If error_string is not None, parsing failed.
    """
    try:
        data = {
            "age":      int(form.get("age", 0)),
            "sex":      int(form.get("sex", 0)),
            "cp":       int(form.get("cp", 0)),
            "trestbps": int(form.get("trestbps", 0)),
            "chol":     int(form.get("chol", 0)),
            "fbs":      int(form.get("fbs", 0)),
            "restecg":  int(form.get("restecg", 0)),
            "thalach":  int(form.get("thalach", 0)),
            "exang":    int(form.get("exang", 0)),
            "oldpeak":  float(form.get("oldpeak", 0.0)),
            "slope":    int(form.get("slope", 0)),
            "ca":       int(form.get("ca", 0)),
            "thal":     int(form.get("thal", 0)),
        }

        # Basic range checks
        if not (1 <= data["age"] <= 120):
            return None, "Age must be between 1 and 120."
        if not (50 <= data["trestbps"] <= 250):
            return None, "Blood pressure seems out of range (50-250)."
        if not (100 <= data["chol"] <= 600):
            return None, "Cholesterol seems out of range (100-600)."

        return data, None

    except (ValueError, TypeError) as e:
        return None, f"Invalid input: {e}"


# ================= RISK ENGINE =================
def calculate_risk(d):
    """
    ✅ FIX 3: Now uses ALL features, not hardcoded zeros.
    d = full dict from parse_form()
    """
    age      = d["age"]
    sex      = d["sex"]
    cp       = d["cp"]
    trestbps = d["trestbps"]
    chol     = d["chol"]
    fbs      = d["fbs"]
    restecg  = d["restecg"]
    thalach  = d["thalach"]
    exang    = d["exang"]
    oldpeak  = d["oldpeak"]
    slope    = d["slope"]
    ca       = d["ca"]
    thal     = d["thal"]

    try:
        features = np.array([[age, sex, cp, trestbps, chol, fbs,
                               restecg, thalach, exang, oldpeak, slope, ca, thal]])
        risk_score = model.predict_proba(features)[0][1] * 100
    except Exception as e:
        print(f"[Model Error] {e}")
        risk_score = 50  # fallback

    risk_flags = 0
    reasons = []

    # Chest pain type
    if cp == 0:
        reasons.append("Typical Angina detected (high cardiac risk)")
        risk_flags += 2
    elif cp == 1:
        reasons.append("Atypical Angina symptoms")
        risk_flags += 1
    elif cp == 2:
        reasons.append("Non-anginal chest pain (lower cardiac risk)")
    elif cp == 3:
        # ✅ FIX 4: Asymptomatic is clinically HIGH risk
        reasons.append("Asymptomatic — silent cardiac risk (often missed)")
        risk_flags += 2

    # Age
    if age < 25:
        reasons.append("Young age (protective factor)")
        risk_flags -= 2
    elif age > 55:
        reasons.append("Age above 55 (risk factor)")
        risk_flags += 2

    # Blood pressure
    if trestbps > 140:
        reasons.append(f"High blood pressure ({trestbps} mmHg)")
        risk_flags += 1

    # Cholesterol
    if chol > 240:
        reasons.append(f"High cholesterol ({chol} mg/dL)")
        risk_flags += 1

    # Fasting blood sugar
    if fbs == 1:
        reasons.append("Fasting blood sugar > 120 mg/dL (diabetic risk)")
        risk_flags += 1

    # Max heart rate — low thalach is a risk
    if thalach <100:
        reasons.append(f" maximum heart rate ({thalach} bpm)")
        risk_flags += 1

    # Exercise angina
    if exang == 1:
        reasons.append("Exercise-induced chest pain")
        risk_flags += 2

    # ST depression
    if oldpeak > 2.0:
        reasons.append(f"Significant ST depression (oldpeak={oldpeak})")
        risk_flags += 2

    # Vessel blockage
    if ca >= 1:
        reasons.append(f"{ca} major vessel(s) with blockage detected")
        risk_flags += 3

    if not reasons:
        reasons = ["All health indicators appear normal"]

    # ✅ FIX 5: Balanced scoring — capped rule contribution
    #rule_score = min(max(risk_flags, 0), 8)
    #final_score = (risk_score * 0.65) + (rule_score * 3)
    
    rule_boost = min(max(risk_flags, 0), 6)

    final_score = risk_score + (rule_boost * 2)

    final_score = min(final_score, 95)
    return round(final_score, 2), reasons


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    data, error = parse_form(request.form)

    if error:
        session["result"] = f"Input Error: {error}"
        session["risk"] = None
        session["reasons"] = []
        session["advice"] = "Please correct your inputs and try again."
        session["raw_data"] = {}
        return redirect(url_for("home"))

    final_score, reasons = calculate_risk(data)

    if final_score >= 70:
        result = "HIGH RISK"
        advice = "Consult a cardiologist immediately."
    elif final_score >= 40:
        result = "MODERATE RISK"
        advice = "Regular checkups and lifestyle changes recommended."
    else:
        result = "LOW RISK"
        advice = "Maintain a healthy lifestyle."

    # ✅ FIX 6: Store EVERYTHING in session — PDF will read from here, not re-parse form
    session["result"]   = result
    session["risk"]     = final_score
    session["reasons"]  = reasons
    session["advice"]   = advice
    session["raw_data"] = data   # store parsed dict, not raw form strings

    return redirect(url_for("home"))


# ================= PDF DOWNLOAD =================
@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    """
    ✅ FIX 7: Reads entirely from session — no form re-parsing, no re-calculation.
    This means refresh / back-button won't change the result.
    """
    result      = session.get("result")
    final_score = session.get("risk")
    reasons     = session.get("reasons", [])
    advice      = session.get("advice")
    d           = session.get("raw_data", {})

    if not result or final_score is None:
        return "No prediction found. Please run a prediction first.", 400

    # ---- Build PDF using Platypus (no hardcoded Y coords) ----
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=inch * 0.75,
        leftMargin=inch * 0.75,
        topMargin=inch,
        bottomMargin=inch
    )

    styles = getSampleStyleSheet()
    story  = []

    # Title
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=20, textColor=colors.HexColor("#C0392B"), spaceAfter=6
    )
    story.append(Paragraph("Heart Disease Diagnostic Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 14))

    # Risk result banner
    if final_score >= 70:
        banner_color = colors.HexColor("#C0392B")
    elif final_score >= 40:
        banner_color = colors.HexColor("#E67E22")
    else:
        banner_color = colors.HexColor("#27AE60")

    result_style = ParagraphStyle(
        "Result", parent=styles["Heading1"],
        textColor=banner_color, fontSize=16
    )
    story.append(Paragraph(f"Diagnosis: {result}", result_style))
    story.append(Paragraph(f"Risk Score: {final_score}%", styles["Heading2"]))
    story.append(Spacer(1, 12))

    # Patient summary table
    story.append(Paragraph("Patient Summary", styles["Heading2"]))
    story.append(Spacer(1, 6))

    sex_label = "Male" if d.get("sex") == 1 else "Female"
    cp_map = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic"
}

    cp_label = cp_map.get(d.get("cp"), "Unknown")
    summary_data = [
        ["Field", "Value"],
        ["Age",             str(d.get("age", "N/A"))],
        ["Gender",          sex_label],
        ["Blood Pressure",  f"{d.get('trestbps', 'N/A')} mmHg"],
        ["Cholesterol",     f"{d.get('chol', 'N/A')} mg/dL"],
        ["Chest Pain Type", cp_label],
        ["Max Heart Rate",  f"{d.get('thalach', 'N/A')} bpm"],
        ["Fasting BS>120",  "Yes" if d.get("fbs") == 1 else "No"],
        ["Exercise Angina", "Yes" if d.get("exang") == 1 else "No"],
        ["ST Depression",   str(d.get("oldpeak", "N/A"))],
        ["Vessels Blocked", str(d.get("ca", "N/A"))],
    ]

    table = Table(summary_data, colWidths=[2.5 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F2F3F4"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    # Medical interpretation
    story.append(Paragraph("Medical Interpretation", styles["Heading2"]))
    story.append(Spacer(1, 6))
    for r in reasons:
        story.append(Paragraph(f"• {r}", styles["Normal"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 12))

    # Doctor recommendation
    story.append(Paragraph("Doctor Recommendation", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(advice, styles["Normal"]))
    story.append(Spacer(1, 20))

    # Disclaimer
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey, fontName="Helvetica-Oblique"
    )
    story.append(Paragraph(
        "This report is AI-generated and does not constitute a medical diagnosis. "
        "Always consult a qualified healthcare professional.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)

    return Response(
        buffer,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment;filename=Heart_Report.pdf"}
    )
    session.clear()

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)