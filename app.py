from flask import Flask, render_template, request, Response, session, redirect, url_for
import pickle
import numpy as np
import io
from reportlab.pdfgen import canvas
from datetime import datetime

app = Flask(__name__)
app.secret_key = "heart_ai_secret_key"

model = pickle.load(open("heartt_model.pkl", "rb"))

# ================= HOME =================
@app.route("/")
def home():
    raw_data = session.get("raw_data")

    return render_template(
    "index.html",
    prediction_text=session.get("result"),
    risk=session.get("risk"),
    reasons=session.get("reasons"),
    advice=session.get("advice"),
    raw_data=raw_data
)
    

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    session["raw_data"] = request.form.to_dict()
    age = int(request.form["age"])
    sex = int(request.form["sex"])
    cp = int(request.form["cp"])
    trestbps = int(request.form["trestbps"])
    chol = int(request.form["chol"])
    fbs = int(request.form["fbs"])
    restecg = int(request.form["restecg"])
    thalach = int(request.form["thalach"])
    exang = int(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = int(request.form["slope"])
    ca = int(request.form["ca"])
    thal = int(request.form["thal"])

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)[0]

    try:
        risk_score = model.predict_proba(input_data)[0][1] * 100
    except:
        risk_score = 50

    # ===== REASONS =====
    reasons = []

    # FIXED chest pain logic
    if cp == 0:
        reasons.append("Typical Angina (Chest Pain) detected - High Risk")
    elif cp == 1:
        reasons.append("Atypical Angina symptoms present")

    if age > 55: reasons.append("Age factor (> 55 years)")
    if trestbps > 140: reasons.append("High Blood Pressure")
    if chol > 240: reasons.append("High Cholesterol")
    if exang == 1: reasons.append("Exercise induced chest pain")
    if ca >= 1: reasons.append(f"{ca} blood vessel(s) blockage")

    if prediction == 1 and not reasons:
        reasons.append("Complex cardiovascular pattern detected by AI")

    if not reasons:
        reasons = ["All health indicators are within normal range"]

    # ===== RESULT =====
    if prediction == 1:
        result = "⚠️ HIGH RISK"
        advice = "Immediate cardiologist consultation required."
    else:
        result = "✅ LOW RISK"
        advice = "Maintain healthy lifestyle and regular checkups."

    # ===== SAVE IN SESSION =====
    session["result"] = result
    session["risk"] = round(risk_score, 2)
    session["reasons"] = reasons
    session["advice"] = advice

    return redirect(url_for("home"))

# ================= PDF DOWNLOAD =================
@app.route("/download_pdf", methods=["POST"])
def download_pdf():

    age = int(request.form.get("age"))
    sex = int(request.form.get("sex"))
    cp = int(request.form.get("cp"))
    trestbps = int(request.form.get("trestbps"))
    chol = int(request.form.get("chol"))
    fbs = int(request.form.get("fbs"))
    restecg = int(request.form.get("restecg"))
    thalach = int(request.form.get("thalach"))
    exang = int(request.form.get("exang"))
    oldpeak = float(request.form.get("oldpeak"))
    slope = int(request.form.get("slope"))
    ca = int(request.form.get("ca"))
    thal = int(request.form.get("thal"))

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    risk_score = model.predict_proba(input_data)[0][1] * 100

    result = "HIGH RISK ⚠️" if prediction == 1 else "LOW RISK ✅"

    # ===== PDF =====
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)

    # HEADER
    p.setFont("Helvetica-Bold", 18)
    p.drawString(50, 800, "🏥 Heart Disease Diagnostic Report")
    p.setFont("Helvetica", 10)
    p.drawString(50, 780, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # RESULT
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, 750, f"Final Diagnosis: {result}")
    p.drawString(50, 730, f"Risk Score: {round(risk_score,2)}%")

    # SUMMARY
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, 690, "Patient Summary:")

    p.setFont("Helvetica", 11)
    p.drawString(50, 670, f"Age: {age}")
    p.drawString(50, 655, f"Gender: {'Male' if sex == 1 else 'Female'}")
    p.drawString(50, 640, f"BP: {trestbps}")
    p.drawString(50, 625, f"Cholesterol: {chol}")
    p.drawString(50, 610, f"Heart Rate: {thalach}")

    # INTERPRETATION
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, 570, "Medical Interpretation:")

    p.setFont("Helvetica", 11)
    y = 550

    if cp == 0:
        p.drawString(50, y, "- Typical Angina detected")
        y -= 15
    if age > 55:
        p.drawString(50, y, "- Age risk factor")
        y -= 15
    if trestbps > 140:
        p.drawString(50, y, "- High blood pressure")
        y -= 15
    if chol > 240:
        p.drawString(50, y, "- High cholesterol")
        y -= 15
    if exang == 1:
        p.drawString(50, y, "- Exercise induced pain")
        y -= 15
    if ca >= 1: 
        p.drawString(50,y,f"{ca} blood vessel(s) blockage")
        y-=15

    if y == 550:
        p.drawString(50, y, "- All indicators normal")
    
    # ADVICE
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, 420, "Doctor Recommendation:")

    p.setFont("Helvetica", 11)
    if prediction == 1:
        p.drawString(50, 400, "Immediate cardiologist consultation required.")
    else:
        p.drawString(50, 400, "Maintain healthy lifestyle.")

    # FOOTER
    p.setFont("Helvetica-Oblique", 9)
    p.drawString(50, 50, "AI-generated report - Not a medical diagnosis")

    p.showPage()
    p.save()

    buffer.seek(0)

    return Response(buffer, mimetype='application/pdf',
                    headers={'Content-Disposition': 'attachment;filename=Heart_Report.pdf'})

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)