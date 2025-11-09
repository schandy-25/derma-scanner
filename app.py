# app.py
import os
import io
import tempfile
from datetime import datetime
from typing import Dict, Tuple

import gradio as gr
import pandas as pd
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from fastapi import FastAPI

# ---- runtime mode & port ----
PORT = int(os.getenv("PORT", "7860"))
APP_MODE = os.getenv("APP_MODE") or ("gradio" if os.getenv("SPACE_ID") else "gradio")

# ---- lazy predictor ----
PREDICTOR = None
def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        from src.infer import Predictor
        PREDICTOR = Predictor()
    return PREDICTOR

# ---- HAM10000 code -> human label ----
CODE2HUMAN = {
    "akiec": "Actinic keratosis / Intraepithelial carcinoma",
    "bcc":   "Basal cell carcinoma (BCC)",
    "bkl":   "Benign keratosis-like lesion",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevus (mole)",
    "vasc":  "Vascular lesion",
}

LOW_RISK  = {"nv", "bkl", "df", "vasc"}
MED_RISK  = {"bcc", "akiec"}
HIGH_RISK = {"mel"}

def triage_message(top_code: str, conf_pct: float) -> Tuple[str, str]:
    human = CODE2HUMAN.get(top_code, top_code.upper())
    pct = f"{conf_pct:.1f}%"
    if top_code in HIGH_RISK:
        return ("⚠️ Urgent: Consultation Recommended",
                f"Top finding: **{human}** with **{pct}** confidence. "
                "This could be serious—please consult a dermatologist promptly.")
    if top_code in MED_RISK:
        return ("🩺 Consultation Recommended",
                f"Top finding: **{human}** with **{pct}** confidence. "
                "Please have a dermatologist review.")
    return ("✅ Likely Benign (not a diagnosis)",
            f"Top finding: **{human}** with **{pct}** confidence. "
            "Your skin looks okay from this image, but this is **not** a medical diagnosis.")

def make_summary_html(top_code: str, top_pct: float) -> str:
    human = CODE2HUMAN.get(top_code, top_code.upper())
    # Red badge for higher-risk classes, green for likely benign
    risky = {"mel", "bcc", "akiec"}
    color, note = ( "#e53935", "Consult a dermatologist" ) if top_code in risky else ( "#2e7d32", "Likely benign (not a diagnosis)" )
    return f"""
    <div style="border:1px solid #eee; border-radius:14px; padding:18px; background:#fff;">
      <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:12px; height:12px; border-radius:50%; background:{color};"></div>
        <h3 style="margin:0;">Prediction</h3>
      </div>
      <div style="margin-top:10px; font-size:16px;">
        Top finding: <b>{human}</b> <span style="opacity:0.7;">({top_code.upper()})</span><br/>
        Confidence: <b>{top_pct:.1f}%</b>
      </div>
      <div style="margin-top:12px; font-size:14px; color:#555;">
        {note}. This tool is educational and <b>not</b> a medical device.
      </div>
    </div>
    """

# ---- PDF report ----
def make_pdf(image: Image.Image, probs_df: pd.DataFrame, headline: str, subtext: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # header band
    c.setFillColor(colors.HexColor("#b30000"))
    c.rect(0, height - 60, width, 60, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 40, "Derma Scanner — Non-diagnostic AI Report")

    # meta
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 80, f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    c.drawString(40, height - 95, "Dataset base: HAM10000 · This tool is not a medical device.")

    # image (wrap PIL with ImageReader)
    img_w, img_h = 260, 260
    img_reader = ImageReader(image.convert("RGB").resize((img_w, img_h)))
    c.drawImage(img_reader, 40, height - 370, width=img_w, height=img_h,
                preserveAspectRatio=True, mask='auto')

    # headline + text
    c.setFont("Helvetica-Bold", 14)
    c.drawString(320, height - 130, headline)
    c.setFont("Helvetica", 11)
    t = c.beginText(320, height - 150)
    t.textLines(subtext)
    c.drawText(t)

    # table
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 410, "Class probabilities")
    c.setFont("Helvetica", 10)
    y = height - 430
    c.drawString(40, y, "Label (human)")
    c.drawString(320, y, "Code")
    c.drawString(400, y, "Probability (%)")
    c.line(40, y - 3, 520, y - 3)
    y -= 18
    for _, row in probs_df.iterrows():
        if y < 60:
            c.showPage()
            y = height - 60
        c.drawString(40, y, str(row["label (human)"])[:45])
        c.drawString(320, y, str(row["code"]))
        c.drawRightString(520, y, f'{row["probability (%)"]:.2f}')
        y -= 16

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.gray)
    c.drawString(40, 40, "Disclaimer: Educational demo. Not for diagnosis, screening, or treatment decisions.")
    c.save()
    buf.seek(0)
    return buf.read()

# ---- core inference ----
def run_inference(image: Image.Image):
    predictor = get_predictor()
    result = predictor.predict(image)  # {"top_class": code, "probs": {code: float, ...}}

    probs: Dict[str, float] = result["probs"]
    rows = [{
        "label (human)": CODE2HUMAN.get(code, code.upper()),
        "code": code,
        "probability (%)": round(p * 100.0, 4),
    } for code, p in probs.items()]
    df = pd.DataFrame(rows).sort_values("probability (%)", ascending=False).reset_index(drop=True)
    plot_df = df.copy()

    top_code = result["top_class"]
    top_pct = float(probs[top_code] * 100.0)
    title, body = triage_message(top_code, top_pct)

    pdf_bytes = make_pdf(image, df, title, body)
    return title, body, plot_df, df, pdf_bytes

# ---- UI ----
RED = "#b30000"; WHITE = "#ffffff"
INTRO_HTML = f"""
<div style="background:linear-gradient(90deg,{RED} 0%, {RED} 60%, #d11a2a 100%); color:{WHITE}; padding:22px; border-radius:14px; margin-bottom:16px;">
  <h1 style="margin:0; font-size:26px;">Derma Scanner</h1>
  <p style="margin:6px 0 0 0; font-size:14px;">
    Upload a dermatoscopic image to get class probabilities for common skin lesions (HAM10000 classes).
    <b>This is not a diagnosis.</b> Always consult a dermatologist for medical advice.
  </p>
</div>
"""
DISCLAIMER = (
    "**Medical disclaimer:** This demo is for educational purposes only. "
    "It is **not** a medical device and does **not** provide diagnoses."
)

def build_gradio():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="red"),
                   css="body{background:#ffffff;} .container{max-width:1080px; margin:auto;}") as demo:
        gr.HTML(INTRO_HTML)

        with gr.Row():
            with gr.Column(scale=5):
                # File input avoids upload_progress issues
                file_in = gr.File(
                    label="Upload dermatoscopic image (PNG/JPG)",
                    file_types=[".png", ".jpg", ".jpeg"]
                )
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
                gr.Markdown(DISCLAIMER)
            with gr.Column(scale=7):
                result_title = gr.Markdown("")   # triage headline (kept)
                result_body = gr.Markdown("")    # triage body (kept)
                summary_card = gr.HTML("")       # NEW: prediction card
                table = gr.Dataframe(
                    headers=["label (human)", "code", "probability (%)"],
                    row_count=7,
                    col_count=(3, "fixed"),
                    label="Class probabilities (table)",
                )
                pdf_out = gr.File(label="Download PDF report", file_types=[".pdf"])

        def on_click(file_obj):
            if file_obj is None:
                return ("Upload an image to begin.",
                        "Tip: dermatoscopic close-ups work best.",
                        "", None, None)

            img = Image.open(file_obj.name).convert("RGB")
            title, body, plot_df, table_df, pdf_bytes = run_inference(img)

            # Build prediction card from top class
            top_code = table_df.iloc[0]["code"]
            top_pct  = float(table_df.iloc[0]["probability (%)"])
            summary_html = make_summary_html(top_code, top_pct)

            # Persist PDF to a real temp file (gr.File expects a path)
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            pdf_path = os.path.join(tempfile.gettempdir(), f"derma_scanner_report_{ts}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            return title, body, summary_html, table_df, pdf_path

        analyze_btn.click(
            fn=on_click,
            inputs=[file_in],
            outputs=[result_title, result_body, summary_card, table, pdf_out]
        )
    return demo

# ---- optional API stub ----
api = FastAPI(title="Derma Scanner API")
@api.get("/health")
def health():
    return {"status": "ok"}
@api.post("/predict")
def predict_api():
    return {"detail": "Use the Gradio UI for now."}

def run_gradio():
    demo = build_gradio()
    demo.launch(server_name="0.0.0.0", server_port=PORT, show_api=False, share=False)

def run_api():
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    if APP_MODE == "gradio":
        run_gradio()
    elif APP_MODE == "api":
        run_api()
    elif APP_MODE == "both":
        import threading
        t1 = threading.Thread(target=run_api, daemon=True)
        t2 = threading.Thread(target=run_gradio, daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()
    else:
        run_gradio()
