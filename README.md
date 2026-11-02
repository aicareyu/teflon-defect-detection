# Teflon / fitting defect detection

Care Yu Automation station demo: a live camera inspects cream plastic plumbing fittings (T, elbow, brass-hex adapter) and shows **GOOD** or **DEFECTIVE**.

The live path is OpenCV, not the leftover ResNet training scripts. A part is isolated from the table, compared with photos in `dataset/good`, and failed only when a real mark, puncture, or crushed opening is found.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows you can also double-click `run_app.bat`. Open **http://localhost:8501**.

1. Leave **Input** on **Live camera** and keep **Open camera** on.
2. Set **Camera index** (this station uses **2**) and click **Reconnect** if the picture is blank.
3. Place one fitting in view and hold still. The result appears by itself.
4. Or use **Upload image** → **ANALYZE**.

## Layout

```text
app.py                 Streamlit station UI
src/inspection.py      isolate → match GOOD photos → decide
src/defect_detector.py marks, punctures, crushed openings
dataset/good           reference OK parts
dataset/not_good       labeled defects (evaluation only)
```

## Limits

This is a small labeled set for a demo. Lighting, pose, blur, and camera dropouts still affect scores. It is not a production inspection lock.
