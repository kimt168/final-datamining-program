from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from utils import load_models, clean_text, predict_label_and_probs

app = FastAPI()
templates = Jinja2Templates(directory="app")

tfidf, svm_model, mnb_model, label_encoder = load_models()

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "input_text": None,
        "selected_model": "svm",
        "probabilities": {},
        "error": None
    })

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, title: str = Form(...), model: str = Form(...)):
    try:
        cleaned = clean_text(title)
        selected_model = svm_model if model == "svm" else mnb_model
        prediction, probabilities = predict_label_and_probs(cleaned, selected_model, tfidf, label_encoder)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": prediction,
            "input_text": title,
            "selected_model": model,
            "probabilities": probabilities,
            "error": None
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": None,
            "input_text": title,
            "selected_model": model,
            "probabilities": {},
            "error": str(e)
        })