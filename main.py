import os
import io
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai
from google.genai import types

app = FastAPI(title="AI Study Chat Assistant")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

FREE_TIER_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-2.5-flash-lite"
]

def generate_with_fallback(contents, config):
    last_error = None
    for model_name in FREE_TIER_MODELS:
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_error = e
            continue
    raise last_error

class ChatMessage(BaseModel):
    message: str
    context: str = ""

class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: str

class QuizResponse(BaseModel):
    questions: list[QuizQuestion]

@app.get("/", response_class=HTMLResponse)
def serve_chat_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    start_page: int = Form(1),
    end_page: int = Form(15)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")
    try:
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        total_pages = len(reader.pages)

        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, end_page)

        if start_idx >= total_pages:
            raise HTTPException(status_code=400, detail="You have reached the end of this PDF!")

        selected_pages = reader.pages[start_idx:end_idx]
        extracted_text = "\n".join([page.extract_text() or "" for page in selected_pages])

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in these pages.")

        return {
            "filename": file.filename,
            "total_pages": total_pages,
            "loaded_from": start_idx + 1,
            "loaded_to": end_idx,
            "has_more": end_idx < total_pages,
            "text": extracted_text[:8000]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Error: {str(e)}")

@app.post("/api/chat")
def chat_endpoint(req: ChatMessage):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        system_instruction = (
            "You are a helpful AI Study Assistant. Explain concepts clearly using structured bullet points."
        )

        full_prompt = req.message
        if req.context.strip():
            full_prompt = f"Study Notes:\n{req.context[:8000]}\n\nStudent Request: {req.message}"

        response = generate_with_fallback(
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

@app.post("/api/quiz", response_model=QuizResponse)
def quiz_endpoint(req: ChatMessage):
    topic_text = req.context if req.context.strip() else req.message
    if len(topic_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Please chat about a topic or upload notes first.")

    try:
        prompt = f"Create 3 multiple-choice questions from these notes. Provide 4 options each and the exact matching correct_answer.\n\nNotes:\n{topic_text[:8000]}"

        response = generate_with_fallback(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuizResponse,
                temperature=0.2,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz Error: {str(e)}")
