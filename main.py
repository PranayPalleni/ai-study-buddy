import os
import io
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai
from google.genai import types

app = FastAPI(title="AI Study Chat Assistant")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")
    try:
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        pages_to_read = reader.pages[:15]
        extracted_text = "\n".join([page.extract_text() or "" for page in pages_to_read])
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in this PDF.")
        return {
            "filename": file.filename,
            "pages": len(reader.pages),
            "text": extracted_text[:6000]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Error: {str(e)}")

@app.post("/api/chat")
def chat_endpoint(req: ChatMessage):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing in Render Environment Variables.")

    try:
        system_instruction = (
            "You are a helpful, clear AI Study Assistant. Explain concepts simply, "
            "use bullet points when helpful, and keep answers concise and accurate."
        )

        full_prompt = req.message
        if req.context.strip():
            full_prompt = f"Study Notes Context:\n{req.context[:6000]}\n\nStudent Question: {req.message}"

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
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
        prompt = f"Generate 3 multiple-choice questions based on this topic/notes. Provide 4 options each and the exact matching correct_answer.\n\nContent:\n{topic_text[:6000]}"

        response = client.models.generate_content(
            model="gemini-3.8-flash",
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
