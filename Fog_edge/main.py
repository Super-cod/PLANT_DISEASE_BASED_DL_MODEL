import base64
import mimetypes
import os
from typing import Optional

import requests
from flask import Flask, jsonify, render_template_string, request

try:
    from asgiref.wsgi import WsgiToAsgi
except ImportError:
    WsgiToAsgi = None

flask_app = Flask(__name__)

PREDICT_URL = os.getenv("PREDICT_URL", "http://85.192.56.254:8000/predict")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDfZSnBHJD3F5Ptelfd6GCWDyFigsM8L-c")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Plant Chat Assistant</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #0b1220;
      color: #e5e7eb;
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
      min-height: 100vh;
    }
    .panel, .chat-card {
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .panel {
      padding: 18px;
    }
    h1, h2, h3 { margin-top: 0; }
    label {
      display: block;
      margin-top: 12px;
      margin-bottom: 6px;
      font-size: 14px;
      color: #cbd5e1;
    }
    input[type="text"], textarea, input[type="file"] {
      width: 100%;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid #334155;
      background: #0f172a;
      color: #f8fafc;
    }
    textarea { min-height: 92px; resize: vertical; }
    button {
      margin-top: 14px;
      width: 100%;
      padding: 12px 16px;
      border: none;
      border-radius: 12px;
      background: #22c55e;
      color: #052e16;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { opacity: 0.94; }
    .chat-card {
      display: flex;
      flex-direction: column;
      min-height: calc(100vh - 48px);
      overflow: hidden;
    }
    .chat-head {
      padding: 18px 20px;
      border-bottom: 1px solid #1f2937;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .chat-box {
      flex: 1;
      padding: 18px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
    }
    .msg {
      max-width: 78%;
      padding: 14px;
      border-radius: 16px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .user {
      align-self: flex-end;
      background: #2563eb;
      color: white;
      border-bottom-right-radius: 6px;
    }
    .assistant {
      align-self: flex-start;
      background: #1f2937;
      color: #f8fafc;
      border-bottom-left-radius: 6px;
      border: 1px solid #334155;
    }
    .meta {
      font-size: 13px;
      color: #cbd5e1;
      margin-top: 8px;
      padding: 10px;
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 12px;
    }
    .preview {
      margin-top: 14px;
      width: 100%;
      border-radius: 14px;
      display: none;
      border: 1px solid #334155;
    }
    .small {
      font-size: 12px;
      color: #94a3b8;
    }
    .row { display: flex; gap: 10px; }
    .row > div { flex: 1; }
    @media (max-width: 900px) {
      .wrap { grid-template-columns: 1fr; }
      .chat-card { min-height: 70vh; }
      .msg { max-width: 92%; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h2>Plant Image + Chat</h2>
      <p class="small">Upload an image, get disease prediction from your server, then send prediction + your question to Gemini.</p>

      <label>Image file</label>
      <input id="image" type="file" accept="image/*" />
      <img id="preview" class="preview" alt="preview" />

      <label>Your message</label>
      <textarea id="message" placeholder="Example: What does this disease mean and what should I do next?"></textarea>

      <button id="sendBtn">Send</button>

      <div class="meta" id="predictionMeta">
        No prediction yet.
      </div>
    </div>

    <div class="chat-card">
      <div class="chat-head">
        <div>
          <h3 style="margin:0;">Gemini Plant Assistant</h3>
          <div class="small">Prediction server + Gemini response</div>
        </div>
      </div>
      <div class="chat-box" id="chatBox">
        <div class="msg assistant">Hi. Upload a plant image and ask me something about it. I will first get the prediction from your model server, then answer using Gemini with that prediction as context.</div>
      </div>
    </div>
  </div>

<script>
const imageInput = document.getElementById('image');
const preview = document.getElementById('preview');
const sendBtn = document.getElementById('sendBtn');
const chatBox = document.getElementById('chatBox');
const predictionMeta = document.getElementById('predictionMeta');
const messageEl = document.getElementById('message');

function addMessage(text, role) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
});

sendBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  const message = messageEl.value.trim();

  if (!file) {
    addMessage('Please upload an image first.', 'assistant');
    return;
  }
  if (!message) {
    addMessage('Please enter a message before sending.', 'assistant');
    return;
  }

  addMessage(message, 'user');
  addMessage('Processing image, getting prediction, and asking Gemini...', 'assistant');

  const formData = new FormData();
  formData.append('file', file);
  formData.append('message', message);

  try {
    const res = await fetch('/chat', { method: 'POST', body: formData });
    const data = await res.json();

    chatBox.removeChild(chatBox.lastChild);

    if (!res.ok) {
      addMessage(data.error || 'Something went wrong.', 'assistant');
      return;
    }

    predictionMeta.innerHTML = `
      <b>class_id:</b> ${data.prediction.class_id}<br>
      <b>class_name:</b> ${data.prediction.class_name}<br>
      <b>confidence:</b> ${Number(data.prediction.confidence).toFixed(6)}
    `;

    addMessage(data.reply, 'assistant');
  } catch (err) {
    chatBox.removeChild(chatBox.lastChild);
    addMessage('Request failed: ' + err.message, 'assistant');
  }
});
</script>
</body>
</html>
"""

def predict_image(image_bytes: bytes, filename: str) -> dict:
    files = {
        "file": (
            filename,
            image_bytes,
            mimetypes.guess_type(filename)[0] or "image/jpeg",
        )
    }
    response = requests.post(PREDICT_URL, files=files, timeout=90)
    response.raise_for_status()
    return response.json()

def gemini_generate(user_message: str, prediction: dict, image_bytes: bytes, mime_type: str) -> str:
    if not GEMINI_API_KEY:
        return (
            "Prediction succeeded, but GEMINI_API_KEY is not set. "
            "Add your Gemini API key to enable AI replies."
        )

    prompt = f"""
You are a helpful plant disease assistant.

The image has already been classified by an external prediction server with:
- class_id: {prediction.get('class_id')}
- class_name: {prediction.get('class_name')}
- confidence: {prediction.get('confidence')}

User message:
{user_message}

Instructions:
- Use the prediction as primary context.
- Also inspect the image briefly for consistency.
- Explain what the prediction likely means in plain English.
- Mention uncertainty if confidence is low or if visual signs may differ.
- Give practical next steps, care advice, and when to verify with an expert.
- Keep the answer concise but useful.
""".strip()

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                ],
            }
        ]
    }
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return f"Gemini returned an unexpected response: {data}"

@flask_app.route("/")
def index():
    return render_template_string(HTML)

@flask_app.route("/chat", methods=["POST"])
def chat():
    uploaded = request.files.get("file")
    user_message = (request.form.get("message") or "").strip()

    if not uploaded:
        return jsonify({"error": "No file uploaded."}), 400
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    image_bytes = uploaded.read()
    filename = uploaded.filename or "image.jpg"
    mime_type = uploaded.mimetype or mimetypes.guess_type(filename)[0] or "image/jpeg"

    try:
        prediction = predict_image(image_bytes, filename)
    except Exception as e:
        return jsonify({"error": f"Prediction request failed: {str(e)}"}), 500

    try:
        reply = gemini_generate(user_message, prediction, image_bytes, mime_type)
    except Exception as e:
        return jsonify({"error": f"Gemini request failed: {str(e)}", "prediction": prediction}), 500

    return jsonify({
        "prediction": prediction,
        "reply": reply,
    })

app = WsgiToAsgi(flask_app) if WsgiToAsgi else flask_app

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=8000, debug=True)
