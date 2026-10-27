import os
from pathlib import Path
from google import genai

# Load API key from .env
env_file = Path(__file__).resolve().parent / ".env"

for line in env_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

try:
    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents="Write three short career tips for a beginner Data Analyst."
    )

    print("Response:", repr(response.text))
    print("Candidates:", response.candidates)

except Exception as e:
    print("Error type:", type(e).__name__)
    print("Error details:", str(e))