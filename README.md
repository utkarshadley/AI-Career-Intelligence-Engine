# AI Career Intelligence Engine

An educational AI/ML project that combines salary prediction, job-chance estimation, feature-importance endpoints, and AI-generated career guidance in a FastAPI application.

> **Status:** The API has been tested locally by the project author. A hosted deployment is not currently confirmed as working. Model files are downloaded from Hugging Face at startup and are large, so first startup can take time and hosting environments may run out of disk space or memory.

## Features

- **Salary prediction** using a trained scikit-learn model.
- **Job-chance estimation** using a separate trained model.
- **Feature importance** endpoints to help inspect model signals.
- **AI career advice** generated through the Google Gen AI SDK.
- **FastAPI interactive API docs** at `/docs`.
- HTML frontend served by the API, with static assets under `/static`.

## Architecture

- API: FastAPI + Uvicorn
- ML: scikit-learn models serialized as pickle files
- Model storage: Hugging Face model repository `innovatex0/ai-career-intelligence-models`
- Career advice: Google Gen AI API
- Data handling: pandas and NumPy

## Requirements

- Python 3.10+ (the project has been run with Python 3.12)
- Git
- A Google AI Studio / Gemini API key for career-advice functionality
- Internet access for downloading model files from Hugging Face

## Run locally

1. Clone the repository:

   ```bash
   git clone https://github.com/utkarshadley/AI-Career-Intelligence-Engine.git
   cd AI-Career-Intelligence-Engine
   ```

2. Create and activate a virtual environment:

   **Windows (PowerShell):**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create a local `.env` file in the project root (do not commit it):

   ```env
   GEMINI_API_KEY=put_your_own_key_here
   ```

   Keep your API key private. Never paste it into issues, screenshots, source code, or public commits.

5. Start the API:

   ```bash
   uvicorn ai:app --reload
   ```

6. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation. If the frontend is configured in the repository, open [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

On first startup, the application downloads the model artifacts from Hugging Face. This can take significant time and disk space. Subsequent startups can reuse the Hugging Face cache if it remains available.

## API endpoints

The application includes these endpoints:

- `POST /predict-salary` — predict a salary from submitted profile fields.
- `POST /job-chance` — estimate job chance from submitted profile fields.
- `POST /salary-importance` — return salary-model feature-importance information.
- `POST /job-importance` — return job-model feature-importance information.
- `POST /career-advice` — generate personalized career guidance.

Use `/docs` to inspect the current request schemas and execute test requests. Endpoint schemas may evolve with the code.

## Sample input

The following is an **illustrative example only**; it is not a recorded model output and does not guarantee a particular prediction.

```json
{
  "experience_years": 2,
  "skills_count": 5,
  "skills": ["Python", "SQL", "Power BI", "Pandas", "Excel"],
  "location": "India",
  "education_level": "Bachelor's",
  "company_size": "Medium",
  "job_title": "Data Analyst",
  "remote_work": false,
  "certifications": 1,
  "industry": "Technology"
}
```

Use the request schema shown in `/docs` for each endpoint; the complete career-advice schema has additional optional personalization fields.

## Model files and data

Large `.pkl` model files are intentionally excluded from this Git repository. The application retrieves them from the Hugging Face repository configured in `ai.py`. Review that model repository for artifact details and availability.

The prediction outputs depend on the model training data and encoding. The current code labels salary values as **USD**, based on the training data; they should not be interpreted as verified Indian-market salary estimates.

## 📊 Power BI Dashboard

Explore the interactive Power BI dashboard for AI Career Intelligence Engine.

🔗 **[View Power BI Dashboard](https://app.powerbi.com/view?r=eyJrIjoiOTU3MGZlNGYtOWZjNi00MzIxLWE3YTMtNzQyMmYzN2MwMmNiIiwidCI6IjIxMGY5Y2VjLTRkOWEtNGQ1MS1hYjJmLWMxOTBhNDZiZDg0MCJ9)**

## Limitations and responsible use

- Predictions are estimates, not guarantees of salary, employability, hiring, or career outcomes.
- Model accuracy and fairness must be assessed on a suitable held-out dataset before real-world decisions are made.
- Training-data coverage and category encoding can affect results for uncommon job titles, locations, education levels, and industries.
- Career advice depends on the configured third-party AI service and can be incomplete or incorrect.
- API availability depends on internet access, Hugging Face model availability, disk space, memory, and a valid API key.
- A successful local run does not mean a hosted deployment is working. Hosted deployment is not claimed as verified here.
- Do not submit sensitive personal information or private resumes unless you understand how the configured AI provider handles data.

## Repository safety

- `.env` and environment files are excluded by `.gitignore`.
- `*.pkl` files and the large training dataset are excluded from Git.
- Never commit API keys, tokens, private datasets, or personal information.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/)
- [Google Gen AI SDK](https://googleapis.github.io/python-genai/)
- [scikit-learn](https://scikit-learn.org/)
