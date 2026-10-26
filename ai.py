import os
import pickle
from pathlib import Path

import pandas as pd
import re
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from google.genai import types
from google import genai


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].lstrip()

        key, separator, value = line.partition("=")

        if separator and key.strip():
            os.environ.setdefault(
                key.strip(),
                value.strip().strip("\"'")
            )

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY .env file mein nahi mili"
    )

gemma_client = genai.Client(api_key=api_key)


# ==========================================
# 2. INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="AI Career Intelligence Engine",
    description=(
        "Salary prediction, job-chance estimation, "
        "feature importance and AI-powered career advice."
    ),
    version="2.0.0"
)

# Serve the simple HTML frontend from the static/ folder.
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# ==========================================
# 3. LOAD TRAINED ML MODELS
# ==========================================


import pickle
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "innovatex0/ai-career-intelligence-models"

def load_pickle(filename):
    file_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=filename,
        repo_type="model"
    )

    with open(file_path, "rb") as file:
        return pickle.load(file)


salary_model = load_pickle("salary_model.pkl")
salary_columns = load_pickle("columns.pkl")

job_model = load_pickle("job_model.pkl")
job_columns = load_pickle("job_columns.pkl")



# ==========================================
# 4. INPUT DATA SCHEMA
# ==========================================


from pydantic import BaseModel, Field


class InputData(BaseModel):
    # Existing ML inputs — preserve these
    experience_years: int = Field(ge=0, le=50)
    skills_count: int = Field(ge=0, le=100)
    skills: list[str] = Field(default_factory=list)
    location: str
    education_level: str
    company_size: str
    job_title: str
    remote_work: bool = False
    certifications: int = Field(default=0, ge=0, le=50)
    industry: str | None = None

    # Additional personalization inputs
    interests: list[str] = Field(default_factory=list)
    budget: str = "Free resources preferred"
    daily_available_hours: float = Field(default=2, ge=0.5, le=24)
    career_preference: str = "Job"
    risk_tolerance: str = "Medium"
    preferred_language: str = "Hinglish"
    career_stage: str = "Student"
    resume_text: str | None = None
    additional_context: str | None = None



# ==========================================
# 5. PREPARE MODEL INPUT
# ==========================================

# Salaries in the training data look like USD. Keep multiplier 1.0 to show the model's
# own numbers honestly; a USD->INR rate would only inflate a synthetic-data estimate.
SALARY_CURRENCY = "USD"   # training salaries (100k-200k) are USD, not INR
SALARY_MULTIPLIER = 1.0

# Free-text job titles -> the titles the model knows (first match wins).
JOB_TITLE_RULES = [
    (r"full[\s-]*stack|software|\bsde\b|programmer", "Software Engineer"),
    (r"machine learning|\bml\b|\bai\b|deep learning", "Machine Learning Engineer"),
    (r"data scien", "Data Scientist"),
    (r"business analy", "Business Analyst"),
    (r"data analy|\bbi\b|business intelligence|analytics", "Data Analyst"),
    (r"front[\s-]*end|web develop|\bui\b|\breact\b", "Frontend Developer"),
    (r"back[\s-]*end|\bjava\b|\bnode\b|\bapi\b", "Backend Developer"),
    (r"devops|\bsre\b|site reliability", "DevOps Engineer"),
    (r"cloud", "Cloud Engineer"),
    (r"security|cyber", "Cybersecurity Analyst"),
    (r"product (manager|owner)", "Product Manager"),
]

LOCATION_ALIASES = {
    "united states": "USA", "us": "USA", "america": "USA",
    "united kingdom": "UK", "england": "UK", "britain": "UK",
    "wfh": "Remote", "work from home": "Remote",
}

FEATURE_LABELS = (
    ("job_title_", "Job title: "), ("education_level_", "Education: "),
    ("industry_", "Industry: "), ("company_size_", "Company size: "),
    ("location_", "Location: "), ("experience_level_", "Experience level: "),
)


# Categories dropped by get_dummies(drop_first=True) in training = "all zeros" baseline.
BASELINE_TITLE = "AI Engineer"
BASELINE_LOCATION = "Australia"
BASELINE_SIZE = "Enterprise"


def _options(columns, prefix):
    return [c[len(prefix):] for c in columns if c.startswith(prefix)]


def _match(value, options):
    v = (value or "").strip().lower()
    for option in options:
        if option.lower() == v:
            return option
    return None


def map_job_title(text, options):
    exact = _match(text, options)
    if exact:
        return exact
    lowered = (text or "").lower()
    if re.search(r"\bai engineer\b|artificial intelligence engineer", lowered):
        return BASELINE_TITLE
    for pattern, title in JOB_TITLE_RULES:
        if title in options and re.search(pattern, lowered):
            return title
    return None


def map_education(text):
    """Return the training-data label. 'bachelor' is the dropped baseline -> None."""
    t = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    if any(k in t for k in ("phd", "doctor")):
        return "phd"
    if any(k in t for k in ("master", "mca", "mba", "mtech", "msc", "mcom", "postgrad")):
        return "master"
    if any(k in t for k in ("diploma", "polytechnic")):
        return "diploma"
    if any(k in t for k in ("highschool", "12th", "10th", "secondary", "intermediate")):
        return "high school"
    return None


def experience_level(years):
    # Bins inferred from the training sample: 0 -> blank, 1 -> Fresher, 4-5 -> Junior,
    # 6-10 -> Mid, 15+ -> Senior. "Fresher" (and blank) = baseline, all columns 0.
    if years > 10:
        return "Senior"
    if years > 5:
        return "Mid"
    if years > 1:
        return "Junior"
    return None


def prepare_model_input(data, columns):
    """Build ONE row with exactly the training columns. Returns (DataFrame, warnings)."""
    cols = list(columns)
    row = dict.fromkeys(cols, 0)
    warnings = []

    def put(name, value=1):
        if name in row:
            row[name] = value
            return True
        return False

    put("experience_years", data.experience_years)
    put("skills_count", len(data.skills) if data.skills else data.skills_count)
    put("remote_work", 1 if data.remote_work else 0)
    put("certifications", data.certifications)

    # Job title
    titles = _options(cols, "job_title_")
    title = map_job_title(data.job_title, titles)
    if title == BASELINE_TITLE:
        pass  # baseline category: all job_title_* columns stay 0
    elif title:
        put(f"job_title_{title}")
        if title.lower() != data.job_title.strip().lower():
            warnings.append(f"Job title '{data.job_title}' was mapped to '{title}' (closest title the model knows).")
    else:
        warnings.append(f"Job title '{data.job_title}' is not known to the model; the baseline title was used.")

    # Education
    edu = map_education(data.education_level)
    if edu and not put(f"education_level_{edu}"):
        warnings.append(f"Education '{data.education_level}' has no matching model column.")

    # Location
    locations = _options(cols, "location_")
    loc_in = data.location.strip()
    loc = _match(LOCATION_ALIASES.get(loc_in.lower(), loc_in), locations)
    if loc:
        put(f"location_{loc}")
    elif loc_in.lower() == BASELINE_LOCATION.lower():
        pass
    else:
        warnings.append(f"Location '{loc_in}' is not known to the model (known: {', '.join(locations)}); baseline used.")

    # Company size
    size = _match(data.company_size, _options(cols, "company_size_"))
    if size:
        put(f"company_size_{size}")
    elif data.company_size.strip().lower() == BASELINE_SIZE.lower():
        pass
    else:
        warnings.append(f"Company size '{data.company_size}' is not a model category; ignored.")

    # Industry (optional)
    if data.industry:
        ind = _match(data.industry, _options(cols, "industry_"))
        if ind:
            put(f"industry_{ind}")

    # Experience level derived from years
    level = experience_level(data.experience_years)
    if level:
        put(f"experience_level_{level}")

    return pd.DataFrame([row], columns=cols), warnings


def pretty_value(v):
    low = v.lower()
    if low in ("usa", "uk"):
        return v.upper()
    if low == "phd":
        return "PhD"
    return v if any(ch.isupper() for ch in v) else v.title()


def group_label(name):
    """location_India -> 'Location' (one-hot columns are grouped back into their feature)."""
    for prefix, label in FEATURE_LABELS:
        if name.startswith(prefix):
            return label.rstrip(": ")
    return name.replace("_", " ").capitalize()


def top_features(result, n=10):
    result = result.copy()
    result["Importance"] = result["Importance"].astype(float)
    result["Label"] = result["Feature"].map(group_label)
    grouped = result.groupby("Label", as_index=False)["Importance"].sum()
    grouped["Feature"] = grouped["Label"]
    grouped = grouped.sort_values("Importance", ascending=False).head(n)
    return grouped[["Feature", "Label", "Importance"]].to_dict(orient="records")


def positive_probability(df):
    """Probability (0-1) of the positive class from the job model."""
    classes = list(job_model.classes_)
    idx = next((classes.index(c) for c in (1, True, "1", "Yes", "yes") if c in classes), None)
    if idx is None:
        raise ValueError(f"Positive class not found in {classes}")
    return float(job_model.predict_proba(df)[0][idx])


def _clone(data, **changes):
    return data.model_copy(update=changes) if hasattr(data, "model_copy") else data.copy(update=changes)


def what_if(data, columns, score_fn, n=8):
    """Re-score the SAME profile with one thing changed at a time (model-agnostic explanation)."""
    base = score_fn(data)
    n_skills = len(data.skills) if data.skills else data.skills_count
    scenarios = [
        ("+2 years experience", dict(experience_years=min(data.experience_years + 2, 50))),
        ("+3 skills", dict(skills=list(data.skills) + ["extra1", "extra2", "extra3"], skills_count=n_skills + 3)),
        ("+2 certifications", dict(certifications=min(data.certifications + 2, 50))),
        ("Remote work: " + ("No" if data.remote_work else "Yes"), dict(remote_work=not data.remote_work)),
    ]
    for t in _options(columns, "job_title_"):
        if t.lower() != (data.job_title or "").strip().lower():
            scenarios.append((f"Job title: {t}", dict(job_title=t)))
    for e in ("Diploma", "Bachelor", "Master", "PhD"):
        scenarios.append((f"Education: {e}", dict(education_level=e)))
    for s in _options(columns, "company_size_"):
        scenarios.append((f"Company size: {s}", dict(company_size=s)))
    for loc in _options(columns, "location_"):
        scenarios.append((f"Location: {pretty_value(loc)}", dict(location=loc)))
    out = []
    for label, changes in scenarios:
        try:
            score = score_fn(_clone(data, **changes))
        except Exception:
            continue
        if abs(score - base) > 1e-9:
            out.append({"change": label, "score": score, "delta": score - base})
    out.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return {"base": base, "changes": out[:n]}


# ==========================================
# 6. HOME ROUTE
# ==========================================

@app.get("/")
def home():
    index_file = BASE_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AI Career Intelligence Engine is running", "docs": "/docs"}


# ==========================================
# 7. SALARY PREDICTION
# ==========================================

@app.post("/predict-salary")
def predict_salary(data: InputData):
    try:
        df, warnings = prepare_model_input(data, salary_columns)
        prediction = salary_model.predict(df)[0]

        return {
            "predicted_salary": round(float(prediction) * SALARY_MULTIPLIER, 2),
            "currency": SALARY_CURRENCY,
            "warnings": warnings,
            "active_features": [c for c in df.columns if df[c].iloc[0] != 0],
            "note": (
                "This is a model estimate, "
                "not a guaranteed salary."
            )
        }

    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 8. JOB CHANCE PREDICTION
# ==========================================

@app.post("/job-chance")
def job_chance(data: InputData):
    try:
        df, warnings = prepare_model_input(data, job_columns)
        probabilities = job_model.predict_proba(df)[0]
        class_list = list(job_model.classes_)

        positive_index = next(
            (class_list.index(label) for label in (1, True, "1", "Yes", "yes") if label in class_list),
            None,
        )
        if positive_index is None:
            return {"error": f"Positive job-chance class not found. Model classes: {class_list}"}
        probability = float(probabilities[positive_index])

        return {
            "job_chance_percent": round(probability * 100, 2),
            "model_classes": [str(c) for c in class_list],
            "raw_probabilities": [round(float(p), 4) for p in probabilities],
            "warnings": warnings,
            "active_features": [c for c in df.columns if df[c].iloc[0] != 0],
            "note": (
                "Probability that a profile like yours earns an ABOVE-MEDIAN salary in the training data "
                "(the job model was trained on salary > median). It is not a chance of getting hired."
            )
        }

    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 8b. PERSONALISED "WHAT CHANGES MY RESULT" (what-if)
# ==========================================

@app.post("/salary-whatif")
def salary_whatif(data: InputData):
    try:
        score = lambda d: float(salary_model.predict(prepare_model_input(d, salary_columns)[0])[0]) * SALARY_MULTIPLIER
        result = what_if(data, salary_columns, score)
        result["currency"] = SALARY_CURRENCY
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/job-whatif")
def job_whatif(data: InputData):
    try:
        score = lambda d: positive_probability(prepare_model_input(d, job_columns)[0]) * 100
        return what_if(data, job_columns, score)
    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 9. SALARY FEATURE IMPORTANCE
# ==========================================

@app.get("/salary-importance")
def salary_importance():
    try:
        importance = salary_model.feature_importances_
        result = pd.DataFrame({
            "Feature": salary_columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        return top_features(result)

    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 10. JOB MODEL FEATURE IMPORTANCE
# ==========================================

@app.get("/job-importance")
def job_importance():
    try:
        importance = job_model.feature_importances_
        result = pd.DataFrame({
            "Feature": job_columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        return top_features(result)

    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 11. GEMMA AI CAREER ADVICE
# ==========================================

LAST_ADVICE_SOURCE = "Gemma AI"

def format_career_advice(text: str) -> dict:
    sections = {}
    current_section = "Career Overview"
    current_lines = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        heading = re.match(r"^#{1,3}\s*(?:\d+[\.\)]\s*)?(.+)$", line)

        if heading:
            if current_lines:
                sections[current_section] = "\n".join(current_lines)
            current_section = heading.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines)

    # If the model doesn't return Markdown headings, still show its answer.
    if not sections and text.strip():
        sections["Career Advice"] = text.strip()

    return sections


def generate_career_advice(
    experience_years,
    skills_count,
    location,
    education_level,
    company_size,
    job_title,
    skills: list[str] | None = None,
    interests: list[str] | None = None,
    budget: str = "Free resources preferred",
    daily_available_hours: float = 2,
    career_preference: str = "Job",
    risk_tolerance: str = "Medium",
    preferred_language: str = "Hinglish",
    career_stage: str = "Student",
    resume_text: str | None = None,
    additional_context: str | None = None,
):
    """Generate advice from Gemma; extract visible text robustly and use a local fallback if needed."""
    global LAST_ADVICE_SOURCE
    skills = skills or []
    interests = interests or []
    skills_text = ", ".join(skills) if skills else "Not provided"
    interests_text = ", ".join(interests) if interests else "Not provided"

    # Keep the prompt compact. Long prompts can sometimes result in empty output.
    prompt = f"""Act as a practical career advisor. Reply in {preferred_language}.
Create a concise, personalized career plan (maximum 300 words) using only this profile:
Target role: {job_title}
Experience: {experience_years} years
Skills: {skills_text}
Education: {education_level}
Career stage: {career_stage}
Location: {location}
Company size preference: {company_size}
Interests: {interests_text}
Learning budget: {budget}
Study time per day: {daily_available_hours} hours
Preference: {career_preference}
Risk tolerance: {risk_tolerance}
Resume text: {resume_text or 'Not provided'}
Additional context: {additional_context or 'Not provided'}

Use these 8 Markdown headings, each with concise bullets:
## Career Snapshot
## Priority Skill Gaps
## Best-Fit Career Options
## 30/60/90-Day Roadmap
## Market Reality
## Resume & Interview
## Next Actions
## Honest Verdict

Be specific to the given skills and target role. Do not claim a skill is missing if it was not assessed. If resume is not provided, say that resume-specific feedback is limited. Do not invent salary figures, market statistics, sources, or job guarantees. Give exactly 3 next actions."""

    def visible_text(response):
        # response.text can be None when the SDK response contains thought-only parts.
        try:
            text_value = response.text
            if isinstance(text_value, str) and text_value.strip():
                return text_value.strip()
        except Exception:
            pass

        collected = []
        for candidate in (getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            for part in (getattr(content, "parts", None) or []):
                part_text = getattr(part, "text", None)
                is_thought = bool(getattr(part, "thought", False))
                if isinstance(part_text, str) and part_text.strip() and not is_thought:
                    collected.append(part_text.strip())
        return "\n\n".join(collected).strip()

    # Try the full compact prompt, then one shorter retry if the model returns no visible text.
    attempts = [
        (prompt, 1600),
        (
            f"Give a concise {preferred_language} career plan for a {career_stage} targeting {job_title}. "
            f"Education: {education_level}. Experience: {experience_years} years. "
            f"Skills: {skills_text}. Interests: {interests_text}. Location: {location}. "
            f"Budget: {budget}. Study time: {daily_available_hours} hours/day. "
            "Use headings: Snapshot, Skill Gaps, Career Options, 30/60/90-Day Roadmap, "
            "Market Reality, Resume & Interview, 3 Next Actions, Honest Verdict. "
            "Do not invent market statistics or guarantee employment.",
            900,
        ),
    ]

    last_error = None
    for attempt_number, (attempt_prompt, token_limit) in enumerate(attempts, start=1):
        try:
            response = gemma_client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=attempt_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=3000,
                ),
            )
            text_output = visible_text(response)
            finish_reasons = [
                str(getattr(candidate, "finish_reason", "unknown"))
                for candidate in (getattr(response, "candidates", None) or [])
            ]
            print(f"Gemma attempt {attempt_number}; finish reasons: {finish_reasons}")
            print(f"Gemma visible output length: {len(text_output)}")
            if text_output:
                LAST_ADVICE_SOURCE = "Gemma AI"
                return text_output
            last_error = f"Gemma returned no visible text. Finish reasons: {finish_reasons}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"Gemma attempt {attempt_number} failed: {last_error}")

    # Local fallback prevents the endpoint/UI from failing when the provider returns no text.
    # It is explicitly labelled as a fallback, not as Gemma-generated content.
    LAST_ADVICE_SOURCE = "Local fallback (Gemma response unavailable)"
    skill_line = skills_text if skills else "No skills were provided; assess your current level before choosing gaps."
    interest_line = interests_text if interests else "Interests not provided; compare roles through small projects."
    resume_line = (
        "Use measurable project outcomes and relevant tools in your resume."
        if resume_text else
        "Resume text was not provided, so specific resume feedback is limited."
    )
    return f"""## Career Snapshot
- Target: {job_title} | Stage: {career_stage} | Education: {education_level}
- Current skills supplied: {skill_line}
- Interests: {interest_line}

## Priority Skill Gaps
- Compare the target role's job descriptions against your listed skills before calling anything a gap.
- Choose one high-priority missing requirement and practise it with a small project.
- Validate skill priorities against recent vacancies in {location}.

## Best-Fit Career Options
- {job_title}: primary target; build one role-relevant portfolio project.
- A junior or internship-level {job_title} opportunity: suitable to gain practical experience if available.
- An adjacent analytics role: consider it if its required skills overlap with your current skills.

## 30/60/90-Day Roadmap
- Days 1–30: study for {daily_available_hours} hours/day and complete one small project.
- Days 31–60: improve the project, document your analysis, and publish it in a portfolio.
- Days 61–90: tailor your resume and apply to relevant {career_preference.lower()} opportunities.

## Market Reality
- Current vacancies and salary ranges have not been checked by this endpoint. Verify them using recent listings; no market figures are assumed.

## Resume & Interview
- {resume_line}
- Practise explaining one project and how you validated its results.
- Interview practice: “How would you approach a messy dataset?” and “How would you explain an insight to a non-technical stakeholder?”

## Next Actions
1. Find five current job descriptions for {job_title} in {location} and list repeated requirements.
2. Match those requirements against your skills and choose one gap to address.
3. Start one project that demonstrates the target role's core skills.

## Honest Verdict
Your target may be achievable with consistent, role-focused practice, but readiness cannot be confirmed from this profile alone. This is a local fallback because the AI provider returned no usable text; it is not live market research or a job guarantee.

Diagnostic: {last_error or 'No additional error details.'}"""


@app.post("/career-advice")
def career_advice(data: InputData):
    try:
        advice = generate_career_advice(
            experience_years=data.experience_years,
            skills_count=len(data.skills) if data.skills else data.skills_count,
            location=data.location,
            education_level=data.education_level,
            company_size=data.company_size,
            job_title=data.job_title,
            skills=data.skills,
            interests=data.interests,
            budget=data.budget,
            daily_available_hours=data.daily_available_hours,
            career_preference=data.career_preference,
            risk_tolerance=data.risk_tolerance,
            preferred_language=data.preferred_language,
            career_stage=data.career_stage,
            resume_text=data.resume_text,
            additional_context=data.additional_context,
        )

        return {
            "career_advice": format_career_advice(advice),
            "generated_by": LAST_ADVICE_SOURCE,
            "profile_used": {
                "target_role": data.job_title,
                "education": data.education_level,
                "career_stage": data.career_stage,
                "skills_provided": data.skills,
                "interests": data.interests,
                "budget": data.budget,
                "daily_available_hours": data.daily_available_hours,
                "career_preference": data.career_preference,
                "preferred_language": data.preferred_language,
            },
            "disclaimer": (
                "AI-generated career guidance. "
                "Market data has not been independently verified "
                "unless sources are explicitly provided. "
                "No job or salary is guaranteed."
            )
        }

    except Exception as e:
        return {"error": str(e)}