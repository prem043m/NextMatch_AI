import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
USERS_PATH = DATA_DIR / "users.csv"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"
USER_PROFILES_PATH = DATA_DIR / "user_profiles.csv"
RECOMMENDATION_HISTORY_PATH = DATA_DIR / "recommendation_history.csv"
CREDENTIALS_PATH = DATA_DIR / "credentials.csv"

USER_COLUMNS = [
    "user_id",
    "name",
    "age",
    "location",
    "profession",
    "experience_years",
    "education",
    "skills",
    "mbti",
    "traits",
    "career_goal",
    "networking_intent",
    "interests",
    "professional_summary",
    "about_me",
]

MBTI_OPTIONS = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

NETWORKING_INTENT_OPTIONS = [
    "Find Mentor",
    "Find Mentee",
    "Research Collaboration",
    "Startup Partner",
    "Knowledge Sharing",
    "Career Growth",
    "Professional Networking",
    "Team Building",
]

PROFESSION_OPTIONS = [
    "Data Scientist",
    "ML Engineer",
    "AI Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "DevOps Engineer",
    "Cloud Engineer",
    "Cybersecurity Analyst",
    "Business Analyst",
    "Product Manager",
    "Project Manager",
    "Consultant",
    "Financial Analyst",
    "Investment Advisor",
    "Accountant",
    "Doctor",
    "Nurse",
    "Healthcare Analyst",
    "UI/UX Designer",
    "Graphic Designer",
    "Content Writer",
    "Marketing Specialist",
]

CAREER_GOAL_OPTIONS = [
    "AI Research",
    "Leadership",
    "Startup Founder",
    "Cloud Computing",
    "Data Analytics",
    "Healthcare Innovation",
    "Financial Growth",
    "Cybersecurity",
    "Design Leadership",
    "Product Management",
]

import bcrypt
from src.utils.config import Config

# --- Password Hashing & Security Helpers ---

def hash_password(password: str) -> str:
    # bcrypt automatically generates salt and embeds it in the hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False

def load_credentials_raw() -> pd.DataFrame:
    if CREDENTIALS_PATH.exists() and CREDENTIALS_PATH.stat().st_size > 0:
        df = pd.read_csv(CREDENTIALS_PATH)
        # Ensure all expected columns exist (forward-compatible)
        for col, default in [
            ("email", ""),
            ("failed_attempts", 0),
            ("locked_until", ""),
            ("last_login", ""),
            ("created_at", ""),
        ]:
            if col not in df.columns:
                df[col] = default
        
        # Cast columns to correct pandas types to avoid type conversion crashes (e.g. string to float64)
        df["failed_attempts"] = df["failed_attempts"].fillna(0).astype(int)
        df["locked_until"] = df["locked_until"].fillna("").astype(str)
        df["last_login"] = df["last_login"].fillna("").astype(str)
        df["created_at"] = df["created_at"].fillna("").astype(str)
        df["username"] = df["username"].fillna("").astype(str)
        df["email"] = df["email"].fillna("").astype(str)
        df["password_hash"] = df["password_hash"].fillna("").astype(str)
        df["user_id"] = df["user_id"].fillna("").astype(str)
        df["role"] = df["role"].fillna("").astype(str)
        return df
    columns = ["username", "email", "password_hash", "user_id", "role",
                "failed_attempts", "locked_until", "last_login", "created_at"]
    return pd.DataFrame(columns=columns)

def init_credentials():
    """Initializes credentials file using Config variables, upgrading schema as needed."""
    if CREDENTIALS_PATH.exists():
        try:
            df = pd.read_csv(CREDENTIALS_PATH)
            # Remove legacy SHA-256 schema if present
            if 'salt' in df.columns:
                os.remove(CREDENTIALS_PATH)
        except Exception:
            pass

    if not CREDENTIALS_PATH.exists():
        columns = ["username", "email", "password_hash", "user_id", "role",
                   "failed_attempts", "locked_until", "last_login", "created_at"]
        admin_pass = Config.ADMIN_PASSWORD
        h = hash_password(admin_pass)
        df = pd.DataFrame([{
            "username": Config.ADMIN_NAME,
            "email": Config.ADMIN_EMAIL,
            "password_hash": h,
            "user_id": "ADMIN",
            "role": "admin",
            "failed_attempts": 0,
            "locked_until": "",
            "last_login": "",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }], columns=columns)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CREDENTIALS_PATH, index=False)

# ── Account Lock Constants ─────────────────────────────────────────────────
MAX_FAILED_ATTEMPTS = 5          # lock after this many consecutive failures
LOCK_DURATION_MINUTES = 15       # how long to lock the account


def is_account_locked(row) -> tuple:
    """
    Returns (is_locked: bool, remaining_minutes: int).
    locked_until is a datetime ISO string; empty means not locked.
    """
    locked_until_str = str(row.get("locked_until", "")).strip()
    if not locked_until_str or locked_until_str in ("", "nan", "NaT"):
        return False, 0
    try:
        locked_until = datetime.fromisoformat(locked_until_str)
        if datetime.now() < locked_until:
            remaining = int((locked_until - datetime.now()).total_seconds() / 60) + 1
            return True, remaining
        return False, 0
    except Exception:
        return False, 0


def authenticate_user(username: str, password: str) -> Tuple[bool, str, str, str]:
    """
    Returns (is_authenticated, user_id, role, message).
    Checks account lock, bcrypt password, increments failed_attempts on failure.
    """
    init_credentials()
    df = load_credentials_raw()
    match = df[df["username"].str.lower() == username.lower()]
    if match.empty:
        match = df[df["email"].str.lower() == username.lower()]
        if match.empty:
            return False, "", "", "Username or email not found."

    row = match.iloc[0]
    idx = match.index[0]

    # Check if account is locked
    locked, mins = is_account_locked(row)
    if locked:
        return False, "", "", f"Account locked due to repeated failures. Try again in {mins} minute(s)."

    if verify_password(password, str(row["password_hash"])):
        # Successful login — reset failed attempts and record last_login
        df.loc[idx, "failed_attempts"] = 0
        df.loc[idx, "locked_until"] = ""
        df.loc[idx, "last_login"] = datetime.now().isoformat(timespec="seconds")
        df.to_csv(CREDENTIALS_PATH, index=False)
        return True, str(row["user_id"]), str(row["role"]), "Success"

    # Failed login — increment counter
    current_fails = int(row.get("failed_attempts", 0) or 0) + 1
    df.loc[idx, "failed_attempts"] = current_fails
    if current_fails >= MAX_FAILED_ATTEMPTS:
        lock_until = datetime.now() + timedelta(minutes=LOCK_DURATION_MINUTES)
        df.loc[idx, "locked_until"] = lock_until.isoformat(timespec="seconds")
        df.to_csv(CREDENTIALS_PATH, index=False)
        return False, "", "", f"Too many failed attempts. Account locked for {LOCK_DURATION_MINUTES} minutes."

    df.to_csv(CREDENTIALS_PATH, index=False)
    remaining = MAX_FAILED_ATTEMPTS - current_fails
    return False, "", "", f"Invalid password. {remaining} attempt(s) remaining before lockout."

def register_credentials(username: str, email: str, password: str, user_id: str, role: str = "user") -> Tuple[bool, str]:
    init_credentials()
    df = load_credentials_raw()
    if not df.empty:
        if username.lower() in df["username"].str.lower().tolist():
            return False, "Username already exists."
        non_empty_emails = [e.lower() for e in df["email"].dropna().tolist() if str(e).strip()]
        if email.strip().lower() in non_empty_emails:
            return False, "Email address already registered."

    h = hash_password(password)
    new_cred = {
        "username": username,
        "email": email.strip(),
        "password_hash": h,
        "user_id": user_id,
        "role": role,
        "failed_attempts": 0,
        "locked_until": "",
        "last_login": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([new_cred])], ignore_index=True)
    df.to_csv(CREDENTIALS_PATH, index=False)
    return True, "Registered successfully."

def get_email_for_user(user_id: str) -> str:
    init_credentials()
    df = load_credentials_raw()
    match = df[df["user_id"] == user_id]
    if not match.empty:
        return str(match.iloc[0].get("email", ""))
    return ""


def update_password_hash(user_id: str, new_hash: str) -> bool:
    init_credentials()
    df = load_credentials_raw()
    idx = df[df["user_id"] == user_id].index
    if len(idx) > 0:
        df.loc[idx[0], "password_hash"] = new_hash
        df.loc[idx[0], "failed_attempts"] = 0
        df.loc[idx[0], "locked_until"] = ""
        df.to_csv(CREDENTIALS_PATH, index=False)
        return True
    return False


def update_last_login(user_id: str) -> None:
    """Records timestamp of last successful login."""
    init_credentials()
    df = load_credentials_raw()
    idx = df[df["user_id"] == user_id].index
    if len(idx) > 0:
        df.loc[idx[0], "last_login"] = datetime.now().isoformat(timespec="seconds")
        df.to_csv(CREDENTIALS_PATH, index=False)


def disable_account(user_id: str) -> bool:
    """Locks account indefinitely (locked_until set to year 9999)."""
    init_credentials()
    df = load_credentials_raw()
    idx = df[df["user_id"] == user_id].index
    if len(idx) > 0:
        df.loc[idx[0], "locked_until"] = "9999-12-31T23:59:59"
        df.to_csv(CREDENTIALS_PATH, index=False)
        return True
    return False


def enable_account(user_id: str) -> bool:
    """Clears any lock on the account."""
    init_credentials()
    df = load_credentials_raw()
    idx = df[df["user_id"] == user_id].index
    if len(idx) > 0:
        df.loc[idx[0], "locked_until"] = ""
        df.loc[idx[0], "failed_attempts"] = 0
        df.to_csv(CREDENTIALS_PATH, index=False)
        return True
    return False


# --- User Data & Profile Loading Helpers ---

def load_users_raw() -> pd.DataFrame:
    return pd.read_csv(USERS_PATH)

def load_feedback_raw() -> pd.DataFrame:
    return pd.read_csv(FEEDBACK_PATH)

def get_feedback_for_user(user_id: str) -> pd.DataFrame:
    feedback_df = load_feedback_raw()
    return feedback_df[feedback_df["user_id"] == user_id].copy()

def generate_next_user_id(users_df: pd.DataFrame) -> str:
    numeric_ids = users_df["user_id"].astype(str).str.replace("U", "", regex=False).astype(int)
    return f"U{numeric_ids.max() + 1:03d}"

def generate_next_user_id_combined() -> str:
    users_df = load_users_raw()
    ids = users_df["user_id"].astype(str).str.replace("U", "", regex=False).astype(int).tolist()
    
    if USER_PROFILES_PATH.exists() and USER_PROFILES_PATH.stat().st_size > 0:
        try:
            profiles_df = pd.read_csv(USER_PROFILES_PATH)
            if not profiles_df.empty:
                profile_ids = profiles_df["user_id"].astype(str).str.replace("U", "", regex=False).astype(int).tolist()
                ids.extend(profile_ids)
        except Exception:
            pass
    return f"U{max(ids) + 1:03d}"

def load_user_profiles_raw() -> pd.DataFrame:
    columns = [
        "user_id", "name", "profession", "location", "experience_years", "mbti", 
        "career_goal", "skills", "interests", "networking_intent", 
        "professional_summary", "about_me", "created_at",
        "notif_welcome", "notif_digest", "notif_system"
    ]
    if USER_PROFILES_PATH.exists() and USER_PROFILES_PATH.stat().st_size > 0:
        try:
            df = pd.read_csv(USER_PROFILES_PATH)
            # Ensure new columns exist
            for col in ["notif_welcome", "notif_digest", "notif_system"]:
                if col not in df.columns:
                    df[col] = True
            # Reorder columns to match or add missing ones
            for col in columns:
                if col not in df.columns:
                    df[col] = ""
            return df[columns]
        except Exception:
            pass
    return pd.DataFrame(columns=columns)

def register_user_profile(profile: dict) -> str:
    new_user_id = generate_next_user_id_combined()
    
    new_profile = {
        "user_id": new_user_id,
        "name": profile.get("name", "").strip(),
        "profession": profile.get("profession", "").strip(),
        "location": profile.get("location", "").strip(),
        "experience_years": int(profile.get("experience_years", 0)),
        "mbti": profile.get("mbti", "INTJ"),
        "career_goal": profile.get("career_goal", "").strip(),
        "skills": profile.get("skills", "").strip(),
        "interests": profile.get("interests", "").strip(),
        "networking_intent": profile.get("networking_intent", "Career Growth"),
        "professional_summary": profile.get("professional_summary", "").strip(),
        "about_me": profile.get("about_me", "").strip(),
        "created_at": datetime.now().isoformat(),
        "notif_welcome": True,
        "notif_digest": True,
        "notif_system": True,
    }
    
    columns = [
        "user_id", "name", "profession", "location", "experience_years", "mbti", 
        "career_goal", "skills", "interests", "networking_intent", 
        "professional_summary", "about_me", "created_at",
        "notif_welcome", "notif_digest", "notif_system"
    ]
    
    df = load_user_profiles_raw()
    df = pd.concat([df, pd.DataFrame([new_profile], columns=columns)], ignore_index=True)
    df.to_csv(USER_PROFILES_PATH, index=False)
    return new_user_id

def update_user_profile(user_id: str, profile: dict) -> None:
    if not USER_PROFILES_PATH.exists():
        return
    
    df = load_user_profiles_raw()
    idx = df[df["user_id"] == user_id].index
    if len(idx) > 0:
        for key, value in profile.items():
            if key in ["experience_years"]:
                try:
                    df.loc[idx[0], key] = int(value)
                except ValueError:
                    df.loc[idx[0], key] = 0
            elif key in ["notif_welcome", "notif_digest", "notif_system"]:
                df.loc[idx[0], key] = bool(value)
            elif key in ["user_id", "created_at"]:
                # Do not modify identifiers or creation timestamp
                continue
            else:
                df.loc[idx[0], key] = str(value).strip()
        df.to_csv(USER_PROFILES_PATH, index=False)

# --- Recommendation History Helpers ---

def load_recommendation_history() -> pd.DataFrame:
    if RECOMMENDATION_HISTORY_PATH.exists() and RECOMMENDATION_HISTORY_PATH.stat().st_size > 0:
        return pd.read_csv(RECOMMENDATION_HISTORY_PATH)
    columns = ["user_id", "recommendation_batch_id", "recommended_user_id", "score", "timestamp"]
    return pd.DataFrame(columns=columns)

def append_recommendation_history_batch(user_id: str, batch_id: str, recs: List[Dict], timestamp: str) -> None:
    columns = ["user_id", "recommendation_batch_id", "recommended_user_id", "score", "timestamp"]
    new_records = []
    for rec in recs:
        new_records.append({
            "user_id": user_id,
            "recommendation_batch_id": batch_id,
            "recommended_user_id": rec["user_id"],
            "score": float(rec.get("final_ranking_score", rec.get("ml_score", 0.0))),
            "timestamp": timestamp
        })
    df = load_recommendation_history()
    df = pd.concat([df, pd.DataFrame(new_records, columns=columns)], ignore_index=True)
    df.to_csv(RECOMMENDATION_HISTORY_PATH, index=False)

# --- Legacy Feedback & Base Profile Helpers (Do not modify logic) ---

def append_user_profile(profile: Dict) -> str:
    users_df = load_users_raw()
    new_user_id = generate_next_user_id(users_df)

    new_user = {"user_id": new_user_id}
    for column in USER_COLUMNS[1:]:
        new_user[column] = profile.get(column, "")

    updated_users = pd.concat(
        [users_df, pd.DataFrame([new_user], columns=USER_COLUMNS)],
        ignore_index=True,
    )
    updated_users.to_csv(USERS_PATH, index=False)
    return new_user_id

def append_feedback(user_id: str, matched_user_id: str, action: int) -> None:
    feedback_df = load_feedback_raw()
    feedback_df.loc[len(feedback_df)] = [
        user_id,
        matched_user_id,
        int(action),
        datetime.now().isoformat(timespec="seconds"),
    ]
    feedback_df.to_csv(FEEDBACK_PATH, index=False)

def append_feedback_batch(feedback_rows: List[Dict]) -> None:
    if not feedback_rows:
        return
    feedback_df = load_feedback_raw()
    columns = ["user_id", "matched_user_id", "action", "timestamp"]
    new_df = pd.DataFrame(feedback_rows, columns=columns)
    feedback_df = pd.concat([feedback_df, new_df], ignore_index=True)
    feedback_df.to_csv(FEEDBACK_PATH, index=False)
