# genai/__init__.py
from genai.groq_client import generate_sql, generate_insight

__all__ = ["generate_sql", "generate_insight"]