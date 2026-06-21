from pathlib import Path

KB = Path("genai/knowledge_base")


def load_text(filename):

    with open(
        KB / filename,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


def build_sql_system_prompt():

    schema = load_text(
        "olist_schema.txt"
    )

    rules = load_text(
        "sql_rules.txt"
    )

    glossary = load_text(
        "glossary.txt"
    )

    business = load_text(
        "business_rules.txt"
    )

    return f"""
You are an expert MotherDuck SQL analyst.

SCHEMA:
{schema}

BUSINESS RULES:
{business}

SQL RULES:
{rules}

GLOSSARY:
{glossary}

Return ONLY valid DuckDB SQL.
"""

def build_insight_system_prompt():

    return """
You are a Senior Business Intelligence Analyst.

Rules:
- Analyze ONLY from provided data.
- Do not repeat the table.
- Focus on trends.
- Focus on rankings.
- Explain business implications.
- Give actionable recommendations.
- Write in Vietnamese.
"""