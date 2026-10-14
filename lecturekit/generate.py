"""Notes (full + short) and quiz (Bloom levels) from the prepared lecture context."""
from .llm import LLM

NOTES_SYSTEM = """You are an academic note-taking assistant. From the lecture material
given, write study notes. Use ONLY the given material; do not add outside facts.
Reply ONLY with JSON:
{"title": str,
 "full_notes": [{"heading": str, "summary": str, "key_points": [str]}],
 "short_notes": [str]}
full_notes: 3-6 topics in the order taught, each with a 2-4 sentence summary and key
points (definitions, formulas, code ideas as stated). short_notes: 6-10 one-line
revision bullets."""

QUIZ_SYSTEM = """You write assessment questions for a university lecture, using ONLY the
given material. Reply ONLY with JSON:
{"questions": [{"question": str, "options": [4 strings], "answer_index": 0-3,
 "bloom_level": "remember"|"understand"|"apply"|"analyze", "explanation": str}]}
Write the requested number of multiple-choice questions with exactly one correct
option, mixing Bloom levels roughly: 20% remember, 30% understand, 30% apply,
20% analyze. Each question must be answerable from the material."""


def make_notes(llm: LLM, context: str) -> dict:
    return llm.complete_json(NOTES_SYSTEM, context, tag="notes")


def make_quiz(llm: LLM, context: str, n: int = 10) -> list[dict]:
    reply = llm.complete_json(
        QUIZ_SYSTEM, f"Write EXACTLY {n} questions (no more, no fewer).\n\n{context}", tag="quiz")
    return reply.get("questions", [])[:n]
