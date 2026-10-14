"""Off-topic filtering (configuration C): the LLM labels each speech segment."""
from .llm import LLM

SYSTEM = """You label segments of a university lecture transcript.
A segment is OFF-TOPIC if it does not teach the subject: class announcements,
deadlines, exam or grading logistics, jokes or small talk, room or equipment
problems, requests to the audience, and personal anecdotes unrelated to the content.
Short explanatory asides that clarify the subject are ON-TOPIC.
Reply ONLY with JSON: {"off_topic": [ids of off-topic segments]}"""


def find_off_topic(llm: LLM, segments: list[dict], slide_titles: list[str],
                   batch: int = 60) -> set[int]:
    off: set[int] = set()
    context = "Slide titles (context): " + " | ".join(t for t in slide_titles if t)[:800]
    for i in range(0, len(segments), batch):
        chunk = segments[i:i + batch]
        lines = "\n".join(f"[{g['id']}] {g['text']}" for g in chunk)
        reply = llm.complete_json(SYSTEM, f"{context}\n\nSegments:\n{lines}", tag="filter")
        valid = {g["id"] for g in chunk}
        off |= {int(x) for x in reply.get("off_topic", []) if int(x) in valid}
    return off
