"""
Responsible-AI guardrails applied to every chatbot answer.

- redact_pii: removes personal email addresses and phone numbers. The corpus includes ~57 staff and faculty profile
  pages with personal @uchicago.edu addresses; the program's shared office mailboxes stay, so the bot can still tell
  applicants whom to contact.
- find_unsupported_claims: a second, cheap LLM call that lists claims in the answer the retrieved passages don't
  support. The app shows them as a warning under the answer instead of blocking it, since the checker can be wrong too.
"""
import json
import re

from openai import OpenAI

# Shared program mailboxes, published for applicants and students to contact.
ALLOWED_EMAILS = {
    "applieddatascience-admissions@uchicago.edu",
    "applieddatascience-advising@uchicago.edu",
    "applieddatascience-advisingonline@uchicago.edu",
}
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")

GROUNDEDNESS_MODEL = "gpt-4o-mini"
GROUNDEDNESS_PROMPT = """You check a chatbot's answer against the source passages it was given.

Question: {question}

Source passages:
{context}

Answer:
{answer}

List every factual claim in the answer (numbers, dates, names, requirements, policies) that the source passages do not
support. Ignore the 'Sources:' line, greetings, and suggestions to contact the program or visit a page. A claim that
restates or reasonably summarizes the passages is supported.

Return JSON only: {{"unsupported": ["<claim>", ...]}} with an empty list when everything is supported."""


def redact_pii(text: str) -> str:
    """Replace personal email addresses and phone numbers; keep the program's shared mailboxes."""
    text = EMAIL_RE.sub(lambda m: m[0] if m[0].lower() in ALLOWED_EMAILS else "[personal email removed]", text)
    return PHONE_RE.sub("[phone number removed]", text)


def find_unsupported_claims(question: str, answer: str, context: str, client: OpenAI | None = None) -> list[str] | None:
    """Claims in `answer` that `context` doesn't support; [] if all are supported, None if the check itself failed."""
    client = client or OpenAI()
    try:
        resp = client.chat.completions.create(
            model=GROUNDEDNESS_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": GROUNDEDNESS_PROMPT.format(
                question=question, context=context, answer=answer)}],
            response_format={"type": "json_object"},
        )
        claims = json.loads(resp.choices[0].message.content)["unsupported"]
        return [str(c) for c in claims if str(c).strip()]
    except Exception:  # a failed check must never block the answer
        return None
