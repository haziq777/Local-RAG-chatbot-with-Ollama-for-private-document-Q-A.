"""Thin Ollama client plus the three LLM jobs: query rewrite, answer, ingest repair."""
import json
import os
import re

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# Contact details taken from the price list (front page): Perth store.
CONTACT = (
    "I'm not sure what you're after, so it's best to ask the team directly.\n\n"
    "**HR Products Perth** · 207 Bannister Rd, Canning Vale WA 6155  \n"
    "📞 **08 9484 6500** (freecall 1 800 998 037) · ✉️ hrsales@hrproducts.com.au"
)

PRICE_NOTE = (
    "ℹ️ This is just the list price. To find out about any additional discounts, please contact "
    "**HR Products Perth** · 207 Bannister Rd, Canning Vale WA 6155 · "
    "📞 **08 9484 6500** (freecall 1 800 998 037) · ✉️ hrsales@hrproducts.com.au"
)

PRICELIST_URL = "https://www.hrproducts.com.au/product-category/price-list/"
PRICELIST_REPLY = f"You can view and download the full HR Products price list here: {PRICELIST_URL}"

PRICELIST_FILLER = {
    "price", "list", "pricelist", "prices", "pdf", "full", "whole", "entire", "complete",
    "latest", "current", "updated", "new", "hr", "products", "2026", "2027", "share", "send", "give", "show", "get",
    "download", "view", "see", "find", "access", "link", "copy", "file", "me", "us", "the", "a", "an", "your", "our",
    "please", "pls", "can", "could", "would", "you", "i", "we", "do", "have", "need", "want", "where", "how", "is",
    "there", "to", "for", "of", "and", "on", "online", "website", "site", "hi", "hello", "hey", "thanks", "thank",
}


def is_pricelist_request(text):
    """True only when the message explicitly asks for the "price list" itself ('send me the price list'), not a product."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words or len(words) > 14 or not re.search(r"price[\s-]*lists?", text.lower()):
        return False
    return all(w in PRICELIST_FILLER for w in words)


CLARIFY = (
    "Please specify what you are after, for example a product name, size, brand or supplier code "
    "(like *20mm dura cap* or *447-007*)."
)

GREETING_WORDS = {"hi", "hii", "hiya", "hello", "hey", "heya", "yo", "gday", "howdy", "greetings", "morning",
                  "afternoon", "evening", "sup", "good", "day", "how", "hows", "whats"}
GREETING_FILLER = {"there", "you", "u", "all", "team", "everyone", "mate", "guys", "are", "how", "doing", "today",
                   "going", "it", "hows", "whats", "up", "again", "thanks", "thank", "hr", "products", "a", "the"}


def is_greeting(text):
    """True for short messages made only of greeting words and filler ('hi', 'good morning, how are you?')."""
    words = re.findall(r"[a-z']+", text.lower().replace("'", ""))
    if not words or len(words) > 8 or re.search(r"\d", text):
        return False
    return words[0] in GREETING_WORDS and all(w in GREETING_WORDS | GREETING_FILLER for w in words)


def greeting_reply(text):
    t = text.lower()
    hello = "Hello!"
    m = re.search(r"good\s+(morning|afternoon|evening|day)", t)
    if m:
        hello = f"Good {m.group(1)}!"
    if re.search(r"how\s+are\s+(you|u)|how'?s\s+it\s+going", t):
        hello += " I'm doing well, thanks for asking."
    return f"{hello} 👋 {CLARIFY}"


REWRITE_SYSTEM = (
    "You turn a user's message about a plumbing/irrigation price list into search parameters. "
    "Fix typos, keep sizes (e.g. 20mm), model numbers and codes exactly. Reply with JSON only: "
    '{"keywords": [lowercase search words, no filler], "code": string or null, '
    '"max_price": number or null, "intent": "lookup" | "cheapest" | "compare" | "list" | "greeting" | "unclear"}. Use "greeting" for hello/hi/good morning/how are you style messages with no product request. Use "unclear" for anything out of scope or vague: questions not about finding a product or price in this price list (general knowledge, coding, jokes, opinions, "list all possible greetings", requests about the assistant itself) and messages that name no product, size, brand or code.'
)

ANSWER_SYSTEM = (
    "You answer questions about a price list using ONLY the rows provided. Each row is "
    "'code | description | group | list price | printed page'. Rules: cite the supplier code for every item you "
    "mention; quote prices exactly as given; never invent codes, prices or products; if no row fits the "
    "question, or the question is not about this price list, say it is not in the price list; for 'cheapest' or 'compare' questions work out the answer "
    "from the rows' prices. Do NOT list the rows: a results table is shown to the user right below your answer. Reply in one or two short sentences that name at most the best 1-3 items (code and price) and, if relevant, say there are more in the table."
)

REPAIR_SYSTEM = (
    "You are given raw text lines from a price-list PDF page. One line is a product row with a supplier code "
    "and price but no description; its description is wrapped onto the lines just above and/or below it. "
    'Reply with JSON only: {"description": string}. Copy the description text from the lines, joining wrapped '
    "lines in order. Do not include other products' text, the code or the price."
)


def available():
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5).json()
        names = {m["name"] for m in tags.get("models", [])}
        return MODEL in names or f"{MODEL}:latest" in names
    except Exception:
        return False


def _chat(messages, fmt=None, stream=False, timeout=120):
    body = {"model": MODEL, "messages": messages, "stream": stream, "options": {"temperature": 0}}
    if fmt:
        body["format"] = fmt
    return requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=timeout, stream=stream)


def _json(messages, timeout=60):
    r = _chat(messages, fmt="json", timeout=timeout)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def rewrite_query(query):
    """Return {keywords, code, max_price, intent} or None if the LLM is unavailable/fails."""
    try:
        out = _json([{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": query}], timeout=30)
        kws = [str(k).lower() for k in out.get("keywords") or [] if str(k).strip()]
        price = out.get("max_price")
        return {
            "keywords": kws,
            "code": out.get("code") or None,
            "max_price": float(price) if isinstance(price, (int, float)) else None,
            "intent": out.get("intent") if out.get("intent") in ("lookup", "cheapest", "compare", "list", "greeting", "unclear") else "lookup",
        }
    except Exception:
        return None


def rows_to_text(rows):
    return "\n".join(
        f'{r["supplier_code"]} | {r["description"]} | {r["subheading"]} | {r["price"]:.2f} | page {r["page"]}' for r in rows
    )


def answer_stream(query, rows, history=()):
    """Yield answer text chunks. `history` is a list of (role, text) for recent turns."""
    msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
    msgs += [{"role": r, "content": t} for r, t in history]
    msgs.append({"role": "user", "content": f"Rows:\n{rows_to_text(rows)}\n\nQuestion: {query}"})
    with _chat(msgs, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("message", {}).get("content", "")


def repair_description(context_lines, code):
    """Ask the model for the wrapped description of the row with `code`; None if it fails."""
    try:
        text = "\n".join(context_lines)
        out = _json([
            {"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user", "content": f"Product row code: {code}\nLines:\n{text}"},
        ])
        desc = re.sub(r"\s+", " ", str(out.get("description", ""))).strip()
        # only accept text that really comes from the page, and not the code itself
        if desc and len(desc) < 300 and code not in desc and desc.lower() in re.sub(r"\s+", " ", text).lower():
            return desc
    except Exception:
        pass
    return None
