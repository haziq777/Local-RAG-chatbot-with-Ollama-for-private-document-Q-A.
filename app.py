import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import llm
from search import search_all

TOP = 10
LLM_ROWS = 10  # rows the model sees; the answer itself stays short, the table carries the list

st.set_page_config(page_title="Price list chat", page_icon="💬")
st.title("HR Price List 2026-27")
st.caption("Ask for a product by description or code. Results show description, supplier code and list price.")


@st.cache_data(ttl=10)
def llm_ready():
    return llm.available()


use_llm = llm_ready()
if not use_llm:
    st.info(f"Ollama model `{llm.MODEL}` not available, so this is plain search only. "
            f"Start Ollama and run `ollama pull {llm.MODEL}` to enable chat answers.")


def table(rows):
    return pd.DataFrame(
        {
            "Supplier code": [h["supplier_code"] for h in rows],
            "List price": [f'{h["price"]:.2f}' for h in rows],
            "Group": [h["subheading"] for h in rows],
            "Description": [h["description"] for h in rows],
            "Section": [h["section"] for h in rows],
            "Page": [h["page"] for h in rows],
            "PDF page": [h["pdf_page"] for h in rows],
        }
    )


def scroll_to_last_question():
    """Scroll the page so the question just asked is in view; the streamed reply then fills in below it."""
    components.html(
        """<script>
        setTimeout(() => {
          const doc = window.parent.document;
          const avatars = doc.querySelectorAll('[data-testid="stChatMessageAvatarUser"]');
          const msgs = doc.querySelectorAll('[data-testid="stChatMessage"]');
          const last = avatars.length ? avatars[avatars.length - 1].closest('[data-testid="stChatMessage"]')
                                      : msgs[msgs.length - 1];
          if (last) last.scrollIntoView({behavior: "smooth", block: "start"});
          // hide the Stop button the moment it is clicked (once per page)
          if (!window.parent.__stopHider) {
            window.parent.__stopHider = true;
            doc.addEventListener("click", (e) => {
              const btn = e.target.closest(".st-key-stop_reply");
              if (btn) btn.style.display = "none";
            }, true);
          }
        }, 150);
        </script>""",
        height=0,
    )


def show(entry):
    if entry.get("text"):
        st.write(entry["text"])
    if entry.get("shown") is not None:
        st.caption(entry["caption"])
        st.dataframe(entry["shown"], hide_index=True, use_container_width=True)
    if entry.get("all") is not None:
        with st.expander(f'Show all {len(entry["all"])} matches'):
            st.dataframe(entry["all"], hide_index=True, use_container_width=True)
    if entry.get("shown") is not None:
        st.caption(llm.PRICE_NOTE)


MAX_CLARIFY = 2  # how many times we ask the user to be specific before sharing the store contact

if "unclear_count" not in st.session_state:
    st.session_state.unclear_count = 0


def unclear_reply():
    """Unclear or out-of-scope: ask for specifics up to MAX_CLARIFY times, then share the Perth store contact."""
    st.session_state.unclear_count += 1
    if st.session_state.unclear_count > MAX_CLARIFY:
        st.session_state.unclear_count = 0
        return llm.CONTACT
    return llm.CLARIFY


if "history" not in st.session_state:
    st.session_state.history = []  # (role, entry dict)


def _request_stop():
    st.session_state.stopped = True


# Clicking Stop interrupts the running reply (Streamlit reruns the script); nothing is shown for it.
st.session_state.pop("stopped", None)

for role, entry in st.session_state.history:
    with st.chat_message(role):
        show(entry)

if query := st.chat_input("e.g. cheapest 20mm dura cap, blue primary nozzle GT885, 315300"):
    st.session_state.history.append(("user", {"text": query}))
    with st.chat_message("user"):
        st.write(query)
    scroll_to_last_question()

    with st.chat_message("assistant"):
        stop_slot = st.empty()
        if use_llm:
            stop_slot.button("⏹ Stop", key="stop_reply", on_click=_request_stop)
        greeting = llm.is_greeting(query)
        pricelist = llm.is_pricelist_request(query)
        plan = llm.rewrite_query(query) if use_llm and not greeting and not pricelist else None
        greeting = greeting or bool(plan and plan["intent"] == "greeting")
        hits = [] if greeting or pricelist else search_all(
            query,
            keywords=plan["keywords"] if plan else None,
            max_price=plan["max_price"] if plan else None,
        )
        entry = {}
        if plan and plan["intent"] == "unclear":
            hits = []
        if greeting:
            st.session_state.unclear_count = 0  # a greeting is not an unclear question
            entry["text"] = llm.greeting_reply(query)
        elif pricelist:
            st.session_state.unclear_count = 0
            entry["text"] = llm.PRICELIST_REPLY
        elif not hits:
            entry["text"] = unclear_reply()
        else:
            if use_llm:
                rows = hits[:LLM_ROWS]
                if plan and plan["intent"] == "cheapest":  # cheapest among the best matches, not just the top few
                    rows = sorted(hits[:60], key=lambda h: h["price"])[:LLM_ROWS]
                recent = [(r, e["text"]) for r, e in st.session_state.history[:-1][-6:] if e.get("text")]
                try:
                    text = st.write_stream(llm.answer_stream(query, rows, recent))
                    if "not in the price list" in text.lower():
                        extra = unclear_reply()
                        st.write(extra)
                        text += f"\n\n{extra}"
                    else:
                        st.session_state.unclear_count = 0
                    entry["text"] = text
                except Exception as e:
                    st.warning(f"LLM answer failed ({type(e).__name__}); showing search results only.")
            if not use_llm:
                st.session_state.unclear_count = 0
            entry["caption"] = f"Top {min(TOP, len(hits))} of {len(hits)} matches, most relevant first"
            entry["shown"] = table(hits[:TOP])
            if len(hits) > TOP:
                entry["all"] = table(hits)
            # text was already streamed above; render only the tables here
            show({k: v for k, v in entry.items() if k != "text"})
        stop_slot.empty()
        if not hits:
            st.write(entry["text"])
        st.session_state.history.append(("assistant", entry))
