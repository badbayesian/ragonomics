"""Streamlit UI for interactive RAG over papers with citations and DOI network. Uses main pipeline functions for retrieval and answers."""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from dataclasses import replace
import hashlib
import html
import re
from typing import List, Optional

import streamlit as st
from openai import OpenAI, BadRequestError

from ragonometrics.core.main import (
    Settings,
    Paper,
    build_and_store_doi_network,
    build_doi_network_from_paper,
    embed_texts,
    load_papers,
    load_settings,
    prepare_chunks_for_paper,
    top_k_context,
)
from ragonometrics.integrations.openalex import format_openalex_context
from ragonometrics.integrations.citec import format_citec_context
from ragonometrics.pipeline import call_openai
from ragonometrics.core.prompts import RESEARCHER_QA_PROMPT
from ragonometrics.pipeline.query_cache import DEFAULT_CACHE_PATH, get_cached_answer, make_cache_key, set_cached_answer
from ragonometrics.pipeline.token_usage import DEFAULT_USAGE_DB, get_recent_usage, get_usage_by_model, get_usage_summary

import networkx as nx
import plotly.graph_objects as go

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    import pytesseract
    from PIL import ImageDraw
except Exception:
    pytesseract = None
    ImageDraw = None


st.set_page_config(page_title="Ragonometrics Chat", layout="wide")


def list_papers(papers_dir: Path) -> List[Path]:
    """List PDF files in the provided directory.

    Args:
        papers_dir: Directory containing PDF files.

    Returns:
        List[Path]: Sorted list of PDF paths.
    """
    if not papers_dir.exists():
        return []
    return sorted(papers_dir.glob("*.pdf"))


@st.cache_data
def load_and_prepare(path: Path, settings: Settings):
    """Load a paper, prepare chunks/embeddings, and cache the result.

    A client is constructed internally so caching depends on `path` and `settings`.

    Args:
        path: PDF path to load.
        settings: Runtime settings.

    Returns:
        tuple[Paper, list[dict], list[list[float]]]: Paper, chunks, embeddings.
    """
    papers = load_papers([path])
    paper = papers[0]
    chunks = prepare_chunks_for_paper(paper, settings)
    if not chunks:
        return paper, [], []
    client = OpenAI()
    chunk_texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
    embeddings = embed_texts(client, chunk_texts, settings.embedding_model, settings.batch_size)
    return paper, chunks, embeddings


def parse_context_chunks(context: str) -> List[dict]:
    """Parse concatenated context into structured chunks.

    Args:
        context: Context string with optional provenance lines.

    Returns:
        List[dict]: Dicts with "meta", "text", and optional "page".
    """
    chunks: List[dict] = []
    for block in context.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        meta = None
        text = block
        page: Optional[int] = None
        if block.startswith("(page "):
            parts = block.split("\n", 1)
            meta = parts[0].strip()
            text = parts[1].strip() if len(parts) > 1 else ""
            m = re.search(r"\(page\s+(\d+)\b", meta)
            if m:
                try:
                    page = int(m.group(1))
                except ValueError:
                    page = None
        chunks.append({"meta": meta, "text": text, "page": page})
    return chunks


def extract_highlight_terms(query: str, max_terms: int = 6) -> List[str]:
    """Extract key terms from a query for highlighting."""
    stop = {
        "the", "and", "or", "but", "a", "an", "of", "to", "in", "for", "on", "with",
        "is", "are", "was", "were", "be", "been", "it", "this", "that", "these",
        "those", "as", "at", "by", "from", "about", "into", "over", "after", "before",
        "what", "which", "who", "whom", "why", "how", "when", "where",
    }
    tokens = re.findall(r"[A-Za-z0-9]{3,}", query.lower())
    terms = []
    for tok in tokens:
        if tok in stop:
            continue
        if tok not in terms:
            terms.append(tok)
        if len(terms) >= max_terms:
            break
    return terms


def highlight_text_html(text: str, terms: List[str]) -> str:
    """Return HTML with highlight marks for matching terms."""
    if not terms:
        return html.escape(text)
    escaped = html.escape(text)
    for term in terms:
        pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
        escaped = pattern.sub(r"<mark>\1</mark>", escaped)
    return escaped


def highlight_image_terms(image, terms: List[str]):
    """Highlight matched terms on a PIL image using OCR."""
    if not terms or not pytesseract or not ImageDraw:
        return image
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return image
    if not data or "text" not in data:
        return image

    img = image.convert("RGBA")
    overlay = ImageDraw.Draw(img, "RGBA")
    terms_lower = {t.lower() for t in terms}
    for i, word in enumerate(data["text"]):
        if not word:
            continue
        w = word.strip().lower()
        if w in terms_lower:
            x = data["left"][i]
            y = data["top"][i]
            w_box = data["width"][i]
            h_box = data["height"][i]
            overlay.rectangle([x, y, x + w_box, y + h_box], fill=(255, 235, 59, 120), outline=(255, 193, 7, 200))
    return img


def render_citation_snapshot(path: Path, citation: dict, key_prefix: str, query: str) -> None:
    """Render a highlighted text snapshot and optional page image for a citation chunk."""
    meta = citation.get("meta") or "Context chunk"
    text = citation.get("text") or ""
    page = citation.get("page")
    terms = extract_highlight_terms(query)

    st.markdown(f"**{meta}**")
    if text:
        snippet = text if len(text) <= 1200 else text[:1200] + "..."
        highlighted = highlight_text_html(snippet, terms)
        st.markdown(
            f"<div style='font-family: monospace; white-space: pre-wrap;'>{highlighted}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No text available for this chunk.")

    if page and convert_from_path:
        show_key = f"{key_prefix}_show_page_{page}"
        if st.checkbox(f"Show page {page} snapshot", key=show_key):
            try:
                images = convert_from_path(str(path), first_page=page, last_page=page)
                if images:
                    img = highlight_image_terms(images[0], terms)
                    st.image(img, caption=f"Page {page}")
            except Exception as exc:
                st.warning(f"Failed to render page {page}: {exc}")
    elif page:
        st.caption(f"Page {page} (snapshot requires pdf2image + poppler)")


def auth_gate() -> None:
    """Simple username/password gate for the Streamlit app."""
    expected_user = os.getenv("STREAMLIT_USERNAME")
    expected_pass = os.getenv("STREAMLIT_PASSWORD")

    if not expected_user or not expected_pass:
        st.sidebar.info("Login disabled (set STREAMLIT_USERNAME/STREAMLIT_PASSWORD to enable).")
        return

    if st.session_state.get("authenticated"):
        return

    st.sidebar.subheader("Login")
    with st.sidebar.form("login_form"):
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if user == expected_user and password == expected_pass:
            st.session_state.authenticated = True
            st.sidebar.success("Logged in.")
            return
        st.sidebar.error("Invalid credentials.")

    st.stop()


def main():
    """Run the Streamlit app."""
    st.title("Ragonometrics — Paper Chatbot")

    settings = load_settings()
    auth_gate()
    client = OpenAI()

    st.sidebar.header("Settings")
    papers_dir = st.sidebar.text_input("Papers directory", str(settings.papers_dir))
    papers_dir = Path(papers_dir)
    top_k = st.sidebar.number_input(
        "Top K context chunks",
        value=int(settings.top_k),
        min_value=1,
        max_value=30,
        step=1,
    )
    model_options = [settings.chat_model]
    extra_models = [m.strip() for m in os.getenv("LLM_MODELS", "").split(",") if m.strip()]
    for m in extra_models:
        if m not in model_options:
            model_options.append(m)
    selected_model = st.sidebar.selectbox("LLM model", options=model_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Make sure `pdftotext`/`pdfinfo` are installed and `OPENAI_API_KEY` is set.")

    files = list_papers(papers_dir)

    if not files:
        st.warning(f"No PDF files found in {papers_dir}")
        return

    file_choice = st.selectbox("Select a paper", options=[p.name for p in files])
    selected_path = next(p for p in files if p.name == file_choice)

    with st.spinner("Loading and preparing paper..."):
        paper, chunks, chunk_embeddings = load_and_prepare(selected_path, settings)

    st.subheader(paper.title)
    st.caption(f"Author: {paper.author} — {paper.path.name}")

    openalex_context = format_openalex_context(paper.openalex)
    citec_context = format_citec_context(paper.citec)
    if openalex_context or citec_context:
        with st.expander("External Metadata", expanded=False):
            if openalex_context:
                st.markdown("**OpenAlex**")
                st.code(openalex_context, language="text")
            if citec_context:
                st.markdown("**CitEc**")
                st.code(citec_context, language="text")

    if not chunks:
        st.info("No text could be extracted from this PDF.")
        return

    if "history" not in st.session_state:
        st.session_state.history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex
        st.session_state.session_started_at = datetime.now(timezone.utc).isoformat()
    if "last_request_id" not in st.session_state:
        st.session_state.last_request_id = None

    if st.sidebar.button("Clear chat history"):
        st.session_state.history = []

    retrieval_settings = settings
    if int(top_k) != settings.top_k:
        retrieval_settings = replace(settings, top_k=int(top_k))

    tab_chat, tab_doi, tab_usage = st.tabs(["Chat", "DOI Network", "Usage"])

    with tab_chat:
        query = st.text_input("Ask a question about this paper", key="query_input")
        send_clicked = st.button("Send")
        vary_clicked = st.button(
            "Try Variation",
            help="Rerun with higher temperature for a slightly different answer.",
        )

        if (send_clicked or vary_clicked) and query:
            request_id = uuid4().hex
            st.session_state.last_request_id = request_id
            with st.spinner("Retrieving context and querying model..."):
                context = top_k_context(
                    chunks,
                    chunk_embeddings,
                    query=query,
                    client=client,
                    settings=retrieval_settings,
                    session_id=st.session_state.session_id,
                    request_id=request_id,
                )

                temperature = None
                cache_allowed = True
                if vary_clicked:
                    cache_allowed = False
                    try:
                        temperature = float(os.getenv("RAG_VARIATION_TEMPERATURE", "0.7"))
                    except Exception:
                        temperature = 0.7

                cached = None
                cache_key = None
                if cache_allowed:
                    cache_key = make_cache_key(query, str(paper.path), selected_model, context)
                    cached = get_cached_answer(DEFAULT_CACHE_PATH, cache_key)

                if cached is not None:
                    answer = cached
                else:
                    openalex_context = format_openalex_context(paper.openalex)
                    citec_context = format_citec_context(paper.citec)
                    user_input = f"Context:\n{context}\n\nQuestion: {query}"
                    prefix_parts = [ctx for ctx in (openalex_context, citec_context) if ctx]
                    if prefix_parts:
                        user_input = f"{'\n\n'.join(prefix_parts)}\n\n{user_input}"
                    try:
                        answer = call_openai(
                            client,
                            model=selected_model,
                            instructions=RESEARCHER_QA_PROMPT,
                            user_input=user_input,
                            max_output_tokens=None,
                            temperature=temperature,
                            usage_context="answer",
                            session_id=st.session_state.session_id,
                            request_id=request_id,
                        ).strip()
                    except BadRequestError as exc:
                        err = str(exc).lower()
                        if temperature is not None and "temperature" in err and "unsupported" in err:
                            st.warning(
                                "The selected model does not support temperature. "
                                "Retrying without variation."
                            )
                            answer = call_openai(
                                client,
                                model=selected_model,
                                instructions=RESEARCHER_QA_PROMPT,
                                user_input=user_input,
                                max_output_tokens=None,
                                temperature=None,
                                usage_context="answer",
                                session_id=st.session_state.session_id,
                                request_id=request_id,
                            ).strip()
                        else:
                            raise
                    if cache_allowed and cache_key is not None:
                        set_cached_answer(
                            DEFAULT_CACHE_PATH,
                            cache_key=cache_key,
                            query=query,
                            paper_path=str(paper.path),
                            model=selected_model,
                            context=context,
                            answer=answer,
                        )

                citations = parse_context_chunks(context)
                st.session_state.history.append(
                    {
                        "query": query,
                        "answer": answer,
                        "context": context,
                        "citations": citations,
                        "paper_path": str(paper.path),
                        "request_id": request_id,
                    }
                )

        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history), start=1):
                q = None
                a = None
                citations: List[dict] = []
                citation_path = paper.path
                request_id = None
                if isinstance(item, tuple):
                    q, a = item
                else:
                    q = item.get("query")
                    a = item.get("answer")
                    context = item.get("context")
                    citations = item.get("citations")
                    item_paper_path = item.get("paper_path")
                    request_id = item.get("request_id")
                    if context:
                        citations = parse_context_chunks(context)
                    elif citations is None:
                        citations = []
                    if item_paper_path:
                        citation_path = Path(item_paper_path)
                    else:
                        citation_path = paper.path
                history_id = request_id
                if not history_id:
                    token = f"{q or ''}|{a or ''}"
                    history_id = hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]

                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")

                if citations:
                    st.markdown("**Citations & Snapshots**")
                    st.caption(
                        f"Showing {len(citations)} chunks (top_k={retrieval_settings.top_k}, total_chunks={len(chunks)})"
                    )
                    tab_labels = []
                    for c_idx, c in enumerate(citations, start=1):
                        page = c.get("page")
                        suffix = f" (p{page})" if page else ""
                        tab_labels.append(f"Citation {c_idx}{suffix}")
                    tabs = st.tabs(tab_labels)
                    for c_idx, (tab, c) in enumerate(zip(tabs, citations), start=1):
                        with tab:
                            key_prefix = f"citation_{history_id}_{c_idx}"
                            render_citation_snapshot(citation_path, c, key_prefix=key_prefix, query=q or "")
                st.markdown("---")

    with tab_doi:
        st.subheader("DOI Network")
        st.markdown("Build and visualize the DOI citation network extracted from the selected paper.")

        def visualize_network(network: dict) -> None:
            """Render a DOI citation network using Plotly.

            Args:
                network: Mapping of source DOI to cited DOIs.
            """
            G = nx.DiGraph()
            for src, targets in network.items():
                G.add_node(src)
                for tgt in targets:
                    G.add_node(tgt)
                    G.add_edge(src, tgt)

            if len(G) == 0:
                st.info("No DOIs or citation edges found for this paper.")
                return

            pos = nx.spring_layout(G, seed=42)

            edge_x = []
            edge_y = []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1, color="#888"),
                hoverinfo="none",
                mode="lines",
            )

            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                hoverinfo="text",
                textposition="top center",
                marker=dict(
                    showscale=False,
                    color="#6175c1",
                    size=10,
                    line_width=1,
                ),
                text=[n if len(n) <= 30 else n[:27] + "..." for n in node_text],
                hovertext=node_text,
            )

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                showlegend=False,
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )

            st.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns([1, 3])
        with col1:
            build_network = st.button("Build DOI Network")
            store_db = st.checkbox("Store network to DB (Postgres)", value=False)

        if build_network:
            with st.spinner("Building DOI network (may make web requests)..."):
                if store_db:
                    db_url = os.environ.get("DATABASE_URL")
                    network = build_and_store_doi_network(paper, db_url=db_url)
                else:
                    network = build_doi_network_from_paper(paper)
            visualize_network(network)

    with tab_usage:
        st.subheader("Token Usage")
        st.caption("Aggregates are computed from the local SQLite usage table.")

        now = datetime.now(timezone.utc)
        last_24h = (now - timedelta(hours=24)).isoformat()

        total = get_usage_summary(db_path=DEFAULT_USAGE_DB)
        session_total = get_usage_summary(db_path=DEFAULT_USAGE_DB, session_id=st.session_state.session_id)
        recent_total = get_usage_summary(db_path=DEFAULT_USAGE_DB, since=last_24h)

        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Total Tokens (All Time)", f"{total.total_tokens}")
        metrics_cols[1].metric("Total Tokens (Session)", f"{session_total.total_tokens}")
        metrics_cols[2].metric("Total Tokens (24h)", f"{recent_total.total_tokens}")
        metrics_cols[3].metric("Calls (All Time)", f"{total.calls}")

        if st.session_state.last_request_id:
            last_query = get_usage_summary(
                db_path=DEFAULT_USAGE_DB,
                request_id=st.session_state.last_request_id,
            )
            st.metric("Last Query Tokens", f"{last_query.total_tokens}")

        st.markdown("---")
        st.subheader("Usage By Model")
        by_model = get_usage_by_model(db_path=DEFAULT_USAGE_DB)
        if by_model:
            st.dataframe(by_model, width="stretch")
        else:
            st.info("No usage records yet.")

        st.markdown("---")
        st.subheader("Recent Usage Records")
        recent = get_recent_usage(db_path=DEFAULT_USAGE_DB, limit=200)
        if recent:
            st.dataframe(recent, width="stretch")
        else:
            st.info("No usage records yet.")


if __name__ == "__main__":
    main()
