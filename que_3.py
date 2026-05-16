import streamlit as st
import pandas as pd
import io
import os
import json
from collections import defaultdict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NER Extractor",
    page_icon="🔍",
    layout="wide",
)

# ── Styling ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }

    h1 { color: #e2e8f0; font-size: 2rem; font-weight: 700; }
    h3 { color: #94a3b8; font-size: 1rem; font-weight: 400; margin-top: -0.5rem; }
    label, .stMarkdown p { color: #cbd5e1; }

    .entity-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .entity-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .entity-text { color: #f1f5f9; font-size: 0.95rem; }

    /* Tag colours */
    .tag-PERSON      { background:#7c3aed; color:#ede9fe; }
    .tag-ORG         { background:#0369a1; color:#e0f2fe; }
    .tag-GPE, .tag-LOC { background:#047857; color:#d1fae5; }
    .tag-DATE,.tag-TIME { background:#b45309; color:#fef3c7; }
    .tag-MONEY,.tag-PERCENT { background:#be185d; color:#fce7f3; }
    .tag-PRODUCT,.tag-WORK_OF_ART { background:#7e22ce; color:#f3e8ff; }
    .tag-EVENT       { background:#9f1239; color:#ffe4e6; }
    .tag-LAW         { background:#1d4ed8; color:#dbeafe; }
    .tag-LANGUAGE    { background:#065f46; color:#d1fae5; }
    .tag-NORP        { background:#92400e; color:#fef3c7; }
    .tag-FAC         { background:#0c4a6e; color:#e0f2fe; }
    .tag-QUANTITY,.tag-ORDINAL,.tag-CARDINAL { background:#374151; color:#e5e7eb; }
    .tag-DEFAULT     { background:#334155; color:#e2e8f0; }

    .summary-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .metric-num { font-size: 2rem; font-weight: 700; color: #818cf8; }
    .metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }

    .stButton>button {
        background: #4f46e5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton>button:hover { background: #4338ca; }

    .stFileUploader { border-radius: 10px; }
    div[data-testid="stExpander"] { background: #1e293b; border: 1px solid #334155; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers: text extraction ────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    except Exception as e:
        return f"[PDF extraction error: {e}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX extraction error: {e}]"


def extract_text_from_csv(file_bytes: bytes) -> str:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        # Concatenate all string-like columns
        parts = []
        for col in df.columns:
            parts.append(f"Column: {col}")
            vals = df[col].dropna().astype(str).tolist()
            parts.extend(vals[:200])  # cap per column
        return "\n".join(parts)
    except Exception as e:
        return f"[CSV extraction error: {e}]"


def extract_text_from_txt(file_bytes: bytes) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc)
        except Exception:
            continue
    return "[Could not decode text file]"


def extract_text_from_excel(file_bytes: bytes) -> str:
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        parts = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            parts.append(f"Sheet: {sheet}")
            for col in df.columns:
                vals = df[col].dropna().astype(str).tolist()
                parts.append(f"Column: {col}")
                parts.extend(vals[:100])
        return "\n".join(parts)
    except Exception as e:
        return f"[Excel extraction error: {e}]"


def get_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    elif name.endswith(".docx"):
        return extract_text_from_docx(data)
    elif name.endswith(".csv"):
        return extract_text_from_csv(data)
    elif name.endswith((".xlsx", ".xls")):
        return extract_text_from_excel(data)
    else:  # .txt, .md, etc.
        return extract_text_from_txt(data)


# ── NER via spaCy ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_nlp():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            return spacy.load("en_core_web_sm")
    except Exception as e:
        return None


def run_ner(text: str, nlp) -> list[dict]:
    """Returns list of {text, label, start, end}."""
    # spaCy has a token limit; chunk if needed
    MAX_CHARS = 1_000_000
    text = text[:MAX_CHARS]
    doc = nlp(text)
    seen = set()
    entities = []
    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if key not in seen:
            seen.add(key)
            entities.append({"text": ent.text.strip(), "label": ent.label_, "count": 1})
        else:
            for e in entities:
                if e["text"] == ent.text.strip() and e["label"] == ent.label_:
                    e["count"] += 1
    return entities


# ── UI ─────────────────────────────────────────────────────────────────────────

st.markdown("# 🔍 NER Extractor")
st.markdown("### Upload any document — get Named Entities instantly")
st.markdown("---")

col_upload, col_options = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Drop your file here",
        type=["pdf", "docx", "csv", "txt", "xlsx", "xls", "md"],
        help="Supports PDF, Word (.docx), CSV, Excel, and plain text files",
    )

with col_options:
    st.markdown("**Filter by Entity Type**")
    ENTITY_TYPES = [
        "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "PERCENT",
        "PRODUCT", "WORK_OF_ART", "EVENT", "LAW", "LANGUAGE", "NORP",
        "FAC", "QUANTITY", "ORDINAL", "CARDINAL",
    ]
    filter_types = st.multiselect(
        "Show only these types (leave empty = show all)",
        options=ENTITY_TYPES,
        default=[],
        label_visibility="collapsed",
    )
    show_text_preview = st.checkbox("Show extracted text preview", value=False)
    sort_by = st.selectbox("Sort results by", ["Frequency (high→low)", "Entity type", "Alphabetical"])

TAG_COLOR_MAP = {
    "PERSON": "tag-PERSON", "ORG": "tag-ORG",
    "GPE": "tag-GPE", "LOC": "tag-LOC",
    "DATE": "tag-DATE", "TIME": "tag-TIME",
    "MONEY": "tag-MONEY", "PERCENT": "tag-PERCENT",
    "PRODUCT": "tag-PRODUCT", "WORK_OF_ART": "tag-WORK_OF_ART",
    "EVENT": "tag-EVENT", "LAW": "tag-LAW",
    "LANGUAGE": "tag-LANGUAGE", "NORP": "tag-NORP",
    "FAC": "tag-FAC", "QUANTITY": "tag-QUANTITY",
    "ORDINAL": "tag-ORDINAL", "CARDINAL": "tag-CARDINAL",
}

if uploaded:
    with st.spinner("📄 Reading file…"):
        raw_text = get_text(uploaded)

    if show_text_preview:
        with st.expander("📝 Extracted Text Preview (first 2000 chars)"):
            st.text(raw_text[:2000] + ("…" if len(raw_text) > 2000 else ""))

    nlp = load_nlp()
    if nlp is None:
        st.error("❌ Could not load spaCy. Make sure it's installed: `pip install spacy && python -m spacy download en_core_web_sm`")
        st.stop()

    with st.spinner("🧠 Running Named Entity Recognition…"):
        entities = run_ner(raw_text, nlp)

    if not entities:
        st.warning("No named entities found in the document.")
        st.stop()

    # Apply filter
    if filter_types:
        entities = [e for e in entities if e["label"] in filter_types]

    # Sort
    if sort_by == "Frequency (high→low)":
        entities = sorted(entities, key=lambda x: -x["count"])
    elif sort_by == "Entity type":
        entities = sorted(entities, key=lambda x: (x["label"], x["text"]))
    else:
        entities = sorted(entities, key=lambda x: x["text"].lower())

    # ── Summary metrics ──
    total_unique = len(entities)
    total_mentions = sum(e["count"] for e in entities)
    type_counts = defaultdict(int)
    for e in entities:
        type_counts[e["label"]] += 1

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="summary-box">
            <div class="metric-num">{total_unique}</div>
            <div class="metric-label">Unique Entities</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="summary-box">
            <div class="metric-num">{total_mentions}</div>
            <div class="metric-label">Total Mentions</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="summary-box">
            <div class="metric-num">{len(type_counts)}</div>
            <div class="metric-label">Entity Types Found</div>
        </div>""", unsafe_allow_html=True)

    # ── Grouped view by type ──
    st.markdown("### 📌 Entities by Type")
    grouped = defaultdict(list)
    for e in entities:
        grouped[e["label"]].append(e)

    for label in sorted(grouped.keys()):
        group = grouped[label]
        color_class = TAG_COLOR_MAP.get(label, "tag-DEFAULT")
        with st.expander(f"**{label}** — {len(group)} unique entities", expanded=(len(grouped) <= 4)):
            tags_html = ""
            for ent in group:
                count_badge = f' <span style="color:#64748b;font-size:0.75rem;">×{ent["count"]}</span>' if ent["count"] > 1 else ""
                tags_html += f'<span class="entity-tag {color_class}">{ent["text"]}</span>{count_badge} '
            st.markdown(f'<div style="line-height:2.2;">{tags_html}</div>', unsafe_allow_html=True)

    # ── Full table ──
    st.markdown("### 📋 Full Results Table")
    df = pd.DataFrame(entities).rename(columns={"text": "Entity", "label": "Type", "count": "Mentions"})
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Download ──
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download results as CSV",
        data=csv_bytes,
        file_name=f"ner_results_{uploaded.name}.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Upload a PDF, Word document, CSV, Excel, or text file to get started.")
    st.markdown("""
    **What this app does:**
    - Extracts text from your document (any format)
    - Runs **Named Entity Recognition (NER)** using spaCy
    - Shows you all people, places, organisations, dates, money values, and more
    - Lets you filter, sort, and download the results
    """)