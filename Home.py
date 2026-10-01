import tempfile
from pathlib import Path

import pypandoc
import streamlit as st
from query import index, QA_PROMPT

st.set_page_config(page_title="Canvassian Due Diligence", page_icon="📑", layout="wide")

st.title("Canvassian Pty Ltd: M&A Due Diligence")
st.caption(
    "AI-assisted review of emails, contracts and board papers. "
    "Findings should be verified against the source documents."
)

CLIENTS = ["All", "PayWise", "Alphabear", "Bravocat", "Charlemont", "Deltaforce", "Echona"]
DOC_TYPES = {"All": None, "Emails": "emails", "Contracts": "contracts", "Board papers": "board_papers"}
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@st.cache_data
def markdown_to_docx(md_text):
    """Convert Markdown text to a Word document (returned as bytes)."""
    with tempfile.NamedTemporaryFile(suffix=".docx") as tmp:
        pypandoc.convert_text(md_text, "docx", format="md", outputfile=tmp.name)
        return Path(tmp.name).read_bytes()


tab_report, tab_ask, tab_findings = st.tabs(["📄 Board Report", "🔎 Ask the Documents", "🗂️ Raw Findings"])

# ---------------------------------------------------------------------------
# Tab 1: Board report
# ---------------------------------------------------------------------------
with tab_report:
    report_path = Path("report.md")
    if report_path.exists():
        report_text = report_path.read_text()
        st.download_button(
            "Download report (Word)",
            markdown_to_docx(report_text),
            file_name="canvassian_due_diligence_report.docx",
            mime=DOCX_MIME,
        )
        st.markdown(report_text)
    else:
        st.info("No report found yet. Run `python report.py` in the terminal first.")

# ---------------------------------------------------------------------------
# Tab 2: Ask the documents
# ---------------------------------------------------------------------------
with tab_ask:
    question = st.text_area(
        "Your question",
        placeholder="e.g. Does the Echona contract allow termination on a change of control?",
    )

    col1, col2, col3 = st.columns(3)
    doc_type_label = col1.selectbox("Document type", list(DOC_TYPES))
    client_label = col2.selectbox("Client mentioned", CLIENTS)
    top_k = col3.slider("Number of excerpts to retrieve", 3, 15, 8)

    if st.button("Search", type="primary") and question.strip():
        # Build the ChromaDB filter
        conditions = []
        if DOC_TYPES[doc_type_label]:
            conditions.append({"doc_type": DOC_TYPES[doc_type_label]})
        if client_label != "All":
            conditions.append({f"mentions_{client_label.lower()}": True})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}
        else:
            where = None

        with st.spinner("Searching the documents..."):
            engine = index.as_query_engine(
                similarity_top_k=top_k,
                vector_store_kwargs={"where": where} if where else {},
                text_qa_template=QA_PROMPT,
            )
            response = engine.query(question)

        st.markdown("### Answer")
        st.markdown(response.response)

        st.markdown("### Source excerpts")
        for node in response.source_nodes:
            label = f"{node.metadata.get('file_name')} ({node.metadata.get('doc_type')}, score {node.score:.3f})"
            with st.expander(label):
                st.write(node.get_content())

# ---------------------------------------------------------------------------
# Tab 3: Raw findings
# ---------------------------------------------------------------------------
with tab_findings:
    findings_path = Path("findings.md")
    if findings_path.exists():
        findings_text = findings_path.read_text()
        st.download_button(
            "Download findings (Word)",
            markdown_to_docx(findings_text),
            file_name="canvassian_raw_findings.docx",
            mime=DOCX_MIME,
        )
        st.markdown(findings_text)
    else:
        st.info("No findings found yet. Run `python report.py` in the terminal first.")