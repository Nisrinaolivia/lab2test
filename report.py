from pathlib import Path

from llama_index.llms.openai import OpenAI
from query import ask, index, collection, QA_PROMPT

CLIENTS = ["paywise", "alphabear", "bravocat", "charlemont", "deltaforce", "echona"]
findings = []


def record(section, question, **kwargs):
    """Ask a question and store the answer with its source files."""
    response = ask(question, **kwargs)
    sources = sorted({n.metadata.get("file_name") for n in response.source_nodes})
    findings.append({
        "section": section,
        "question": question,
        "answer": response.response,
        "sources": sources,
    })


# ---------------------------------------------------------------------------
# 1. Founder risk (Jane Wu)
# ---------------------------------------------------------------------------
record("Founder risk (Jane Wu)",
       "Is there any indication that the founder and CEO Jane Wu may leave the company, "
       "step down, or is losing motivation? When might this happen?")
record("Founder risk (Jane Wu)",
       "What reasons are given for Jane Wu's possible departure, and is there any "
       "succession plan or retention arrangement in place?")

# ---------------------------------------------------------------------------
# 2. PayWise financial position
# ---------------------------------------------------------------------------
record("PayWise financial position",
       "What evidence is there that PayWise is in financial difficulty? Include cash flow "
       "problems, insolvency, restructuring or funding issues.",
       client_name="paywise")
record("PayWise financial position",
       "Has PayWise missed or delayed payments, asked for discounts or extended payment "
       "terms, or indicated it may reduce or terminate its business with Canvassian?",
       client_name="paywise")
record("PayWise financial position",
       "What has Canvassian done or planned in response to PayWise's financial situation?",
       client_name="paywise", doc_type="emails")

# ---------------------------------------------------------------------------
# 3. Change of control: check each key client contract individually
# ---------------------------------------------------------------------------
contract_meta = collection.get(where={"doc_type": "contracts"}, include=["metadatas"])["metadatas"]
contract_files = sorted({m["file_name"] for m in contract_meta})

COC_QUESTION = (
    "Does this contract contain a change of control or assignment clause? "
    "Quote the clause. What are the consequences if Canvassian is acquired "
    "(e.g. termination rights, consent required, notice period, penalties)? "
    "If there is no such clause in the excerpts, say so."
)

for file_name in contract_files:
    client = next((c for c in CLIENTS if c in file_name.lower()), None)
    if not client:
        continue  # only review contracts with the 6 key clients

    engine = index.as_query_engine(
        similarity_top_k=5,
        vector_store_kwargs={"where": {"file_name": file_name}},
        text_qa_template=QA_PROMPT,
    )
    response = engine.query(COC_QUESTION)
    print(f"[Contract reviewed] {file_name}")
    findings.append({
        "section": f"Change of control - {client.capitalize()}",
        "question": COC_QUESTION,
        "answer": response.response,
        "sources": [file_name],
    })

# ---------------------------------------------------------------------------
# 4. Other risks
# ---------------------------------------------------------------------------
OTHER_RISKS = [
    "Is Canvassian involved in or threatened with any litigation, including patent "
    "infringement claims? What is the potential exposure?",
    "Are there any concerns about the CTO or other senior executives (misconduct, "
    "departure, internal investigation)?",
    "Who is Edon Mask and what concerns exist about his role or future at Canvassian?",
    "Are there problems with product launches, product quality or customer dissatisfaction?",
    "Are there any corporate risk concerns raised about the clients Charlemont, "
    "Deltaforce or Echona?",
    "Has Canvassian proposed or entered into any acquisitions or mergers of its own "
    "(e.g. TechFusion)? What are the risks?",
    "Are there any cybersecurity incidents, data breaches or regulatory compliance issues?",
    "Are there issues with staff morale, key employee departures or internal conflict?",
]
for q in OTHER_RISKS:
    record("Other risks", q, top_k=10)

# ---------------------------------------------------------------------------
# 5. Save raw findings (for verification)
# ---------------------------------------------------------------------------
notes = []
for f in findings:
    notes.append(
        f"### {f['section']}\n"
        f"**Question:** {f['question']}\n\n"
        f"**Answer:** {f['answer']}\n\n"
        f"**Sources:** {', '.join(f['sources'])}\n"
    )
notes_text = "\n".join(notes)
Path("findings.md").write_text("# Raw findings\n\n" + notes_text)
print("Saved findings.md")

# ---------------------------------------------------------------------------
# 6. Write the Board report
# ---------------------------------------------------------------------------
REPORT_PROMPT = f"""
You are a senior M&A lawyer acting for a purchaser that is considering acquiring
Canvassian Pty Ltd, a cybersecurity software company. Using ONLY the due diligence
findings below, write a report for the client's Board to support its decision.

Context from the client's instructions:
- The founder, Jane Wu, is regarded as vital to the success of the deal.
- PayWise is Canvassian's largest client (20% of revenue) and is rumoured to be in
  financial difficulty.
- PayWise, Alphabear, Bravocat, Charlemont, Deltaforce and Echona together account
  for 60% of revenue; their contracts must be checked for change of control terms.

Structure the report as follows (Markdown):
1. Executive summary, including an overall recommendation
   (proceed / proceed with conditions / do not proceed) and the key reasons.
2. Critical risk 1: Founder (Jane Wu)
3. Critical risk 2: PayWise financial position and revenue impact
4. Critical risk 3: Change of control. Include a table with one row per contract:
   Client | Contract | Change of control clause? | Consequence on acquisition.
5. Other significant risks identified
6. Recommended actions (e.g. price adjustment, conditions precedent, warranties and
   indemnities, key person/retention arrangements, further investigation).
7. Limitations of this review (AI-assisted review of retrieved excerpts; findings
   should be verified against the source documents).

Cite the source file names for each finding. Do not invent facts that are not in
the findings. Where findings are unclear or conflicting, say so.

FINDINGS:
{notes_text}
"""

llm = OpenAI(model="gpt-4o", temperature=0)
report = llm.complete(REPORT_PROMPT).text
Path("report.md").write_text(report)
print("Saved report.md")

