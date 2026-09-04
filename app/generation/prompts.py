"""Prompt templates for grounded LLM generation.

All prompts follow strict rules to prevent hallucination:
1. Answer using retrieved evidence only
2. Do not invent facts or citations
3. If evidence is insufficient, say so
4. Distinguish evidence from interpretation
"""

from __future__ import annotations

from typing import Optional


SYSTEM_PROMPT_TEMPLATE = """\
You are RAGLens, a citation-aware research assistant. You answer questions
about research papers using ONLY the evidence provided in the context below.

Rules:
1. Answer using only the retrieved evidence. Do not invent facts.
2. Do not invent citations. Only cite sources that appear in the context.
3. Every factual claim should reference a source using [N] format.
4. If evidence is insufficient, say: "I couldn't find sufficient evidence
   in the indexed papers to answer this confidently."
5. Distinguish between evidence (what the papers say) and interpretation
   (your analysis).
6. Never claim a paper says something when the retrieved evidence does not
   support it.
7. Use the section and page information from the sources when possible.

Context format:
SOURCE [N]
Paper: <title>
Section: <section name>
Page: <page number>
Evidence:
<relevant text>

Your response should include:
- The answer to the question
- Citations using [N] format referencing the sources above
- If you cannot find sufficient evidence, state this clearly
"""


QA_PROMPT_TEMPLATE = """\
Question: {query}

Answer the question using the evidence below. Be concise but thorough.
Cite sources using [N] format.

Retrieved Evidence:
{context}

Answer:"""


SUMMARY_PROMPT_TEMPLATE = """\
Please provide a structured summary of the paper "{title}" based on the
retrieved evidence below.

Structure your summary with:
1. Problem: What problem does the paper address?
2. Motivation: Why is this problem important?
3. Methodology: What approach do the authors propose?
4. Dataset: What data do they use?
5. Experiments: How do they evaluate?
6. Results: What are the key findings?
7. Limitations: What limitations do the authors identify?
8. Conclusion: What do they conclude?

For each section, cite the relevant evidence using [N] format.
If the evidence does not support a particular section, note that.
If you cannot find sufficient evidence, state this clearly.

Retrieved Evidence:
{context}

Summary:"""


COMPARISON_PROMPT_TEMPLATE = """\
Compare the following papers based on the retrieved evidence:

Paper A: {paper_a_title}
Paper B: {paper_b_title}

Structure your comparison with:
1. Problem: How do the papers' problem formulations compare?
2. Methodology: What are the key methodological differences?
3. Datasets: What datasets do they use?
4. Models: What models/architectures are involved?
5. Results: How do their results compare?
6. Strengths: What are each paper's strengths?
7. Limitations: What limitations are identified?
8. Differences and Similarities: Key contrasts and commonalities

Cite evidence from both papers using [N] format.
If evidence is insufficient for any section, note it.
"""


LITERATURE_REVIEW_PROMPT_TEMPLATE = """\
Generate a literature review addressing the question:

"{query}"

Your review should include:
1. Introduction: Overview of the topic
2. Major Approaches: Key methods and techniques
3. Comparison: How approaches differ and relate
4. Evolution Over Time: How the field has progressed
5. Common Datasets: Standard evaluation data
6. Common Metrics: How performance is measured
7. Limitations: Known limitations in the field
8. Research Gaps: Areas needing further work
9. References: All cited sources

Synthesize across papers rather than summarizing each individually.
Cite evidence using [N] format.
If evidence is insufficient, state this clearly.

Retrieved Evidence:
{context}

Literature Review:"""


EVIDENCE_SEARCH_PROMPT_TEMPLATE = """\
Find and present evidence from the indexed papers that supports or refutes
the claim: "{claim}"

Present:
1. Supporting evidence (with citations and page references)
2. Contradicting evidence (if any, with citations)
3. Confidence assessment (high/medium/low)
4. Summary of the evidence landscape

Retrieved Evidence:
{context}

Evidence Analysis:"""


REWRITE_QUERY_PROMPT_TEMPLATE = """\
Rewrite the following query to be more specific and self-contained,
preserving the user's original intent. Only rewrite if the query is
ambiguous or could benefit from additional context.

Original query: "{query}"

Rewritten query:"""


def build_prompt(
    template: str,
    **kwargs: str,
) -> str:
    """Build a prompt from a template and keyword arguments."""
    return template.format(**kwargs)


def get_system_prompt() -> str:
    """Return the system prompt for grounded generation."""
    return SYSTEM_PROMPT_TEMPLATE
