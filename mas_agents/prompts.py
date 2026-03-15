"""
Prompt templates for BrowseComp-Plus multi-agent evaluation.

Adapted from search_agent/prompts.py with role-specific extensions
for the 5 canonical architectures (Kim et al., 2025).
"""

# ── Core deep-research system prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a deep research agent that answers questions by searching a knowledge base.

IMPORTANT RULES:
1. You MUST use the search tool to find information. Do NOT answer from memory.
2. To search, output a line starting with "search:" followed by your query. Example:
   search: Who won the 2024 Nobel Prize in Physics?
3. You will receive search results with document snippets. Read them carefully.
4. You may search multiple times with different queries to gather more evidence.
5. When you have enough evidence, output your final answer in EXACTLY this format:

Explanation: <your reasoning citing evidence by [docid], e.g. [20]>
Exact Answer: <your succinct final answer>
Confidence: <0% to 100%>

ALWAYS start by searching. Never skip the search step.
"""

# ── Query formatting ──────────────────────────────────────────────────────────
QUERY_TEMPLATE = """\
Question: {question}

Use the search tool to find relevant information, then provide your answer."""

# ── Role-specific prompts for MAS architectures ──────────────────────────────

ORCHESTRATOR_DIRECTIVE_PROMPT = """\
[Orchestrator] You are coordinating a team of research agents to answer \
the following question. Analyse the current state and issue a precise \
directive to your worker agents. Describe:
1. WHAT information to search for
2. WHY this search strategy will help answer the question
3. How to divide the search across workers

Do NOT output the final answer yet. Output only the directive."""

WORKER_PROMPT = """\
[Worker] You are a research worker. Follow the orchestrator's directive. \
Execute your assigned search queries and report your findings. \
Output ONE concrete finding or answer proposal based on your research."""

SYNTHESIS_PROMPT = """\
[Aggregator] The following proposals were made by parallel research agents \
exploring the same question. Synthesise them into a single, well-supported \
final answer.

{proposals}

Output your final response in the required format:
Explanation: {{your explanation citing evidence documents by [docid]}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100%}}"""

DEBATE_PROMPT = """\
Other agents proposed:
{peer_proposals}

Considering your peers' proposals and evidence, output your updated answer. \
If you agree with a peer, explain why. If you disagree, explain what \
evidence supports your position. Use the search tool if needed.

Output in the required format:
Explanation: {{your explanation citing evidence documents by [docid]}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100%}}"""

ORCHESTRATOR_SYNTHESIS_PROMPT = """\
[Orchestrator] Workers have completed their research. Review their proposals \
below and select or synthesise the single best answer.

{worker_proposals}

Output your final response in the required format:
Explanation: {{your explanation citing evidence documents by [docid]}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100%}}"""

PEER_REFINEMENT_PROMPT = """\
[Peer Review] Your fellow workers proposed:
{peer_proposals}

Considering your peers' proposals and evidence, refine your own answer. \
Use the search tool if you need additional evidence.

Output your refined answer proposal."""

# ── Grader template (for LLM-as-judge evaluation) ────────────────────────────
GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not \
based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect \
based on [correct_answer], in the context of this [question]. You should \
judge whether the extracted_final_answer is semantically equivalent to \
[correct_answer], allowing the extracted_final_answer to be string \
variations of [correct_answer]. You should also allow the \
extracted_final_answer to be more precise or verbose than [correct_answer], \
as long as its additional details are correct. Do not comment on any \
background to the problem, do not attempt to solve the problem, do not \
argue for any answer different than [correct_answer], focus only on \
whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] \
given above, or is within a small margin of error for numerical problems. \
Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, \
non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from \
[response]. Put 100 if there is no confidence score available."""
