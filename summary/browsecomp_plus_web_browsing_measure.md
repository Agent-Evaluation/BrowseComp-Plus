# BrowseComp-Plus: Does the Paper Explain Why the Web Browsing Measure Is Used?

**Paper:** BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent  
**Authors:** Chen, Ma, Zhuang et al. (2025)  
**Source:** Zotero item TV5PDCNR

## Question
Does the paper explain anywhere why the measure for Web Browsing is used?

## Answer
**No.** The paper does not define a specific metric called "Web Browsing," nor does it provide a dedicated justification for why any particular measure captures web browsing ability.

## Metrics Used and Their Significance (Section 4.4)

The paper's central goal is **disentangling retrieval from reasoning** in deep-research agents. Each metric isolates a different dimension of agent behavior:

- **Accuracy** — LLM-as-judge (gpt-4.1) compares the agent's final answer against ground truth (adopted from BrowseComp). This is the bottom-line effectiveness measure, but alone it cannot explain *why* an agent fails. The other metrics provide that diagnostic power.

- **Recall** (of evidence documents) — Fraction of human-verified evidence documents retrieved during the agent's entire interaction. This is the key diagnostic metric: if accuracy is low AND recall is low, the bottleneck is retrieval (the agent never found the right documents); if recall is high but accuracy is still low, the bottleneck is reasoning. The paper demonstrates this via oracle retrieval (Section 4.8.1): when gpt-4.1 is given *all* evidence documents, accuracy jumps from 14.58% to 93.49%, proving retrieval is the primary limiting factor.

- **Search Calls** — Average number of search API invocations per query. This is an efficiency/cost metric. The paper shows (Figure 1, Section 4.7) that better retrievers not only improve accuracy but also *reduce* search calls (e.g., gpt-5 goes from 23.23 to 21.74 calls when switching BM25 → Qwen3-Embed-8B). Each call costs money and adds latency (Table 8 reports actual USD costs).

- **Calibration Error** — Alignment between the agent's self-reported confidence and actual accuracy (from Humanity's Last Exam). Important for trustworthiness in practical deployment — a well-calibrated agent lets users know when to trust its answer vs. escalate to a human.

- **Retrieval Effectiveness (Recall@k, nDCG@k)** — Standard IR metrics that evaluate the retriever *in isolation*, without the agent. This is only possible because BrowseComp-Plus provides a fixed corpus with human-verified relevance judgments (Cranfield paradigm). Table 2 reveals that even the best retriever (Qwen3-Embed-8B) achieves only 14.5% Recall@5 on evidence documents, exposing a major gap in retrieval capability for complex queries.

## Why These Metrics Together
The combination enables **component-level diagnosis** of deep-research systems. Prior benchmarks like BrowseComp only reported accuracy on a live web API, making it impossible to tell whether failures were due to bad retrieval, bad reasoning, or bad search strategy. BrowseComp-Plus uses this metric suite to separate those factors.
