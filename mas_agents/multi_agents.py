"""
Multi-Agent System (MAS) architectures for BrowseComp-Plus, implemented according to:

  "Towards a Science of Scaling Agent Systems"
  Kim et al., 2025 (arXiv:2512.08296), Section 3.1 / Table 2.

Four MAS architectures:
  - Independent MAS                 → CopilotIndependentAgent
  - Centralized MAS                 → CopilotCentralizedAgent
  - Decentralized MAS               → CopilotDecentralizedAgent
  - Hybrid MAS                      → CopilotHybridAgent

Communication topology (C) and orchestration policy (Ω) for each:

  Independent:   C = {(ai, aagg)},              Ω = synthesis_only
  Centralized:   C = {(aorch, ai) : ∀i},        Ω = hierarchical
  Decentralized: C = {(ai, aj) : ∀i,j, i≠j},   Ω = consensus
  Hybrid:        C = star + peer edges,          Ω = hierarchical + lateral

Adapted from plancraft/agents/copilot_multi.py.
"""

import asyncio
from collections import Counter

from copilot import CopilotClient

from .base_agent import CopilotBaseAgent, RetrieverTool
from .llm_client import call_copilot_with_retry
from .prompts import (
    SYSTEM_PROMPT,
    QUERY_TEMPLATE,
    SYNTHESIS_PROMPT,
    ORCHESTRATOR_DIRECTIVE_PROMPT,
    WORKER_PROMPT,
    ORCHESTRATOR_SYNTHESIS_PROMPT,
    DEBATE_PROMPT,
    PEER_REFINEMENT_PROMPT,
)

MAX_ITERATIONS_PER_AGENT = 3


async def _run_agent_with_search(
    client: CopilotClient,
    model_name: str,
    messages: list[dict],
    system_prompt: str,
    retriever_tool,
    base_agent: CopilotBaseAgent,
    max_iterations: int = MAX_ITERATIONS_PER_AGENT,
    temperature: float = 0.7,
) -> str:
    """
    Run a sub-agent that can iteratively search before producing its answer.
    Shared helper used by all MAS architectures for individual agent reasoning.
    """
    local_msgs = list(messages)

    for _ in range(max_iterations):
        response = await call_copilot_with_retry(
            client, model_name, local_msgs, system_prompt, temperature=temperature
        )

        local_msgs.append({"role": "model", "content": response})

        search_query = base_agent._extract_search_query(response)
        if search_query and not base_agent._is_final_answer(response):
            search_results = base_agent._execute_search(search_query)
            local_msgs.append({
                "role": "user",
                "content": f"Search results for '{search_query}':\n\n{search_results}",
            })
            continue

        return response

    return local_msgs[-1]["content"] if local_msgs else "No answer produced."


class CopilotIndependentAgent(CopilotBaseAgent):
    """
    Independent MAS — Paper §3.1:
        A = {a1, ..., an}
        C = {(ai, aagg)}           (agent-to-aggregator only, no peer communication)
        Ω = synthesis_only

    n agents explore in parallel, then ONE aggregation call synthesises results.
    Complexity: O(nk) + O(1) aggregation call.
    """

    def __init__(
        self,
        model_name: str,
        client: CopilotClient,
        retriever: RetrieverTool,
        num_agents: int = 3,
    ):
        super().__init__(model_name, client, retriever)
        self.num_agents = num_agents

    async def act(self, context: str | None) -> str:
        if context:
            self.conversation.append({"role": "user", "content": context})

        # ── Phase 1: n agents explore in parallel (no communication) ──────────
        async def _call_agent(i: int) -> str | None:
            try:
                return await _run_agent_with_search(
                    self.client, self.model_name, self.conversation,
                    SYSTEM_PROMPT, self.retriever, self,
                )
            except Exception as e:
                self.log(f"Agent {i} failed: {e}")
                return None

        results = await asyncio.gather(*[_call_agent(i) for i in range(self.num_agents)])
        proposals = [r for r in results if r is not None]

        if not proposals:
            return "No answer produced: All agents failed"

        # ── Phase 2: synthesis_only aggregation ───────────────────────────────
        proposals_text = "\n\n".join(
            [f"[Agent {i+1} proposal]\n{p}" for i, p in enumerate(proposals)]
        )
        aggregation_system = SYSTEM_PROMPT + "\n\n" + SYNTHESIS_PROMPT.format(
            proposals=proposals_text
        )

        synthesis_msgs = self.conversation + [
            {"role": "user", "content": "Synthesise the proposals above into one final answer."}
        ]

        action = await call_copilot_with_retry(
            self.client, self.model_name, synthesis_msgs,
            aggregation_system, temperature=0.0,
        )

        self.conversation.append({"role": "model", "content": action})
        return action

    async def run_query(self, query_id: str, query_text: str) -> dict:
        self.reset(query_id, query_text)
        initial_prompt = QUERY_TEMPLATE.format(question=query_text)
        await self.act(initial_prompt)
        return self.get_output()


class CopilotCentralizedAgent(CopilotBaseAgent):
    """
    Centralized MAS — Paper §3.1:
        A = {aorch, a1, ..., an}
        C = {(aorch, ai) : ∀i}    (orchestrator-to-all-agents, no peer communication)
        Ω = hierarchical

    A single orchestrator coordinates r rounds. In each round:
      1. Orchestrator generates a directive.
      2. n workers receive the directive in parallel and propose actions.
      3. Orchestrator synthesises the worker outputs.
    Complexity: O(rnk) + O(r) orchestrator calls.
    """

    def __init__(
        self,
        model_name: str,
        client: CopilotClient,
        retriever: RetrieverTool,
        num_agents: int = 3,
        rounds: int = 5,
    ):
        super().__init__(model_name, client, retriever)
        self.num_agents = num_agents
        self.rounds = rounds

    async def act(self, context: str | None) -> str:
        if context:
            self.conversation.append({"role": "user", "content": context})

        working_context = list(self.conversation)
        final_action = "No answer produced: Centralized agent produced no output"

        for r in range(self.rounds):
            # ── Orchestrator directive ─────────────────────────────────────────
            orch_system = SYSTEM_PROMPT + "\n\n" + ORCHESTRATOR_DIRECTIVE_PROMPT
            directive = await call_copilot_with_retry(
                self.client, self.model_name, working_context, orch_system
            )
            self.log(f"Round {r+1}/{self.rounds} | Orchestrator directive: {directive[:100]}...")

            # ── n workers receive directive in parallel ────────────────────────
            worker_msgs = working_context + [
                {"role": "model", "content": directive},
                {"role": "user", "content": WORKER_PROMPT},
            ]

            worker_tasks = [
                _run_agent_with_search(
                    self.client, self.model_name, worker_msgs,
                    SYSTEM_PROMPT, self.retriever, self, temperature=0.7,
                )
                for _ in range(self.num_agents)
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            worker_outputs = [o for o in results if not isinstance(o, Exception)]

            if not worker_outputs:
                self.log(f"Round {r+1} | All workers failed")
                break

            self.log(f"Round {r+1} | Got {len(worker_outputs)} worker outputs")

            # ── Orchestrator synthesises worker outputs ────────────────────────
            worker_summary = "\n\n".join(
                [f"- Worker {i+1}: {o}" for i, o in enumerate(worker_outputs)]
            )
            synthesis_system = SYSTEM_PROMPT + "\n\n" + ORCHESTRATOR_SYNTHESIS_PROMPT.format(
                worker_proposals=worker_summary
            )
            synthesis_msgs = worker_msgs + [
                {"role": "model", "content": worker_summary},
                {"role": "user", "content": "Orchestrator: review proposals and output the final answer."},
            ]
            final_action = await call_copilot_with_retry(
                self.client, self.model_name, synthesis_msgs, synthesis_system
            )
            self.log(f"Round {r+1} | Orchestrator synthesis: {final_action[:100]}...")

            # Check if we have a final answer; if so, stop early
            if self._is_final_answer(final_action):
                break

            # Update context for next round
            working_context = synthesis_msgs + [{"role": "model", "content": final_action}]

        self.conversation.append({"role": "model", "content": final_action})
        return final_action

    async def run_query(self, query_id: str, query_text: str) -> dict:
        self.reset(query_id, query_text)
        initial_prompt = QUERY_TEMPLATE.format(question=query_text)
        await self.act(initial_prompt)
        return self.get_output()


class CopilotDecentralizedAgent(CopilotBaseAgent):
    """
    Decentralized MAS — Paper §3.1:
        A = {a1, ..., an}
        C = {(ai, aj) : ∀i,j, i≠j}   (all-to-all peer communication)
        Ω = consensus

    Agents communicate in d sequential debate rounds. Each agent sees all
    peers' proposals and updates its own position. After d rounds a majority
    vote produces the consensus action.
    Complexity: O(dnk) + O(1). Memory: O(dnk) per agent.
    """

    def __init__(
        self,
        model_name: str,
        client: CopilotClient,
        retriever: RetrieverTool,
        num_agents: int = 3,
        rounds: int = 3,
    ):
        super().__init__(model_name, client, retriever)
        self.num_agents = num_agents
        self.rounds = rounds

    async def act(self, context: str | None) -> str:
        if context:
            self.conversation.append({"role": "user", "content": context})

        # ── Initial proposals: all agents in parallel ─────────────────────────
        initial_tasks = [
            _run_agent_with_search(
                self.client, self.model_name, self.conversation,
                SYSTEM_PROMPT, self.retriever, self, temperature=0.7,
            )
            for _ in range(self.num_agents)
        ]
        results = await asyncio.gather(*initial_tasks, return_exceptions=True)
        proposals = [p for p in results if not isinstance(p, Exception)]

        if not proposals:
            return "No answer produced: All agents failed"

        # ── d debate rounds: all-to-all peer exchange ─────────────────────────
        for d in range(self.rounds):
            self.log(f"Debate round {d+1}/{self.rounds} | {len(proposals)} proposals")

            peer_summary = "\n\n".join(
                [f"- Agent {j+1}: {p}" for j, p in enumerate(proposals)]
            )
            debate_context = DEBATE_PROMPT.format(peer_proposals=peer_summary)

            debate_tasks = [
                _run_agent_with_search(
                    self.client, self.model_name,
                    self.conversation + [{"role": "user", "content": debate_context}],
                    SYSTEM_PROMPT, self.retriever, self, temperature=0.7,
                )
                for _ in range(self.num_agents)
            ]
            results = await asyncio.gather(*debate_tasks, return_exceptions=True)
            updated = [p for p in results if not isinstance(p, Exception)]
            if updated:
                proposals = updated

        # ── Consensus: majority vote after d rounds ───────────────────────────
        if not proposals:
            return "No answer produced: All agents failed during debate"

        counts = Counter(proposals)
        best_action, _ = counts.most_common(1)[0]
        self.log(f"Consensus reached: {best_action[:100]}...")

        self.conversation.append({"role": "model", "content": best_action})
        return best_action

    async def run_query(self, query_id: str, query_text: str) -> dict:
        self.reset(query_id, query_text)
        initial_prompt = QUERY_TEMPLATE.format(question=query_text)
        await self.act(initial_prompt)
        return self.get_output()


class CopilotHybridAgent(CopilotBaseAgent):
    """
    Hybrid MAS — Paper §3.1:
        A = {aorch, a1, ..., an}
        C = star + peer edges      (orchestrator→workers AND worker↔worker)
        Ω = hierarchical + lateral

    Combines orchestrated hierarchy (r orchestrator rounds) with limited
    peer communication (p lateral rounds between workers).
    Complexity: O(rnk + pn) per step.
    """

    def __init__(
        self,
        model_name: str,
        client: CopilotClient,
        retriever: RetrieverTool,
        num_agents: int = 3,
        rounds: int = 1,
        peer_rounds: int = 1,
    ):
        super().__init__(model_name, client, retriever)
        self.num_agents = num_agents
        self.rounds = rounds
        self.peer_rounds = peer_rounds

    async def act(self, context: str | None) -> str:
        if context:
            self.conversation.append({"role": "user", "content": context})

        working_context = list(self.conversation)
        final_action = "No answer produced: Hybrid agent produced no output"

        for r in range(self.rounds):
            # ── Step 1: Orchestrator directive (hierarchical control) ──────────
            orch_system = SYSTEM_PROMPT + "\n\n" + ORCHESTRATOR_DIRECTIVE_PROMPT
            directive = await call_copilot_with_retry(
                self.client, self.model_name, working_context, orch_system
            )
            self.log(f"Round {r+1} | Orchestrator directive: {directive[:100]}...")

            # ── Step 2: Workers propose in parallel (star edges) ──────────────
            worker_msgs = working_context + [
                {"role": "model", "content": directive},
                {"role": "user", "content": WORKER_PROMPT},
            ]
            worker_tasks = [
                _run_agent_with_search(
                    self.client, self.model_name, worker_msgs,
                    SYSTEM_PROMPT, self.retriever, self, temperature=0.7,
                )
                for _ in range(self.num_agents)
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            proposals = [a for a in results if not isinstance(a, Exception)]

            if not proposals:
                self.log(f"Round {r+1} | All workers failed")
                break

            self.log(f"Round {r+1} | Got {len(proposals)} worker proposals")

            # ── Step 3: p lateral peer rounds (peer edges) ────────────────────
            for p in range(self.peer_rounds):
                peer_summary = "\n\n".join(
                    [f"- Worker {j+1}: {prop}" for j, prop in enumerate(proposals)]
                )
                peer_context = PEER_REFINEMENT_PROMPT.format(peer_proposals=peer_summary)

                peer_tasks = [
                    _run_agent_with_search(
                        self.client, self.model_name,
                        worker_msgs + [{"role": "user", "content": peer_context}],
                        SYSTEM_PROMPT, self.retriever, self, temperature=0.7,
                    )
                    for _ in range(self.num_agents)
                ]
                results = await asyncio.gather(*peer_tasks, return_exceptions=True)
                refined = [a for a in results if not isinstance(a, Exception)]
                if refined:
                    proposals = refined
                self.log(f"Round {r+1} | Peer round {p+1} -> {len(proposals)} refined proposals")

            # ── Step 4: Orchestrator synthesises peer-refined proposals ────────
            final_summary = "\n\n".join(
                [f"- Worker {i+1}: {prop}" for i, prop in enumerate(proposals)]
            )
            synthesis_system = SYSTEM_PROMPT + "\n\n" + ORCHESTRATOR_SYNTHESIS_PROMPT.format(
                worker_proposals=final_summary
            )
            synthesis_msgs = worker_msgs + [
                {"role": "model", "content": final_summary},
                {"role": "user", "content": "Orchestrator: workers have exchanged proposals. Select the single best answer."},
            ]
            final_action = await call_copilot_with_retry(
                self.client, self.model_name, synthesis_msgs, synthesis_system
            )
            self.log(f"Round {r+1} | Orchestrator synthesis: {final_action[:100]}...")

            working_context = synthesis_msgs + [{"role": "model", "content": final_action}]

        self.conversation.append({"role": "model", "content": final_action})
        return final_action

    async def run_query(self, query_id: str, query_text: str) -> dict:
        self.reset(query_id, query_text)
        initial_prompt = QUERY_TEMPLATE.format(question=query_text)
        await self.act(initial_prompt)
        return self.get_output()
