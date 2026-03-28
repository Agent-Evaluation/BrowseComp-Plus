/**
 * Multi-Agent System (MAS) architectures for BrowseComp-Plus.
 *
 * "Towards a Science of Scaling Agent Systems"
 * Kim et al., 2025 (arXiv:2512.08296), Section 3.1 / Table 2.
 *
 * Four MAS architectures:
 *   - Independent MAS    → CopilotIndependentAgent
 *   - Centralized MAS    → CopilotCentralizedAgent
 *   - Decentralized MAS  → CopilotDecentralizedAgent
 *   - Hybrid MAS         → CopilotHybridAgent
 *
 * Adapted from mas_agents/multi_agents.py.
 */

import type { CopilotClient } from "@github/copilot-sdk";
import { CopilotBaseAgent, type AgentOutput, type Message } from "./base-agent.js";
import { BM25Bridge } from "./bm25-bridge.js";
import { callCopilotWithRetry } from "./llm-client.js";
import {
  SYSTEM_PROMPT,
  QUERY_TEMPLATE,
  SYNTHESIS_PROMPT,
  ORCHESTRATOR_DIRECTIVE_PROMPT,
  WORKER_PROMPT,
  ORCHESTRATOR_SYNTHESIS_PROMPT,
  DEBATE_PROMPT,
  PEER_REFINEMENT_PROMPT,
} from "./prompts.js";

const MAX_ITERATIONS_PER_AGENT = 3;

/**
 * Run a sub-agent that can iteratively search before producing its answer.
 * Shared helper used by all MAS architectures for individual agent reasoning.
 */
async function runAgentWithSearch(
  client: CopilotClient,
  modelName: string,
  messages: Message[],
  systemPrompt: string,
  baseAgent: CopilotBaseAgent,
  maxIterations = MAX_ITERATIONS_PER_AGENT,
  temperature = 0.7
): Promise<string> {
  const localMsgs: Message[] = [...messages];

  for (let i = 0; i < maxIterations; i++) {
    const response = await callCopilotWithRetry(
      client,
      modelName,
      localMsgs,
      systemPrompt,
      temperature
    );

    localMsgs.push({ role: "model", content: response });

    const searchQuery = baseAgent.extractSearchQuery(response);
    if (searchQuery && !baseAgent.isFinalAnswer(response)) {
      const searchResults = await baseAgent.executeSearch(searchQuery);
      localMsgs.push({
        role: "user",
        content: `Search results for '${searchQuery}':\n\n${searchResults}`,
      });
      continue;
    }

    return response;
  }

  return localMsgs.length > 0
    ? localMsgs[localMsgs.length - 1].content
    : "No answer produced.";
}

// ── Independent MAS ─────────────────────────────────────────────────────────

export class CopilotIndependentAgent extends CopilotBaseAgent {
  numAgents: number;

  constructor(
    modelName: string,
    client: CopilotClient,
    retriever: BM25Bridge,
    numAgents = 3
  ) {
    super(modelName, client, retriever);
    this.numAgents = numAgents;
  }

  async act(context: string | null): Promise<string> {
    if (context) {
      this.conversation.push({ role: "user", content: context });
    }

    // Phase 1: n agents explore in parallel
    const agentTasks = Array.from({ length: this.numAgents }, (_, i) =>
      runAgentWithSearch(
        this.client,
        this.modelName,
        this.conversation,
        SYSTEM_PROMPT,
        this
      ).catch((e) => {
        this.log(`Agent ${i} failed: ${e}`);
        return null;
      })
    );

    const results = await Promise.all(agentTasks);
    const proposals = results.filter((r): r is string => r !== null);

    if (proposals.length === 0) {
      return "No answer produced: All agents failed";
    }

    // Phase 2: synthesis_only aggregation
    const proposalsText = proposals
      .map((p, i) => `[Agent ${i + 1} proposal]\n${p}`)
      .join("\n\n");
    const aggregationSystem =
      SYSTEM_PROMPT + "\n\n" + SYNTHESIS_PROMPT(proposalsText);

    const synthesisMsgs: Message[] = [
      ...this.conversation,
      {
        role: "user",
        content: "Synthesise the proposals above into one final answer.",
      },
    ];

    const action = await callCopilotWithRetry(
      this.client,
      this.modelName,
      synthesisMsgs,
      aggregationSystem,
      0.0
    );

    this.conversation.push({ role: "model", content: action });
    return action;
  }

  async runQuery(queryId: string, queryText: string): Promise<AgentOutput> {
    this.reset(queryId, queryText);
    const initialPrompt = QUERY_TEMPLATE(queryText);
    await this.act(initialPrompt);
    return this.getOutput();
  }
}

// ── Centralized MAS ─────────────────────────────────────────────────────────

export class CopilotCentralizedAgent extends CopilotBaseAgent {
  numAgents: number;
  rounds: number;

  constructor(
    modelName: string,
    client: CopilotClient,
    retriever: BM25Bridge,
    numAgents = 3,
    rounds = 5
  ) {
    super(modelName, client, retriever);
    this.numAgents = numAgents;
    this.rounds = rounds;
  }

  async act(context: string | null): Promise<string> {
    if (context) {
      this.conversation.push({ role: "user", content: context });
    }

    let workingContext: Message[] = [...this.conversation];
    let finalAction =
      "No answer produced: Centralized agent produced no output";

    for (let r = 0; r < this.rounds; r++) {
      // Orchestrator directive
      const orchSystem =
        SYSTEM_PROMPT + "\n\n" + ORCHESTRATOR_DIRECTIVE_PROMPT;
      const directive = await callCopilotWithRetry(
        this.client,
        this.modelName,
        workingContext,
        orchSystem
      );
      this.log(
        `Round ${r + 1}/${this.rounds} | Orchestrator directive: ${directive.slice(0, 100)}...`
      );

      // n workers receive directive in parallel
      const workerMsgs: Message[] = [
        ...workingContext,
        { role: "model", content: directive },
        { role: "user", content: WORKER_PROMPT },
      ];

      const workerTasks = Array.from({ length: this.numAgents }, () =>
        runAgentWithSearch(
          this.client,
          this.modelName,
          workerMsgs,
          SYSTEM_PROMPT,
          this,
          MAX_ITERATIONS_PER_AGENT,
          0.7
        ).catch(() => null)
      );

      const results = await Promise.all(workerTasks);
      const workerOutputs = results.filter(
        (o): o is string => o !== null
      );

      if (workerOutputs.length === 0) {
        this.log(`Round ${r + 1} | All workers failed`);
        break;
      }

      this.log(
        `Round ${r + 1} | Got ${workerOutputs.length} worker outputs`
      );

      // Orchestrator synthesises worker outputs
      const workerSummary = workerOutputs
        .map((o, i) => `- Worker ${i + 1}: ${o}`)
        .join("\n\n");
      const synthesisSystem =
        SYSTEM_PROMPT +
        "\n\n" +
        ORCHESTRATOR_SYNTHESIS_PROMPT(workerSummary);
      const synthesisMsgs: Message[] = [
        ...workerMsgs,
        { role: "model", content: workerSummary },
        {
          role: "user",
          content:
            "Orchestrator: review proposals and output the final answer.",
        },
      ];
      finalAction = await callCopilotWithRetry(
        this.client,
        this.modelName,
        synthesisMsgs,
        synthesisSystem
      );
      this.log(
        `Round ${r + 1} | Orchestrator synthesis: ${finalAction.slice(0, 100)}...`
      );

      if (this.isFinalAnswer(finalAction)) break;

      workingContext = [
        ...synthesisMsgs,
        { role: "model", content: finalAction },
      ];
    }

    this.conversation.push({ role: "model", content: finalAction });
    return finalAction;
  }

  async runQuery(queryId: string, queryText: string): Promise<AgentOutput> {
    this.reset(queryId, queryText);
    const initialPrompt = QUERY_TEMPLATE(queryText);
    await this.act(initialPrompt);
    return this.getOutput();
  }
}

// ── Decentralized MAS ───────────────────────────────────────────────────────

export class CopilotDecentralizedAgent extends CopilotBaseAgent {
  numAgents: number;
  rounds: number;

  constructor(
    modelName: string,
    client: CopilotClient,
    retriever: BM25Bridge,
    numAgents = 3,
    rounds = 3
  ) {
    super(modelName, client, retriever);
    this.numAgents = numAgents;
    this.rounds = rounds;
  }

  async act(context: string | null): Promise<string> {
    if (context) {
      this.conversation.push({ role: "user", content: context });
    }

    // Initial proposals: all agents in parallel
    const initialTasks = Array.from({ length: this.numAgents }, () =>
      runAgentWithSearch(
        this.client,
        this.modelName,
        this.conversation,
        SYSTEM_PROMPT,
        this,
        MAX_ITERATIONS_PER_AGENT,
        0.7
      ).catch(() => null)
    );

    const initialResults = await Promise.all(initialTasks);
    let proposals = initialResults.filter(
      (p): p is string => p !== null
    );

    if (proposals.length === 0) {
      return "No answer produced: All agents failed";
    }

    // d debate rounds: all-to-all peer exchange
    for (let d = 0; d < this.rounds; d++) {
      this.log(
        `Debate round ${d + 1}/${this.rounds} | ${proposals.length} proposals`
      );

      const peerSummary = proposals
        .map((p, j) => `- Agent ${j + 1}: ${p}`)
        .join("\n\n");
      const debateContext = DEBATE_PROMPT(peerSummary);

      const debateTasks = Array.from({ length: this.numAgents }, () =>
        runAgentWithSearch(
          this.client,
          this.modelName,
          [
            ...this.conversation,
            { role: "user", content: debateContext },
          ],
          SYSTEM_PROMPT,
          this,
          MAX_ITERATIONS_PER_AGENT,
          0.7
        ).catch(() => null)
      );

      const debateResults = await Promise.all(debateTasks);
      const updated = debateResults.filter(
        (p): p is string => p !== null
      );
      if (updated.length > 0) {
        proposals = updated;
      }
    }

    // Consensus: majority vote
    if (proposals.length === 0) {
      return "No answer produced: All agents failed during debate";
    }

    // Simple majority: count occurrences
    const counts = new Map<string, number>();
    for (const p of proposals) {
      counts.set(p, (counts.get(p) ?? 0) + 1);
    }
    let bestAction = proposals[0];
    let bestCount = 0;
    for (const [action, count] of counts) {
      if (count > bestCount) {
        bestAction = action;
        bestCount = count;
      }
    }

    this.log(`Consensus reached: ${bestAction.slice(0, 100)}...`);
    this.conversation.push({ role: "model", content: bestAction });
    return bestAction;
  }

  async runQuery(queryId: string, queryText: string): Promise<AgentOutput> {
    this.reset(queryId, queryText);
    const initialPrompt = QUERY_TEMPLATE(queryText);
    await this.act(initialPrompt);
    return this.getOutput();
  }
}

// ── Hybrid MAS ──────────────────────────────────────────────────────────────

export class CopilotHybridAgent extends CopilotBaseAgent {
  numAgents: number;
  rounds: number;
  peerRounds: number;

  constructor(
    modelName: string,
    client: CopilotClient,
    retriever: BM25Bridge,
    numAgents = 3,
    rounds = 1,
    peerRounds = 1
  ) {
    super(modelName, client, retriever);
    this.numAgents = numAgents;
    this.rounds = rounds;
    this.peerRounds = peerRounds;
  }

  async act(context: string | null): Promise<string> {
    if (context) {
      this.conversation.push({ role: "user", content: context });
    }

    let workingContext: Message[] = [...this.conversation];
    let finalAction = "No answer produced: Hybrid agent produced no output";

    for (let r = 0; r < this.rounds; r++) {
      // Step 1: Orchestrator directive
      const orchSystem =
        SYSTEM_PROMPT + "\n\n" + ORCHESTRATOR_DIRECTIVE_PROMPT;
      const directive = await callCopilotWithRetry(
        this.client,
        this.modelName,
        workingContext,
        orchSystem
      );
      this.log(
        `Round ${r + 1} | Orchestrator directive: ${directive.slice(0, 100)}...`
      );

      // Step 2: Workers propose in parallel
      const workerMsgs: Message[] = [
        ...workingContext,
        { role: "model", content: directive },
        { role: "user", content: WORKER_PROMPT },
      ];

      const workerTasks = Array.from({ length: this.numAgents }, () =>
        runAgentWithSearch(
          this.client,
          this.modelName,
          workerMsgs,
          SYSTEM_PROMPT,
          this,
          MAX_ITERATIONS_PER_AGENT,
          0.7
        ).catch(() => null)
      );

      const workerResults = await Promise.all(workerTasks);
      let proposals = workerResults.filter(
        (a): a is string => a !== null
      );

      if (proposals.length === 0) {
        this.log(`Round ${r + 1} | All workers failed`);
        break;
      }

      this.log(
        `Round ${r + 1} | Got ${proposals.length} worker proposals`
      );

      // Step 3: p lateral peer rounds
      for (let p = 0; p < this.peerRounds; p++) {
        const peerSummary = proposals
          .map((prop, j) => `- Worker ${j + 1}: ${prop}`)
          .join("\n\n");
        const peerContext = PEER_REFINEMENT_PROMPT(peerSummary);

        const peerTasks = Array.from({ length: this.numAgents }, () =>
          runAgentWithSearch(
            this.client,
            this.modelName,
            [
              ...workerMsgs,
              { role: "user", content: peerContext },
            ],
            SYSTEM_PROMPT,
            this,
            MAX_ITERATIONS_PER_AGENT,
            0.7
          ).catch(() => null)
        );

        const peerResults = await Promise.all(peerTasks);
        const refined = peerResults.filter(
          (a): a is string => a !== null
        );
        if (refined.length > 0) {
          proposals = refined;
        }
        this.log(
          `Round ${r + 1} | Peer round ${p + 1} -> ${proposals.length} refined proposals`
        );
      }

      // Step 4: Orchestrator synthesises peer-refined proposals
      const finalSummary = proposals
        .map((prop, i) => `- Worker ${i + 1}: ${prop}`)
        .join("\n\n");
      const synthesisSystem =
        SYSTEM_PROMPT +
        "\n\n" +
        ORCHESTRATOR_SYNTHESIS_PROMPT(finalSummary);
      const synthesisMsgs: Message[] = [
        ...workerMsgs,
        { role: "model", content: finalSummary },
        {
          role: "user",
          content:
            "Orchestrator: workers have exchanged proposals. Select the single best answer.",
        },
      ];
      finalAction = await callCopilotWithRetry(
        this.client,
        this.modelName,
        synthesisMsgs,
        synthesisSystem
      );
      this.log(
        `Round ${r + 1} | Orchestrator synthesis: ${finalAction.slice(0, 100)}...`
      );

      workingContext = [
        ...synthesisMsgs,
        { role: "model", content: finalAction },
      ];
    }

    this.conversation.push({ role: "model", content: finalAction });
    return finalAction;
  }

  async runQuery(queryId: string, queryText: string): Promise<AgentOutput> {
    this.reset(queryId, queryText);
    const initialPrompt = QUERY_TEMPLATE(queryText);
    await this.act(initialPrompt);
    return this.getOutput();
  }
}
