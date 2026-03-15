/**
 * Single-Agent System (SAS) for BrowseComp-Plus.
 *
 * Paper §3.1 / Table 2:
 *    A = {a1}
 *    No coordination (baseline)
 *    Max 10 iterations of search + reasoning
 *
 * Adapted from mas_agents/single_agent.py.
 */

import type { CopilotClient } from "@github/copilot-sdk";
import { CopilotBaseAgent, type AgentOutput } from "./base-agent.js";
import { BM25Bridge } from "./bm25-bridge.js";
import { callCopilotWithRetry } from "./llm-client.js";
import { SYSTEM_PROMPT, QUERY_TEMPLATE } from "./prompts.js";

const MAX_ITERATIONS = 10;

export class CopilotSingleAgent extends CopilotBaseAgent {
  constructor(
    modelName: string,
    client: CopilotClient,
    retriever: BM25Bridge
  ) {
    super(modelName, client, retriever);
  }

  async act(context: string | null): Promise<string> {
    if (context) {
      this.conversation.push({ role: "user", content: context });
    }

    for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
      this.stepCount++;

      const actionText = await callCopilotWithRetry(
        this.client,
        this.modelName,
        this.conversation,
        SYSTEM_PROMPT
      );

      this.conversation.push({ role: "model", content: actionText });
      this.log(
        `Iteration ${iteration + 1}/${MAX_ITERATIONS}: ${actionText.slice(0, 120)}...`
      );

      const searchQuery = this.extractSearchQuery(actionText);
      if (searchQuery && !this.isFinalAnswer(actionText)) {
        const searchResults = await this.executeSearch(searchQuery);
        this.log(`  Search: '${searchQuery}' -> ${searchResults.length} chars`);
        this.conversation.push({
          role: "user",
          content: `Search results for '${searchQuery}':\n\n${searchResults}`,
        });
        continue;
      }

      return actionText;
    }

    this.log(`Max iterations (${MAX_ITERATIONS}) reached`);
    return this.conversation.length > 0
      ? this.conversation[this.conversation.length - 1].content
      : "No answer produced.";
  }

  async runQuery(queryId: string, queryText: string): Promise<AgentOutput> {
    this.reset(queryId, queryText);
    const initialPrompt = QUERY_TEMPLATE(queryText);
    await this.act(initialPrompt);
    return this.getOutput();
  }
}
