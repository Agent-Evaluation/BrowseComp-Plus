/**
 * Abstract base class for BrowseComp-Plus Copilot-powered agents.
 *
 * Adapted from mas_agents/base_agent.py:
 *  - Replaces Pyserini direct calls with BM25Bridge (subprocess)
 *  - Tracks retrieved_docids and tool_call_counts for BrowseComp-Plus output format
 *  - Keeps same Copilot SDK interface and conversation management
 */

import type { CopilotClient } from "@github/copilot-sdk";
import { BM25Bridge, type SearchResult } from "./bm25-bridge.js";

export interface AgentOutput {
  query_id: string;
  tool_call_counts: Record<string, number>;
  status: string;
  retrieved_docids: string[];
  result: Array<{ type: string; output: string }>;
}

export interface Message {
  role: "user" | "model" | "assistant";
  content: string;
}

export abstract class CopilotBaseAgent {
  modelName: string;
  client: CopilotClient;
  retriever: BM25Bridge;
  conversation: Message[] = [];
  stepCount = 0;
  queryId: string | null = null;
  queryText: string | null = null;
  retrievedDocids: Set<string> = new Set();
  toolCallCounts: Record<string, number> = { search: 0 };

  constructor(
    modelName: string,
    client: CopilotClient,
    retriever: BM25Bridge
  ) {
    this.modelName = modelName;
    this.client = client;
    this.retriever = retriever;
  }

  reset(queryId: string, queryText: string): void {
    this.conversation = [];
    this.stepCount = 0;
    this.queryId = queryId;
    this.queryText = queryText;
    this.retrievedDocids = new Set();
    this.toolCallCounts = { search: 0 };
  }

  abstract act(context: string | null): Promise<string>;

  async executeSearch(query: string): Promise<string> {
    const results = await this.retriever.search(query);
    this.toolCallCounts.search = (this.toolCallCounts.search ?? 0) + 1;

    for (const r of results) {
      this.retrievedDocids.add(String(r.docid));
    }

    return this.retriever.formatResults(results);
  }

  extractSearchQuery(text: string): string | null {
    const match = text.match(/search:\s*(.+?)(?:\n|$)/i);
    return match ? match[1].trim() : null;
  }

  isFinalAnswer(text: string): boolean {
    const lower = text.toLowerCase();
    const hasExactAnswer = lower.includes("exact answer:");
    const hasSearch = /^search:\s*\S/im.test(text);
    return hasExactAnswer && !hasSearch;
  }

  getOutput(): AgentOutput {
    let finalOutput = "";
    for (let i = this.conversation.length - 1; i >= 0; i--) {
      const msg = this.conversation[i];
      if (msg.role === "model" || msg.role === "assistant") {
        finalOutput = msg.content;
        break;
      }
    }

    return {
      query_id: this.queryId ?? "",
      tool_call_counts: { ...this.toolCallCounts },
      status: "completed",
      retrieved_docids: [...this.retrievedDocids],
      result: [{ type: "output_text", output: finalOutput }],
    };
  }

  log(msg: string): void {
    console.log(`[${this.constructor.name}] ${msg}`);
  }
}
