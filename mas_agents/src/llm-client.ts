/**
 * Copilot SDK wrapper for BrowseComp-Plus agents.
 *
 * Uses the GitHub Copilot SDK (@github/copilot-sdk) to call LLMs through
 * the Copilot CLI. Requires the Copilot CLI to be installed and available
 * in $PATH.
 *
 * Adapted from mas_agents/llm_client.py.
 */

import { CopilotClient, approveAll } from "@github/copilot-sdk";

const MAX_RETRIES = 5;
const INITIAL_RETRY_DELAY = 5_000; // ms
const INTER_REQUEST_DELAY = 2_000; // ms
export const DEFAULT_MODEL = "gpt-4.1";

export async function getCopilotClient(): Promise<CopilotClient> {
  const client = new CopilotClient();
  return client;
}

export interface Message {
  role: "user" | "model" | "assistant";
  content: string;
}

export async function callCopilotWithRetry(
  client: CopilotClient,
  modelName: string,
  messages: Message[],
  systemPrompt: string,
  temperature = 0.0
): Promise<string> {
  // Rate limiting
  await Bun.sleep(INTER_REQUEST_DELAY);

  const fullPrompt = buildPrompt(systemPrompt, messages);

  let retries = 0;
  let delay = INITIAL_RETRY_DELAY;
  let lastError: Error | null = null;

  while (retries < MAX_RETRIES) {
    let session: Awaited<ReturnType<CopilotClient["createSession"]>> | null =
      null;
    try {
      session = await client.createSession({
        model: modelName,
        onPermissionRequest: approveAll,
      });

      const response = await session.sendAndWait({ prompt: fullPrompt });

      if (response?.data?.content) {
        return response.data.content.trim();
      } else {
        throw new Error("Empty response from Copilot SDK");
      }
    } catch (e: unknown) {
      const err = e instanceof Error ? e : new Error(String(e));
      const errorStr = err.message;

      if (errorStr.includes("Timeout") || errorStr.includes("timeout")) {
        console.log(
          `  [timeout] Copilot request timed out. Retrying in ${delay}ms... (attempt ${retries + 1}/${MAX_RETRIES})`
        );
      } else if (errorStr.includes("429") || errorStr.toLowerCase().includes("rate")) {
        console.log(
          `  [rate-limit] Retrying in ${delay}ms... (attempt ${retries + 1}/${MAX_RETRIES})`
        );
      } else {
        console.log(
          `  [error] Copilot API error (attempt ${retries + 1}/${MAX_RETRIES}): ${errorStr}`
        );
      }
      lastError = err;
    }

    retries++;
    if (retries < MAX_RETRIES) {
      await Bun.sleep(delay);
      delay *= 2;
    }
  }

  throw new Error(
    `Failed to call Copilot after ${MAX_RETRIES} retries. Last error: ${lastError?.message}`
  );
}

export function buildPrompt(systemPrompt: string, messages: Message[]): string {
  const parts: string[] = [systemPrompt.trim(), ""];

  for (const msg of messages) {
    if (msg.role === "user") {
      parts.push(msg.content);
      parts.push("");
    } else if (msg.role === "model" || msg.role === "assistant") {
      parts.push(`Assistant: ${msg.content}`);
      parts.push("");
    }
  }

  parts.push(
    "Now respond with either a search query (search: <query>) or your final answer (Explanation/Exact Answer/Confidence):"
  );

  return parts.join("\n");
}
