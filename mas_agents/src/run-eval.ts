/**
 * BrowseComp-Plus multi-agent evaluation runner.
 *
 * Runs a single architecture against queries and saves JSON outputs
 * in the BrowseComp-Plus format under runs/<run_name>/.
 *
 * Usage:
 *   bun run src/run-eval.ts \
 *     --architecture single \
 *     --model gpt-4.1 \
 *     --index-path indexes/bm25/ \
 *     --limit 1
 */

import { parseArgs } from "util";
import { resolve, join } from "path";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "fs";
import { CopilotClient } from "@github/copilot-sdk";
import { BM25Bridge } from "./bm25-bridge.js";
import { CopilotSingleAgent } from "./single-agent.js";
import {
  CopilotIndependentAgent,
  CopilotCentralizedAgent,
  CopilotDecentralizedAgent,
  CopilotHybridAgent,
} from "./multi-agents.js";
import type { AgentOutput } from "./base-agent.js";

const ARCHITECTURES = [
  "single",
  "independent",
  "centralized",
  "decentralized",
  "hybrid",
] as const;

type Architecture = (typeof ARCHITECTURES)[number];

function createAgent(
  arch: Architecture,
  model: string,
  client: CopilotClient,
  retriever: BM25Bridge,
  numAgents: number
) {
  switch (arch) {
    case "single":
      return new CopilotSingleAgent(model, client, retriever);
    case "independent":
      return new CopilotIndependentAgent(model, client, retriever, numAgents);
    case "centralized":
      return new CopilotCentralizedAgent(model, client, retriever, numAgents, 5);
    case "decentralized":
      return new CopilotDecentralizedAgent(model, client, retriever, numAgents, 3);
    case "hybrid":
      return new CopilotHybridAgent(model, client, retriever, numAgents, 1, 1);
  }
}

async function main() {
  const { values } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
      architecture: { type: "string", short: "a" },
      model: { type: "string", short: "m", default: "gpt-4.1" },
      query: { type: "string", short: "q", default: "topics-qrels/queries.tsv" },
      "index-path": { type: "string", short: "i", default: "indexes/bm25/" },
      "output-dir": { type: "string", short: "o" },
      k: { type: "string", default: "5" },
      "snippet-max-tokens": { type: "string", default: "512" },
      "num-agents": { type: "string", default: "3" },
      limit: { type: "string", default: "0" },
    },
    strict: true,
  });

  const arch = values.architecture as Architecture;
  if (!arch || !ARCHITECTURES.includes(arch)) {
    console.error(
      `Error: --architecture must be one of: ${ARCHITECTURES.join(", ")}`
    );
    process.exit(1);
  }

  const model = values.model!;
  const queryPath = values.query!;
  const indexPath = values["index-path"]!;
  const k = parseInt(values.k!);
  const snippetMaxTokens = parseInt(values["snippet-max-tokens"]!);
  const numAgents = parseInt(values["num-agents"]!);
  const limit = parseInt(values.limit!);

  // Resolve output directory
  const modelSlug = model.replace(/\//g, "_").replace(/-/g, "_");
  const outputDir = values["output-dir"]
    ? resolve(values["output-dir"])
    : resolve(`runs/bm25/${arch}_${modelSlug}`);

  mkdirSync(outputDir, { recursive: true });

  // ── Load queries ────────────────────────────────────────────────────────
  if (!existsSync(queryPath)) {
    console.error(`Error: Query file not found: ${queryPath}`);
    process.exit(1);
  }

  const queryContent = readFileSync(queryPath, "utf-8");
  let queries: Array<[string, string]> = queryContent
    .trim()
    .split("\n")
    .map((line) => {
      const [id, ...rest] = line.split("\t");
      return [id.trim(), rest.join("\t").trim()] as [string, string];
    })
    .filter(([id, text]) => id && text);

  console.log(`Loaded ${queries.length} queries from ${queryPath}`);

  if (limit > 0) {
    queries = queries.slice(0, limit);
    console.log(`  Limited to ${limit} queries`);
  }

  // ── Resume: skip already-processed queries ──────────────────────────────
  const processedIds = new Set<string>();
  if (existsSync(outputDir)) {
    for (const file of readdirSync(outputDir)) {
      if (file.startsWith("run_") && file.endsWith(".json")) {
        try {
          const data = JSON.parse(
            readFileSync(join(outputDir, file), "utf-8")
          );
          if (data.query_id) processedIds.add(String(data.query_id));
        } catch {
          // skip
        }
      }
    }
  }

  const remaining = queries.filter(([qid]) => !processedIds.has(qid));
  console.log(
    `Processing ${remaining.length} remaining queries (skipping ${processedIds.size} already done)`
  );

  if (remaining.length === 0) {
    console.log("All queries already processed. Done.");
    return;
  }

  // ── Initialize BM25 bridge ──────────────────────────────────────────────
  console.log(`Initializing BM25 retriever from ${indexPath}...`);
  const retriever = new BM25Bridge({
    indexPath,
    k,
    snippetMaxTokens,
  });
  await retriever.start();

  // ── Initialize Copilot client ───────────────────────────────────────────
  console.log("Starting Copilot SDK client...");
  const client = new CopilotClient();

  // ── Create agent ────────────────────────────────────────────────────────
  const agent = createAgent(arch, model, client, retriever, numAgents);
  console.log(`Agent: ${agent.constructor.name} | Model: ${model}`);
  console.log(`Output: ${outputDir}`);
  console.log("=".repeat(60));

  // ── Process queries ─────────────────────────────────────────────────────
  let completed = 0;
  let failed = 0;

  for (let idx = 0; idx < remaining.length; idx++) {
    const [qid, qtext] = remaining[idx];
    const progress = `[${idx + 1}/${remaining.length}]`;

    try {
      const result: AgentOutput = await agent.runQuery(qid, qtext);

      // Save result
      const ts = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "Z");
      const filename = join(outputDir, `run_${ts}_${qid}.json`);
      writeFileSync(filename, JSON.stringify(result, null, 2), "utf-8");

      const status = result.status;
      const searchCount = result.tool_call_counts?.search ?? 0;
      console.log(`  ${progress} [${qid}] status=${status} | searches=${searchCount}`);

      if (status === "completed") completed++;
      else failed++;
    } catch (exc) {
      console.error(`  ${progress} [${qid}] FAILED: ${exc}`);
      failed++;

      const ts = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "Z");
      const filename = join(outputDir, `run_${ts}_${qid}.json`);
      writeFileSync(
        filename,
        JSON.stringify(
          {
            query_id: qid,
            tool_call_counts: {},
            status: `error: ${exc}`,
            retrieved_docids: [],
            result: [{ type: "output_text", output: "" }],
          },
          null,
          2
        ),
        "utf-8"
      );
    }
  }

  // ── Summary ─────────────────────────────────────────────────────────────
  const total = completed + failed;
  console.log("\n" + "=".repeat(60));
  console.log(`Done! ${completed}/${total} completed, ${failed}/${total} failed`);
  console.log(`Results saved to: ${outputDir}`);

  // Cleanup
  await retriever.stop();
  await client.stop();
  process.exit(0);
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(1);
});
