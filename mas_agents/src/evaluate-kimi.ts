/**
 * LLM-as-Judge evaluator using Kimi K2.5 via NVIDIA NIM endpoint.
 *
 * Uses native fetch() to call the OpenAI-compatible NVIDIA NIM API.
 *
 * Usage:
 *   bun run src/evaluate-kimi.ts \
 *     --input-dir runs/bm25/single_gpt_4.1 \
 *     --api-key nvapi-YOUR_KEY_HERE
 */

import { parseArgs } from "util";
import { resolve, join } from "path";
import {
  existsSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "fs";
import { GRADER_TEMPLATE } from "./prompts.js";

const NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1";
const DEFAULT_MODEL = "moonshotai/kimi-k2.5";

interface GroundTruth {
  question: string;
  answer: string;
}

interface JudgeParsed {
  extracted_final_answer: string | null;
  reasoning: string | null;
  correct: string | null;
  confidence: number | null;
  parse_error: boolean;
}

interface PerQueryMetric {
  query_id: string;
  status: string;
  correct: string;
  confidence: number;
  reasoning?: string | null;
  extracted_answer?: string | null;
  parse_error?: boolean;
  tool_calls?: Record<string, number>;
}

function loadGroundTruth(jsonlPath: string): Map<string, GroundTruth> {
  const gt = new Map<string, GroundTruth>();
  const content = readFileSync(jsonlPath, "utf-8");
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const obj = JSON.parse(trimmed);
    gt.set(String(obj.query_id), {
      question: obj.query,
      answer: obj.answer,
    });
  }
  return gt;
}

async function callKimiJudge(
  apiKey: string,
  prompt: string,
  model: string = DEFAULT_MODEL,
  maxTokens = 4096
): Promise<string> {
  const resp = await fetch(`${NVIDIA_NIM_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature: 1.0,
      top_p: 1.0,
      extra_body: { chat_template_kwargs: { thinking: true } },
    }),
  });

  if (!resp.ok) {
    throw new Error(`NVIDIA NIM API error: ${resp.status} ${resp.statusText}`);
  }

  const data = (await resp.json()) as any;
  return data.choices?.[0]?.message?.content?.trim() ?? "";
}

function parseJudgeResponse(judgeResponse: string): JudgeParsed {
  const result: JudgeParsed = {
    extracted_final_answer: null,
    reasoning: null,
    correct: null,
    confidence: null,
    parse_error: false,
  };

  if (!judgeResponse) {
    result.parse_error = true;
    return result;
  }

  // Extract extracted_final_answer
  let answerMatch =
    judgeResponse.match(
      /\*\*extracted_final_answer:\*\*\s*(.*?)(?:\n|$)/i
    ) ??
    judgeResponse.match(
      /\*\*extracted_final_answer\*\*:\s*(.*?)(?:\n|$)/i
    ) ??
    judgeResponse.match(/extracted_final_answer:\s*(.*?)(?:\n|$)/i);
  if (answerMatch) {
    result.extracted_final_answer = answerMatch[1].trim();
  }

  // Extract reasoning
  let reasoningMatch =
    judgeResponse.match(
      /\*\*reasoning:\*\*\s*(.*?)(?:\n\*\*correct|\ncorrect:|$)/is
    ) ??
    judgeResponse.match(
      /reasoning:\s*(.*?)(?:\n\*\*correct|\ncorrect:|$)/is
    );
  if (reasoningMatch) {
    result.reasoning = reasoningMatch[1].trim();
  }

  // Extract correct (yes/no)
  let correctMatch =
    judgeResponse.match(/\*\*correct:\*\*\s*(yes|no)/i) ??
    judgeResponse.match(/correct:\s*(yes|no)/i);
  if (correctMatch) {
    result.correct = correctMatch[1].trim().toLowerCase();
  } else {
    result.parse_error = true;
  }

  // Extract confidence
  let confMatch =
    judgeResponse.match(/\*\*confidence:\*\*\s*(\d+)/i) ??
    judgeResponse.match(/confidence:\s*(\d+)/i);
  if (confMatch) {
    result.confidence = parseInt(confMatch[1]);
  }

  return result;
}

function extractFinalOutput(resultData: any): string {
  const results = resultData.result ?? [];
  for (let i = results.length - 1; i >= 0; i--) {
    if (results[i].type === "output_text" && results[i].output) {
      return results[i].output;
    }
  }
  return "";
}

async function main() {
  const { values } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
      "input-dir": { type: "string" },
      "ground-truth": {
        type: "string",
        default: "data/browsecomp_plus_decrypted.jsonl",
      },
      "api-key": { type: "string" },
      model: { type: "string", default: DEFAULT_MODEL },
      "output-file": { type: "string" },
      "max-retries": { type: "string", default: "3" },
    },
    strict: true,
  });

  const inputDir = values["input-dir"];
  if (!inputDir) {
    console.error("Error: --input-dir is required");
    process.exit(1);
  }

  // API key
  const apiKey = values["api-key"] ?? process.env.NVIDIA_API_KEY;
  if (!apiKey) {
    console.error(
      "Error: NVIDIA API key required. Use --api-key or set NVIDIA_API_KEY env var."
    );
    process.exit(1);
  }

  const model = values.model!;
  const maxRetries = parseInt(values["max-retries"]!);

  // Load ground truth
  const gtPath = values["ground-truth"]!;
  if (!existsSync(gtPath)) {
    console.error(`Error: Ground truth file not found: ${gtPath}`);
    process.exit(1);
  }
  const gt = loadGroundTruth(gtPath);
  console.log(`Loaded ${gt.size} ground truth entries`);

  // Load run results
  if (!existsSync(inputDir)) {
    console.error(`Error: Input directory not found: ${inputDir}`);
    process.exit(1);
  }

  const runFiles = readdirSync(inputDir)
    .filter((f) => f.startsWith("run_") && f.endsWith(".json"))
    .sort();
  console.log(`Found ${runFiles.length} result files in ${inputDir}`);

  // Evaluate
  const perQueryMetrics: PerQueryMetric[] = [];
  let correctCount = 0;
  let totalEvaluated = 0;
  const confidences: number[] = [];
  const correctnessList: number[] = [];

  for (let idx = 0; idx < runFiles.length; idx++) {
    const rf = runFiles[idx];
    const progress = `[${idx + 1}/${runFiles.length}]`;

    let runData: any;
    try {
      runData = JSON.parse(readFileSync(join(inputDir, rf), "utf-8"));
    } catch (e) {
      console.log(`  [warn] Could not read ${rf}: ${e}`);
      continue;
    }

    const qid = String(runData.query_id ?? "");
    const status = runData.status ?? "";

    if (!gt.has(qid)) {
      console.log(`  [warn] Query ID ${qid} not found in ground truth, skipping`);
      continue;
    }

    // Failed runs count as incorrect
    if (status !== "completed") {
      perQueryMetrics.push({
        query_id: qid,
        status,
        correct: "no",
        confidence: 0,
        reasoning: "Run did not complete",
      });
      correctnessList.push(0);
      confidences.push(0);
      totalEvaluated++;
      continue;
    }

    const agentOutput = extractFinalOutput(runData);
    if (!agentOutput) {
      perQueryMetrics.push({
        query_id: qid,
        status: "no_output",
        correct: "no",
        confidence: 0,
        reasoning: "No output text found in result",
      });
      correctnessList.push(0);
      confidences.push(0);
      totalEvaluated++;
      continue;
    }

    // Call judge
    const gtEntry = gt.get(qid)!;
    const judgePrompt = GRADER_TEMPLATE(
      gtEntry.question,
      agentOutput,
      gtEntry.answer
    );

    let judgeResponse = "";
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        judgeResponse = await callKimiJudge(apiKey, judgePrompt, model);
        break;
      } catch (e) {
        if (attempt < maxRetries - 1) {
          console.log(`  [retry] Judge call for ${qid} failed: ${e}`);
        } else {
          console.error(
            `  [error] Judge call for ${qid} failed after ${maxRetries} attempts: ${e}`
          );
        }
      }
    }

    const parsed = parseJudgeResponse(judgeResponse);
    const isCorrect = parsed.correct === "yes";

    if (isCorrect) {
      correctCount++;
      correctnessList.push(1);
    } else {
      correctnessList.push(0);
    }

    const confidence = parsed.confidence ?? 100;
    confidences.push(confidence);
    totalEvaluated++;

    perQueryMetrics.push({
      query_id: qid,
      status: "evaluated",
      correct: parsed.correct ?? "unknown",
      confidence,
      extracted_answer: parsed.extracted_final_answer,
      reasoning: parsed.reasoning,
      parse_error: parsed.parse_error,
      tool_calls: runData.tool_call_counts ?? {},
    });

    console.log(
      `  ${progress} [${qid}] correct=${parsed.correct} confidence=${confidence}`
    );
  }

  // Compute aggregate metrics
  const accuracy = totalEvaluated > 0 ? correctCount / totalEvaluated : 0;
  const meanConfidence =
    confidences.length > 0
      ? confidences.reduce((a, b) => a + b, 0) / confidences.length / 100
      : 0;
  const calibrationError = Math.abs(meanConfidence - accuracy) * 100;

  const searchCounts = perQueryMetrics
    .filter((m) => m.tool_calls)
    .map((m) => m.tool_calls?.search ?? 0);
  const avgSearches =
    searchCounts.length > 0
      ? searchCounts.reduce((a, b) => a + b, 0) / searchCounts.length
      : 0;

  // Build summary
  const summary = {
    LLM: inputDir,
    Judge: model,
    "Accuracy (%)": Math.round(accuracy * 10000) / 100,
    "Total Evaluated": totalEvaluated,
    Correct: correctCount,
    avg_tool_stats: { search: Math.round(avgSearches * 100) / 100 },
    "Calibration Error (%)": Math.round(calibrationError * 100) / 100,
    Retriever: "BM25",
    "Evaluation Date": new Date().toISOString().split("T")[0],
    per_query_metrics: perQueryMetrics,
  };

  // Save
  const outputFile =
    values["output-file"] ?? join(inputDir, "eval_summary.json");
  writeFileSync(outputFile, JSON.stringify(summary, null, 2), "utf-8");

  console.log(`\n${"=".repeat(60)}`);
  console.log(`Evaluation Complete`);
  console.log(
    `  Accuracy:          ${(accuracy * 100).toFixed(1)}% (${correctCount}/${totalEvaluated})`
  );
  console.log(`  Calibration Error: ${calibrationError.toFixed(2)}%`);
  console.log(`  Avg Searches:      ${avgSearches.toFixed(1)}`);
  console.log(`  Summary saved to:  ${outputFile}`);
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(1);
});
