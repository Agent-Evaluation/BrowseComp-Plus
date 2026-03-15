/**
 * BM25 retriever bridge — spawns a long-lived Python process that serves
 * Pyserini BM25 searches over a JSONL stdin/stdout protocol.
 */

import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

export interface SearchResult {
  docid: string;
  score: number;
  snippet: string;
}

interface BridgeRequest {
  action: string;
  query?: string;
  k?: number;
  snippet_max_tokens?: number;
}

interface BridgeResponse {
  status?: string;
  results?: SearchResult[];
  error?: string;
}

export class BM25Bridge {
  private proc: Awaited<ReturnType<typeof Bun.spawn>> | null = null;
  private stdin: { write(data: string): number; flush(): Promise<void>; end(): void } | null = null;
  private reader: ReadableStreamDefaultReader<string> | null = null;
  private buffer = "";
  private ready = false;
  private indexPath: string;
  private k: number;
  private snippetMaxTokens: number;
  private pythonExe: string;

  constructor(opts: {
    indexPath: string;
    k?: number;
    snippetMaxTokens?: number;
    pythonExe?: string;
  }) {
    this.indexPath = opts.indexPath;
    this.k = opts.k ?? 5;
    this.snippetMaxTokens = opts.snippetMaxTokens ?? 512;
    this.pythonExe = opts.pythonExe ?? "uv";
  }

  async start(): Promise<void> {
    const bridgeScript = resolve(
      dirname(fileURLToPath(import.meta.url)),
      "..",
      "bm25_searcher.py"
    );

    const args =
      this.pythonExe === "uv"
        ? ["run", "python", bridgeScript, "--index-path", this.indexPath]
        : [bridgeScript, "--index-path", this.indexPath];

    this.proc = Bun.spawn([this.pythonExe, ...args], {
      stdin: "pipe",
      stdout: "pipe",
      stderr: "inherit",
      env: {
        ...process.env,
        JAVA_HOME:
          process.env.JAVA_HOME ??
          "C:\\Program Files\\Microsoft\\jdk-21.0.10.7-hotspot",
      },
      cwd: resolve(dirname(fileURLToPath(import.meta.url)), "..", ".."),
    });

    // Store typed stdin handle
    this.stdin = this.proc.stdin as unknown as { write(data: string): number; flush(): Promise<void>; end(): void };

    // Set up line reader from stdout using a manual text decoding transform
    const stdout = this.proc.stdout as ReadableStream<Uint8Array>;
    const textDecoder = new TextDecoder();
    const transform = new TransformStream<Uint8Array, string>({
      transform(chunk, controller) {
        controller.enqueue(textDecoder.decode(chunk, { stream: true }));
      },
      flush(controller) {
        const final = textDecoder.decode();
        if (final) controller.enqueue(final);
      },
    });
    const readable = stdout.pipeThrough(transform);
    this.reader = readable.getReader();

    // Wait for "ready" signal
    const readyMsg = await this.readLine();
    const parsed = JSON.parse(readyMsg) as BridgeResponse;
    if (parsed.status !== "ready") {
      throw new Error(`BM25 bridge did not signal ready: ${readyMsg}`);
    }
    this.ready = true;
    console.log("BM25 bridge ready");
  }

  async search(query: string): Promise<SearchResult[]> {
    if (!this.ready || !this.proc) {
      throw new Error("BM25 bridge not started");
    }

    const req: BridgeRequest = {
      action: "search",
      query,
      k: this.k,
      snippet_max_tokens: this.snippetMaxTokens,
    };

    this.stdin!.write(JSON.stringify(req) + "\n");
    await this.stdin!.flush();

    const responseLine = await this.readLine();
    const resp = JSON.parse(responseLine) as BridgeResponse;

    if (resp.error) {
      throw new Error(`BM25 bridge error: ${resp.error}`);
    }

    return resp.results ?? [];
  }

  formatResults(results: SearchResult[]): string {
    if (results.length === 0) return "No results found.";
    return results
      .map(
        (r) => `[DocID: ${r.docid}] (score: ${r.score.toFixed(4)})\n${r.snippet}`
      )
      .join("\n\n---\n\n");
  }

  async stop(): Promise<void> {
    if (this.proc) {
      try {
        this.stdin!.write(JSON.stringify({ action: "quit" }) + "\n");
        await this.stdin!.flush();
        this.stdin!.end();
      } catch {
        // ignore
      }
      this.proc.kill();
      this.proc = null;
    }
    this.ready = false;
  }

  private async readLine(): Promise<string> {
    if (!this.reader) throw new Error("No reader");

    while (true) {
      const newlineIdx = this.buffer.indexOf("\n");
      if (newlineIdx !== -1) {
        const line = this.buffer.slice(0, newlineIdx);
        this.buffer = this.buffer.slice(newlineIdx + 1);
        return line;
      }

      const { done, value } = await this.reader.read();
      if (done) throw new Error("BM25 bridge stdout closed unexpectedly");
      this.buffer += value;
    }
  }
}
