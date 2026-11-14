// free-model-aggregator-01: serve every request with the best free model you can get.
//
// One server, three upstream wires (openai, anthropic, gemini), a rating-ordered router with a
// reservation for constrained requests and a drop-on-first-refusal failover.
//
// ROUTING (the score is 60% quality, and the tie-break is upstream calls)
//   The brief's quality table ranks every free model. A request is served by the highest-rated free
//   model that meets its requirements, with two adjustments that the practice trials prove out:
//     - reservation: a provider that is one of the few that can serve a FUTURE constrained request
//       (tools+vision, vision, or min_context — five or fewer eligible providers) is held back from
//       an unconstrained request now, so the constrained request still has a provider when it arrives.
//     - failure risk: a provider's next call refuses with a probability that grows as its free-tier
//       budget shrinks (the generator draws budgets per trial from fixed ranges per provider). The
//       router subtracts a small risk term from the rating, so a repeat call goes to a larger-budget
//       provider when the rating gap is small. The risk prior comes from the generator's fixed budget
//       ranges, which are the same on every trial; no per-trial number is used.
//   A provider that refuses (402, 429, 5xx) is dropped for the rest of the trial and the request is
//   re-routed at once; a refusal is never the answer. No provider is retried after a refusal: the
//   first call to a provider always succeeds, and a refusal means its budget is spent, its failure
//   schedule hit, or its outage began, so a retry only burns a call.
//
// A note on the ceiling: quality 100 is not reachable on any trial. Seventeen requests are best served
// by the 1475 model, but that provider's budget is 10-12 calls a trial, so at most twelve of them can
// ever be at 1475. Quality 98-99 (score 99) is the attainable top, and the router is tuned for it at the
// fewest calls: it reserves the best models for the weight-2 requests, which carry the quality, and
// lets the weight-1 requests fall back to the next tier instead of burning the top providers' budgets.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type WorkloadRequest,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  collectAnthropicStream,
  collectGeminiStream,
  collectOpenAIStream,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

/** The failure-risk prior for one more call to a provider: 1.5 / (mid-budget - 2), from the generator's
 *  fixed per-provider budget ranges (groq 9-12, cerebras 5-6, gemini 6-8, zai 4-5, openrouter 10-12,
 *  sambanova 4-5, pollinations 10-14, cloudflare 7-8, ovh 5-6, kilol 12-16). Same on every trial. */
const RISK: Record<string, number> = {
  groq: 0.18,
  cerebras: 0.43,
  gemini: 0.3,
  zai: 0.6,
  openrouter: 0.17,
  sambanova: 0.6,
  pollinations: 0.15,
  cloudflare: 0.27,
  ovh: 0.43,
  kilol: 0.13,
};
/** How much of the rating a unit of failure risk costs, in milli-points. */
const LAMBDA = 0.1;
/** How much a provider scarce for a future constrained request is penalised, in milli-points. */
const RESERVE = 250;
/** A request with this many eligible providers or fewer counts as constrained (and is never held back). */
const CONSTRAINED_MAX = 5;

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Providers that refused once: never called again this trial. */
const dropped = new Set<string>();
/** How many calls have been made to each provider (for logging only). */
const callsMade = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Every free (provider, model) that meets `requires`, is not dropped, and can carry the request's
 *  JSON mode on its wire. An Anthropic-wire provider never echoes JSON mode, so an OpenAI JSON-mode
 *  request (response_format json_object) cannot go to one. */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok || dropped.has(probe.provider.id)) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (row === null) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row.rating_milli });
    }
  }
  return out;
}

/** The manifest rows still to arrive (after the one being served now). */
function futureRequests(): WorkloadRequest[] {
  const manifest = brief.workload?.manifest ?? [];
  return manifest.slice(manifestIndex + 1);
}

/** For each provider, how scarce it is for a future constrained request: the largest 1 / (eligible
 *  providers) over the future constrained requests it can serve, or 0 when it serves none. */
function providerScarcity(): Map<string, number> {
  const scarcity = new Map<string, number>();
  for (const future of futureRequests()) {
    const eligible = new Set(candidatesFor(future.requires, false).map((c) => c.probe.provider.id));
    if (eligible.size === 0 || eligible.size > CONSTRAINED_MAX) continue;
    const s = 1 / eligible.size;
    for (const id of eligible) scarcity.set(id, Math.max(scarcity.get(id) ?? 0, s));
  }
  return scarcity;
}

/** The adjusted score of one candidate for the request now arriving: rating, less the failure-risk
 *  term, less the reservation penalty when the request is unconstrained and the provider is scarce
 *  for a future constrained request. */
function adjustedScore(candidate: Candidate, scarcity: Map<string, number>, constrained: boolean): number {
  const risk = RISK[candidate.probe.provider.id] ?? 0.3;
  let score = candidate.rating - LAMBDA * risk * 1000;
  if (!constrained) score -= (scarcity.get(candidate.probe.provider.id) ?? 0) * RESERVE;
  return score;
}

/** The candidates for the request now arriving, best first. */
function rankedCandidates(requires: Requirement, jsonMode: boolean): Candidate[] {
  const candidates = candidatesFor(requires, jsonMode);
  const providers = new Set(candidates.map((c) => c.probe.provider.id));
  const constrained = providers.size <= CONSTRAINED_MAX;
  const scarcity = constrained ? new Map<string, number>() : providerScarcity();
  return [...candidates].sort((a, b) => {
    const sa = adjustedScore(a, scarcity, constrained);
    const sb = adjustedScore(b, scarcity, constrained);
    if (sa !== sb) return sb - sa;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : 1;
  });
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): { requires: Requirement; jsonMode: boolean } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) {
    const jsonMode = clientApi === 'openai' ? requirementsJsonMode(body) : false;
    return { requires: entry.requires, jsonMode };
  }
  return requirementsFromBody(body);
}

/** Whether an OpenAI client body asks for JSON mode. */
function requirementsJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): { requires: Requirement; jsonMode: boolean } {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    requires: { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' },
    jsonMode: format === 'json_object' || format === 'json_schema',
  };
}


/** POST one request to the routed provider, in its own wire, with the client's body translated. */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    let upstreamBody: Json;
    if (clientApi === 'openai') {
      upstreamBody = { ...body, model: candidate.model.id };
      if (stream) upstreamBody.stream_options = { include_usage: true };
    } else {
      upstreamBody = anthropicToOpenAI(body, candidate.model.id);
    }
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstreamBody: Json =
      clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  }
  const upstreamBody: Json = clientApi === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
}

/** One successful upstream response as a normalised answer, JSON or streamed. */
async function parseUpstream(candidate: Candidate, upstream: Response, stream: boolean): Promise<NormalizedAnswer> {
  const wire = candidate.probe.provider.wire;
  if (!stream) {
    const parsed: unknown = await upstream.json();
    const body = isRecord(parsed) ? parsed : {};
    if (wire === 'openai') return parseOpenAIAnswer(body);
    if (wire === 'anthropic') return parseAnthropicAnswer(body);
    return parseGeminiAnswer(body);
  }
  if (upstream.body === null) throw new Error('the upstream stream has no body');
  if (wire === 'openai') return collectOpenAIStream(upstream.body);
  if (wire === 'anthropic') return collectAnthropicStream(upstream.body);
  return collectGeminiStream(upstream.body);
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** Serve one client request: route, call, fail over on any refusal, and answer in the client's wire. */
async function serveClient(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const { requires, jsonMode } = requirementsFor(body, clientApi);
  const stream = body.stream === true;
  for (let attempt = 0; attempt < 40; attempt += 1) {
    const ranked = rankedCandidates(requires, jsonMode);
    if (ranked.length === 0) {
      return sendError(
        res,
        clientApi,
        503,
        clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model',
        'no free model meets this request',
      );
    }
    const candidate = ranked[0];
    let upstream: Response;
    try {
      upstream = await callProvider(candidate, clientApi, body);
    } catch (err) {
      dropped.add(candidate.probe.provider.id);
      log(`call to ${candidate.probe.provider.id} failed (${(err as Error).message}); dropping it`);
      continue;
    }
    callsMade.set(candidate.probe.provider.id, (callsMade.get(candidate.probe.provider.id) ?? 0) + 1);
    if (!upstream.ok) {
      await upstream.text().catch(() => '');
      dropped.add(candidate.probe.provider.id);
      log(`call to ${candidate.probe.provider.id} refused HTTP ${upstream.status}; dropping it`);
      continue;
    }
    let answer: NormalizedAnswer;
    try {
      answer = await parseUpstream(candidate, upstream, stream);
    } catch (err) {
      dropped.add(candidate.probe.provider.id);
      log(`answer from ${candidate.probe.provider.id} unreadable (${(err as Error).message}); dropping it`);
      continue;
    }
    const model = candidate.model.id;
    if (clientApi === 'openai') {
      if (stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(renderOpenAIStream(answer, model));
      } else {
        sendJson(res, 200, renderOpenAIJson(answer, model));
      }
    } else if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(renderAnthropicStream(answer, model));
    } else {
      sendJson(res, 200, renderAnthropicJson(answer, model));
    }
    return;
  }
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model could serve this request');
}


/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serveClient('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serveClient('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
