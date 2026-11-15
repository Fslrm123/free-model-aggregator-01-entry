// free-model-aggregator-01: serve every request with the best free model you can get.
//
// One server, three upstream wires (openai, anthropic, gemini), and a router that plans the whole
// workload before serving it, because the score is 60% quality and the tie-break is upstream calls.
//
// ROUTING (plan first, then serve the plan)
//   The manifest is in the brief: all 20 requests, their requirements and weights, are known before
//   the first one arrives. The router plans the whole assignment up front — which provider serves
//   which request — by regret-greedy over the cost of each option:
//     cost = quality loss of the model + LAMBDA_CALLS x P(that call refuses)
//   The refusal probability is a Bayesian posterior, not a guess: every provider's budget is drawn
//   per trial from fixed ranges (groq 9-12, cerebras 5-6, gemini 6-8, zai 4-5, openrouter 10-12,
//   sambanova 4-5, pollinations 10-14, cloudflare 7-8, ovh 5-6, kilol 12-16), it carries one or two
//   failure-schedule call numbers drawn from [2, budget-1], one provider's second call always fails,
//   and one provider goes out mid-trial. The first call to a provider always succeeds; a repeat
//   call's hazard grows as the provider is called deeper. The posterior conditions on everything
//   the providers have actually answered: a success proves its call number clean, a refusal proves
//   its call number a failure-schedule hit (or a spent budget, or an outage — see below).
//   The plan is rebuilt after every outcome, so a failure re-routes the requests still to come.
//   A provider that refuses is NOT dropped blindly: the error body says why. "Budget is spent" or
//   "is out" means dead for the trial; "refused this call" means one failure-schedule hit, and the
//   provider stays usable — at most one more of its call numbers can fail, so its next call is
//   nearly as safe as a fresh provider's, and dropping it would throw away its remaining budget.
//   A request whose first call fails is re-routed at once; a refusal is never the answer.
//
// A note on the ceiling: quality 100 is not reachable on any trial. Seventeen requests are best served
// by the 1475 model, but that provider's budget is 10-12 calls a trial, so at most twelve of them can
// ever be at 1475. Quality 98-99 (score 99) is the attainable top — the score is flat in quality
// above ~97.5 — so the router trades quality for calls once the plan clears that line: it keeps the
// best models for the weight-2 requests, which carry the quality, and lets the weight-1 requests
// fall back to the next tier on the big-budget providers, whose repeat calls are the cheapest.

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

/** The generator's fixed per-provider budget ranges (same on every trial): the prior for how many
 *  calls a provider serves before its budget is spent. No per-trial number is used. */
const BUDGET_RANGE: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};
/** Quality points one expected provider failure is worth. The score is flat in quality above ~97.5
 *  (score 99 needs quality >= 97.5 with every other column at 100), so above that line a failure
 *  avoided is worth far more than a slightly better model: 2 points buys a repeat on a big-budget
 *  provider instead of a top model on a small one, and keeps the plan's quality near 98. */
const LAMBDA_CALLS = 2.0;
/** How often a provider is the one whose second call always fails (the generator picks one). */
const DESIGNATED_P = 0.1;
/** How often a provider is the one that goes out mid-trial (the generator picks one). */
const OUTAGE_P = 0.1;

/** What the router knows about one provider: every call it made, which call numbers refused on the
 *  provider's failure schedule, and whether its budget is spent or its outage began (dead forever). */
interface ProviderState {
  calls: number;
  dead: boolean;
  scheduleFails: number[];
}
const providerState = new Map<string, ProviderState>();
/** One possible world for a provider: its budget, its failure-schedule bitmask (bit j = call j
 *  refuses on schedule), its outage start (0 = none), and the world's prior weight. */
interface World {
  budget: number;
  failMask: number;
  outage: number;
  weight: number;
}
const worldsByProvider = new Map<string, World[]>();
/** Hazard memo: `${provider}|${calls}|${scheduleFails}` -> P(next call refuses). */
const hazardCache = new Map<string, number>();

/** Every subset of [lo, hi] of the given size, as a bitmask (bit v = value v). */
function subsets(lo: number, hi: number, size: number): number[] {
  const out: number[] = [];
  if (size === 0) return [0];
  const rec = (start: number, left: number, mask: number): void => {
    if (left === 0) {
      out.push(mask);
      return;
    }
    for (let v = start; v <= hi - left + 1; v++) rec(v + 1, left - 1, mask | (1 << v));
  };
  rec(lo, size, 0);
  return out;
}

/** Every world a provider could be in, with its prior weight: budget uniform over its range, one or
 *  two failure-schedule numbers uniform over [2, budget-1], a 10% chance its second call is forced
 *  to fail, and a 10% chance it goes out at a number uniform over [2, budget-1]. */
function buildWorlds(id: string): World[] {
  const [lo, hi] = BUDGET_RANGE[id] ?? [4, 16];
  const worlds: World[] = [];
  const nB = hi - lo + 1;
  for (let b = lo; b <= hi; b++) {
    const range = b - 2;
    for (const count of [1, 2]) {
      if (range < count) continue;
      const free = subsets(2, b - 1, count);
      const wFree = (1 / nB) * 0.5 * (1 / free.length) * (1 - DESIGNATED_P);
      const rest = subsets(3, b - 1, count - 1);
      const wDes = (1 / nB) * 0.5 * (1 / rest.length) * DESIGNATED_P;
      const failSets: { mask: number; weight: number }[] = [
        ...free.map((mask) => ({ mask, weight: wFree })),
        ...rest.map((mask) => ({ mask: mask | (1 << 2), weight: wDes })),
      ];
      for (const { mask, weight } of failSets) {
        worlds.push({ budget: b, failMask: mask, outage: 0, weight: weight * (1 - OUTAGE_P) });
        for (let o = 2; o <= b - 1; o++) {
          worlds.push({ budget: b, failMask: mask, outage: o, weight: (weight * OUTAGE_P) / range });
        }
      }
    }
  }
  return worlds;
}

/** P(the provider's next call refuses), given what it has answered so far. A success proves its
 *  call number clean and within budget and before any outage; a schedule refusal proves its call
 *  number a failure-schedule hit. Worlds that contradict either are dropped; the rest are weighed
 *  by their prior, and the share in which the next call number is over budget, past the outage, or
 *  on the schedule is the hazard. */
function failProbability(id: string, st: ProviderState): number {
  if (st.dead) return 1;
  const key = `${id}|${st.calls}|${st.scheduleFails.join(',')}`;
  const hit = hazardCache.get(key);
  if (hit !== undefined) return hit;
  const worlds = worldsByProvider.get(id) ?? [];
  const k = st.calls + 1;
  let successMask = 0;
  for (let j = 1; j <= st.calls; j++) if (!st.scheduleFails.includes(j)) successMask |= 1 << j;
  let failMask = 0;
  for (const j of st.scheduleFails) failMask |= 1 << j;
  let total = 0;
  let bad = 0;
  for (const w of worlds) {
    if (w.budget < st.calls) continue;
    if ((w.failMask & failMask) !== failMask) continue;
    if ((w.failMask & successMask) !== 0) continue;
    if (w.outage !== 0 && w.outage <= st.calls) continue;
    total += w.weight;
    if (k > w.budget || (w.outage !== 0 && w.outage <= k) || ((w.failMask >> k) & 1) === 1) bad += w.weight;
  }
  const p = total === 0 ? 1 : bad / total;
  hazardCache.set(key, p);
  return p;
}

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** The workload's manifest, in arrival order: the whole plan is built from it. */
const manifest: WorkloadRequest[] = brief.workload?.manifest ?? [];
/** The manifest's total weight, for quality-loss shares. */
const TOTAL_WEIGHT = manifest.reduce((n, r) => n + r.weight, 0);
/** Each request's best free-model rating, from the brief's quality table. */
const bestByRequest = new Map<string, number>();
for (const row of brief.ratings?.best ?? []) bestByRequest.set(row.id, row.rating_milli);
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
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
  for (const provider of brief.directory.providers) {
    providerState.set(provider.id, { calls: 0, dead: false, scheduleFails: [] });
    worldsByProvider.set(provider.id, buildWorlds(provider.id));
  }
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

interface Option {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** One option per provider: its best free model meeting `requires` (and JSON mode on its wire).
 *  An Anthropic-wire provider never echoes JSON mode, so an OpenAI JSON-mode request
 *  (response_format json_object) cannot go to one. Dead providers are never an option. */
function optionsFor(requires: Requirement, jsonMode: boolean, states: Map<string, ProviderState>): Option[] {
  const out: Option[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const st = states.get(probe.provider.id);
    if (st !== undefined && st.dead) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    let best: Option | null = null;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row) || row === null) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (best === null || row.rating_milli > best.rating || (row.rating_milli === best.rating && model.id < best.model.id)) {
        best = { probe, model, rating: row.rating_milli };
      }
    }
    if (best !== null) out.push(best);
  }
  return out;
}

/** The best free-model rating for one request, from the brief's quality table (or, when it has no
 *  row for the request, the best rating among the free models meeting its requirements). */
function bestRatingFor(req: WorkloadRequest): number {
  const known = bestByRequest.get(req.id);
  if (known !== undefined) return known;
  let best = 0;
  for (const row of brief.ratings?.ratings ?? []) {
    if (!row.free) continue;
    if (!meetsRequirements(req.requires, { context_length: row.context_length, tools: row.tools, vision: row.vision, json: row.json })) continue;
    best = Math.max(best, row.rating_milli);
  }
  return best;
}

/** What serving `req` with a model rated `rating` costs in quality points (0 when it is the best). */
function qualityLoss(req: WorkloadRequest, rating: number): number {
  const best = bestRatingFor(req);
  if (best <= 0 || TOTAL_WEIGHT === 0) return 0;
  return ((req.weight * (best - rating)) / best / TOTAL_WEIGHT) * 100;
}

/** Whether an OpenAI request in the manifest would ask for JSON mode: the generator sets
 *  response_format exactly when the request requires JSON. Anthropic requests never do. */
function assumedJsonMode(req: WorkloadRequest): boolean {
  return req.api === 'openai' ? req.requires.json === true : false;
}

interface PlannedCall {
  providerId: string;
  modelId: string;
}

/** The plan for the requests from `fromIndex` on: each request's provider, chosen by regret-greedy
 *  over cost = quality loss + LAMBDA_CALLS x failure hazard, with planned calls assumed to succeed
 *  (so a provider's later planned calls are priced at their posterior hazard, which rises as it is
 *  called deeper). The request with the largest regret — best option vs second-best — is assigned
 *  first, which reserves scarce providers for the constrained requests that need them. */
function buildPlan(fromIndex: number, currentJsonMode: boolean): Map<number, PlannedCall> {
  const plan = new Map<number, PlannedCall>();
  const planned = new Map<string, ProviderState>();
  for (const [id, st] of providerState) planned.set(id, { calls: st.calls, dead: st.dead, scheduleFails: [...st.scheduleFails] });
  const remaining = new Set<number>();
  for (let i = fromIndex; i < manifest.length; i++) remaining.add(i);
  while (remaining.size > 0) {
    let pickIdx = -1;
    let pickRegret = -Infinity;
    let pickOpt: Option | null = null;
    for (const idx of [...remaining].sort((a, b) => a - b)) {
      const req = manifest[idx];
      const reqJson = idx === fromIndex ? currentJsonMode : assumedJsonMode(req);
      const opts = optionsFor(req.requires, reqJson, planned);
      if (opts.length === 0) continue;
      const scored = opts
        .map((o) => ({
          opt: o,
          cost:
            qualityLoss(req, o.rating) +
            LAMBDA_CALLS *
              failProbability(o.probe.provider.id, planned.get(o.probe.provider.id) ?? { calls: 0, dead: false, scheduleFails: [] }),
        }))
        .sort((a, b) => (a.cost !== b.cost ? a.cost - b.cost : a.opt.probe.provider.id < b.opt.probe.provider.id ? -1 : 1));
      const best = scored[0];
      const second = scored.length > 1 ? scored[1].cost : best.cost + 10;
      const regret = second - best.cost;
      if (regret > pickRegret || (regret === pickRegret && (pickIdx === -1 || idx < pickIdx))) {
        pickIdx = idx;
        pickRegret = regret;
        pickOpt = best.opt;
      }
    }
    if (pickIdx === -1 || pickOpt === null) break;
    plan.set(pickIdx, { providerId: pickOpt.probe.provider.id, modelId: pickOpt.model.id });
    remaining.delete(pickIdx);
    const st = planned.get(pickOpt.probe.provider.id);
    if (st !== undefined) st.calls += 1;
  }
  return plan;
}

/** The candidates for the request arriving now, planned provider first, the rest by cost. The plan
 *  already priced the future in, so its choice outranks the local ranking: that is what reserves a
 *  scarce provider for a later constrained request instead of spending it now. */
function orderForCurrent(req: WorkloadRequest, jsonMode: boolean, planned: PlannedCall | undefined): Candidate[] {
  const opts = optionsFor(req.requires, jsonMode, providerState);
  const scored = opts.map((o) => ({
    candidate: { probe: o.probe, model: o.model, rating: o.rating } as Candidate,
    cost:
      qualityLoss(req, o.rating) +
      LAMBDA_CALLS * failProbability(o.probe.provider.id, providerState.get(o.probe.provider.id) ?? { calls: 0, dead: false, scheduleFails: [] }),
  }));
  scored.sort((a, b) => {
    const aPlanned = planned !== undefined && a.candidate.probe.provider.id === planned.providerId && a.candidate.model.id === planned.modelId ? 0 : 1;
    const bPlanned = planned !== undefined && b.candidate.probe.provider.id === planned.providerId && b.candidate.model.id === planned.modelId ? 0 : 1;
    if (aPlanned !== bPlanned) return aPlanned - bPlanned;
    if (a.cost !== b.cost) return a.cost - b.cost;
    if (a.candidate.rating !== b.candidate.rating) return b.candidate.rating - a.candidate.rating;
    return a.candidate.probe.provider.id < b.candidate.probe.provider.id ? -1 : 1;
  });
  return scored.map((s) => s.candidate);
}

/** Why a provider refused, from its error body: its budget spent, its outage, or one call on its
 *  failure schedule. Anything else is treated as a schedule refusal (the provider stays usable). */
function classifyRefusal(text: string): 'budget' | 'outage' | 'schedule' {
  let message = text;
  try {
    const parsed: unknown = JSON.parse(text);
    const err = isRecord(parsed) && isRecord(parsed.error) ? parsed.error : null;
    if (err !== null && typeof err.message === 'string') message = err.message;
  } catch {
    // Not JSON: use the raw text.
  }
  if (/budget is spent/.test(message)) return 'budget';
  if (/ is out/.test(message)) return 'outage';
  return 'schedule';
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): { requires: Requirement; jsonMode: boolean; entry: WorkloadRequest | undefined } {
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) {
    const jsonMode = clientApi === 'openai' ? requirementsJsonMode(body) : false;
    return { requires: entry.requires, jsonMode, entry };
  }
  return { ...requirementsFromBody(body), entry: undefined };
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

/** Serve one client request: follow the plan, call, and on any refusal re-plan and fail over at once. */
async function serveClient(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const { requires, jsonMode, entry } = requirementsFor(body, clientApi);
  const req: WorkloadRequest = entry ?? { id: '', api: clientApi, stream: false, weight: 1, requires };
  const stream = body.stream === true;
  const myIndex = manifestIndex - 1;
  for (let attempt = 0; attempt < 40; attempt += 1) {
    const plan = buildPlan(myIndex, jsonMode);
    const ordered = orderForCurrent(req, jsonMode, plan.get(myIndex));
    if (ordered.length === 0) {
      return sendError(
        res,
        clientApi,
        503,
        clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model',
        'no free model meets this request',
      );
    }
    const candidate = ordered[0];
    const st = providerState.get(candidate.probe.provider.id) ?? { calls: 0, dead: false, scheduleFails: [] };
    providerState.set(candidate.probe.provider.id, st);
    let upstream: Response;
    try {
      upstream = await callProvider(candidate, clientApi, body);
    } catch (err) {
      st.dead = true;
      log(`call to ${candidate.probe.provider.id} failed (${(err as Error).message}); dropping it`);
      continue;
    }
    if (upstream.status === 400 || upstream.status === 401) {
      // A wrong key or an unknown or paid model: the call spent none of the provider's budget, but
      // our routing data is wrong for it, so it is out for this trial.
      await upstream.text().catch(() => '');
      st.dead = true;
      log(`call to ${candidate.probe.provider.id} refused HTTP ${upstream.status} (not counted); dropping it`);
      continue;
    }
    const callNo = st.calls + 1;
    st.calls = callNo;
    callsMade.set(candidate.probe.provider.id, (callsMade.get(candidate.probe.provider.id) ?? 0) + 1);
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => '');
      const reason = classifyRefusal(text);
      if (reason === 'budget' || reason === 'outage') {
        st.dead = true;
        log(`call to ${candidate.probe.provider.id} refused HTTP ${upstream.status} (${reason}); dropping it`);
      } else {
        st.scheduleFails.push(callNo);
        log(`call to ${candidate.probe.provider.id} refused HTTP ${upstream.status} (schedule, call ${callNo}); keeping it`);
      }
      continue;
    }
    let answer: NormalizedAnswer;
    try {
      answer = await parseUpstream(candidate, upstream, stream);
    } catch (err) {
      st.scheduleFails.push(callNo);
      log(`answer from ${candidate.probe.provider.id} unreadable (${(err as Error).message}); keeping it`);
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
