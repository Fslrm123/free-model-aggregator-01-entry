// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. The entry forwards each request
// to a provider on its own wire (openai, anthropic or gemini) and translates the answer back.
//
// The translations are built around one normalised answer — { text, inputTokens, outputTokens, stop } —
// parsed from any provider's JSON or SSE, then rendered into the wire the client used. Streaming
// responses are collected from the provider and re-emitted to the client; the providers are local and
// fast, and nothing in the trial measures within-request latency, so collecting first is safe and keeps
// the usage and stop reason exact. Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same provider answer give the same bytes on a replay. The two
// model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}


/** One request, reduced to the fields every wire carries. */
export interface Extracted {
  system: string;
  userText: string;
  /** An inline image (data URL) or a remote one, or null. */
  image: { mime: string; data: string } | { url: string } | null;
  tools: { name: string; description: string; parameters: Json }[];
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function splitDataUrl(url: string): { mime: string; data: string } | { url: string } {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (m !== null) return { mime: m[1], data: m[2] };
  return { url };
}

/** The shared fields of an OpenAI Chat Completions request body. */
export function extractOpenAI(body: Json): Extracted {
  let system = '';
  let userText = '';
  let image: Extracted['image'] = null;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
    } else if (raw.role === 'user') {
      userText = textOf(raw.content);
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
            image = splitDataUrl(block.image_url.url);
          }
        }
      }
    }
  }
  const tools: Extracted['tools'] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const stops: string[] = [];
  if (typeof body.stop === 'string') stops.push(body.stop);
  else if (Array.isArray(body.stop)) for (const s of body.stop) if (typeof s === 'string') stops.push(s);
  return {
    system,
    userText,
    image,
    tools,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops,
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** The shared fields of an Anthropic Messages request body. */
export function extractAnthropic(body: Json): Extracted {
  const system = textOf(body.system);
  let userText = '';
  let image: Extracted['image'] = null;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'user') continue;
    userText = textOf(raw.content);
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          image = { mime: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data };
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          image = { url: block.source.url };
        }
      }
    }
  }
  const tools: Extracted['tools'] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const stops: string[] = [];
  if (Array.isArray(body.stop_sequences)) for (const s of body.stop_sequences) if (typeof s === 'string') stops.push(s);
  return {
    system,
    userText,
    image,
    tools,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops,
    // An Anthropic request never carries JSON mode: its wire echoes json_mode false.
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** An OpenAI user content part (text + optional image) for the extracted request. */
function openAIUserParts(ex: Extracted): Json[] {
  const parts: Json[] = [{ type: 'text', text: ex.userText }];
  if (ex.image !== null) {
    if ('data' in ex.image) parts.push({ type: 'image_url', image_url: { url: `data:${ex.image.mime};base64,${ex.image.data}` } });
    else parts.push({ type: 'image_url', image_url: { url: ex.image.url } });
  }
  return parts;
}

/** An Anthropic user content block array (text + optional image) for the extracted request. */
function anthropicUserBlocks(ex: Extracted): Json[] {
  const blocks: Json[] = [{ type: 'text', text: ex.userText }];
  if (ex.image !== null) {
    if ('data' in ex.image) blocks.push({ type: 'image', source: { type: 'base64', media_type: ex.image.mime, data: ex.image.data } });
    else blocks.push({ type: 'image', source: { type: 'url', url: ex.image.url } });
  }
  return blocks;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`.
 *  JSON mode is NOT carried: the Anthropic wire echoes json_mode false, so the router must never send a
 *  JSON-mode OpenAI request to an Anthropic-wire provider. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const ex = extractOpenAI(body);
  const out: Json = { model, messages: [{ role: 'user', content: anthropicUserBlocks(ex) }] };
  if (ex.maxTokens !== null) out.max_tokens = ex.maxTokens;
  if (ex.system !== '') out.system = ex.system;
  if (ex.temperature !== null) out.temperature = ex.temperature;
  if (ex.topP !== null) out.top_p = ex.topP;
  if (ex.stops.length > 0) out.stop_sequences = ex.stops;
  if (ex.tools.length > 0) {
    out.tools = ex.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (ex.stream) out.stream = true;
  return out;
}

/** A Gemini generateContent request body for the extracted request (the model goes in the URL path). */
function geminiRequestBody(ex: Extracted): Json {
  const parts: Json[] = [{ text: ex.userText }];
  if (ex.image !== null) {
    if ('data' in ex.image) parts.push({ inlineData: { mimeType: ex.image.mime, data: ex.image.data } });
    else parts.push({ fileData: { fileUri: ex.image.url, mimeType: 'image/png' } });
  }
  const body: Json = { contents: [{ role: 'user', parts }] };
  if (ex.system !== '') body.systemInstruction = { parts: [{ text: ex.system }] };
  if (ex.tools.length > 0) {
    body.tools = [{ functionDeclarations: ex.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (ex.maxTokens !== null) config.maxOutputTokens = ex.maxTokens;
  if (ex.temperature !== null) config.temperature = ex.temperature;
  if (ex.topP !== null) config.topP = ex.topP;
  if (ex.stops.length > 0) config.stopSequences = ex.stops;
  if (ex.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGeminiRequest(body: Json): Json {
  return geminiRequestBody(extractOpenAI(body));
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGeminiRequest(body: Json): Json {
  return geminiRequestBody(extractAnthropic(body));
}

/** One provider answer, reduced to what either client wire needs. The providers in this trial always
 *  answer with the family-0 stop reason (stop / end_turn / STOP); `finish` is that reason in the
 *  OpenAI spelling. */
export interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  finish: string;
}

/** An OpenAI Chat Completions JSON answer as a normalised answer. */
export function parseOpenAIAnswer(body: Json): NormalizedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    finish: typeof choice.finish_reason === 'string' && choice.finish_reason !== '' ? choice.finish_reason : 'stop',
  };
}

/** An Anthropic Messages JSON answer as a normalised answer. */
export function parseAnthropicAnswer(body: Json): NormalizedAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    finish: finishReasonOf(body.stop_reason),
  };
}

/** A Gemini generateContent JSON answer as a normalised answer. */
export function parseGeminiAnswer(body: Json): NormalizedAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    finish: 'stop',
  };
}


/** Split one SSE text stream into its parsed `data:` payloads, in order. `[DONE]` ends the sequence. */
export async function* ssePayloads(stream: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const framesOf = function* (text: string): Generator<unknown> {
    for (const frame of text.split('\n\n')) {
      const lines = frame.split('\n').filter((l) => l.trim() !== '');
      if (lines.length === 0) continue;
      const dataLines = lines.filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trim());
      if (dataLines.length === 0) continue;
      const payload = dataLines.join('\n');
      if (payload === '[DONE]') return;
      try {
        yield JSON.parse(payload) as unknown;
      } catch {
        // A malformed frame: skip it.
      }
    }
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const payload of framesOf(frame)) yield payload;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const payload of framesOf(buffer)) yield payload;
  } finally {
    reader.releaseLock();
  }
}

/** An OpenAI SSE stream as a normalised answer: text pieces joined, usage from the usage chunk. */
export async function collectOpenAIStream(stream: ReadableStream<Uint8Array>): Promise<NormalizedAnswer> {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let finish = 'stop';
  for await (const payload of ssePayloads(stream)) {
    if (!isRecord(payload)) continue;
    if (isRecord(payload.usage)) {
      inputTokens = numberOrZero(payload.usage.prompt_tokens);
      outputTokens = numberOrZero(payload.usage.completion_tokens);
    }
    const choices = Array.isArray(payload.choices) ? payload.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') finish = choice.finish_reason;
  }
  return { text, inputTokens, outputTokens, finish };
}

/** An Anthropic SSE stream as a normalised answer. */
export async function collectAnthropicStream(stream: ReadableStream<Uint8Array>): Promise<NormalizedAnswer> {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  for await (const payload of ssePayloads(stream)) {
    if (!isRecord(payload)) continue;
    const type = payload.type;
    if (type === 'message_start' && isRecord(payload.message) && isRecord(payload.message.usage)) {
      inputTokens = numberOrZero(payload.message.usage.input_tokens);
    } else if (type === 'content_block_delta' && isRecord(payload.delta) && payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string') {
      text += payload.delta.text;
    } else if (type === 'message_delta') {
      if (isRecord(payload.delta) && typeof payload.delta.stop_reason === 'string') stop = payload.delta.stop_reason;
      if (isRecord(payload.usage)) outputTokens = numberOrZero(payload.usage.output_tokens);
    }
  }
  return { text, inputTokens, outputTokens, finish: finishReasonOf(stop) };
}

/** A Gemini SSE stream as a normalised answer: text pieces joined, usage from the final frame. */
export async function collectGeminiStream(stream: ReadableStream<Uint8Array>): Promise<NormalizedAnswer> {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  for await (const payload of ssePayloads(stream)) {
    if (!isRecord(payload)) continue;
    const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
    const cand = isRecord(candidates[0]) ? candidates[0] : null;
    if (cand !== null && isRecord(cand.content) && Array.isArray(cand.content.parts)) {
      for (const part of cand.content.parts) {
        if (isRecord(part) && typeof part.text === 'string') text += part.text;
      }
    }
    if (isRecord(payload.usageMetadata)) {
      inputTokens = numberOrZero(payload.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(payload.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, inputTokens, outputTokens, finish: 'stop' };
}

/** A normalised answer as an OpenAI Chat Completions JSON object. */
export function renderOpenAIJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: answer.finish, logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A normalised answer as an Anthropic Messages JSON object. */
export function renderAnthropicJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopReasonOf(answer.finish),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A normalised answer as an OpenAI SSE stream: role chunk, 13-char pieces, finish, usage, [DONE]. */
export function renderOpenAIStream(answer: NormalizedAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (let at = 0; at < answer.text.length; at += 13) {
    frames.push(sseEvent(null, chunk({ content: answer.text.slice(at, at + 13) }, null)));
  }
  frames.push(sseEvent(null, chunk({}, answer.finish)));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalised answer as an Anthropic SSE stream: start, text deltas, stop, usage, stop. */
export function renderAnthropicStream(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (let at = 0; at < answer.text.length; at += 13) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text.slice(at, at + 13) } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(answer.finish), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
