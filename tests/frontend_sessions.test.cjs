// Run with: node --test tests/frontend_sessions.test.cjs
// Exercise the actual page script with a controlled HTTP/WebSocket transport.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { webcrypto } = require('node:crypto');

const html = fs.readFileSync(path.join(__dirname, '../api/static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page() {
  const elements = new Map();
  const requests = [];
  const sockets = [];
  const alerts = [];
  let respond = body => ({ ok: true, json: async () => ({ status: 'started', thread_id: body.thread_id }) });
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, {
      value: '', innerHTML: '', textContent: '', disabled: false, scrollTop: 0,
      focus() {}, addEventListener() {},
    });
    return elements.get(selector);
  };
  class Socket {
    static OPEN = 1;
    constructor(url) { this.url = url; this.readyState = 0; sockets.push(this); }
    open() { this.readyState = 1; this.onopen(); }
    close() { this.readyState = 3; this.onclose?.(); }
    event(event, data = {}) {
      this.onmessage({ data: JSON.stringify({ type: 'monitor_event', event, data, message: event, timestamp: new Date().toISOString() }) });
    }
  }
  const context = vm.createContext({
    document: { querySelector: element, querySelectorAll: () => [] },
    crypto: webcrypto, Uint8Array, WebSocket: Socket,
    location: { protocol: 'http:', host: 'localhost:8000' },
    setTimeout: () => 1, clearTimeout() {}, alert: message => alerts.push(message),
    fetch: async (url, options) => {
      if (url === '/api/files') return { json: async () => ({ files: [] }) };
      const body = JSON.parse(options.body);
      requests.push(body);
      return respond(body);
    },
  });
  vm.runInContext(script, context);
  return {
    run: code => vm.runInContext(code, context),
    element, requests, sockets, alerts,
    reply: fn => { respond = fn; },
    send(query) { element('#queryInput').value = query; return vm.runInContext('sendTask()', context); },
  };
}

test('connect before submitting; follow-ups reuse the thread and preserve both turns', async () => {
  const p = page();
  const first = p.send('Remember that my name is Alice');
  assert.equal(p.requests.length, 0);
  p.sockets[0].open();
  await first;
  const id = p.requests[0].thread_id;
  assert.match(id, /^[0-9a-f-]{36}$/);
  assert.ok(p.sockets[0].url.endsWith('/ws/' + id));
  assert.equal(p.element('#sendBtn').disabled, true);
  await p.send('duplicate while running');
  assert.equal(p.requests.length, 1);
  p.sockets[0].event('task_result', { result: 'Hello Alice' });
  assert.equal(p.element('#sendBtn').disabled, false);
  await p.send('What is my name?');
  assert.equal(p.requests[1].thread_id, id);
  assert.equal(p.sockets.length, 1);
  p.sockets[0].event('task_result', { result: 'Alice' });
  assert.equal(p.run('Object.keys(sessions).length'), 1);
  assert.equal(p.run('sessions[activeSessionId].logs.length'), 4);
  assert.match(p.element('#logArea').innerHTML, /Hello Alice/);
  assert.match(p.element('#logArea').innerHTML, /What is my name/);
});

test('new sessions are isolated and switching back resumes the original thread', async () => {
  const p = page();
  const first = p.send('First session');
  p.sockets[0].open();
  await first;
  const originalId = p.requests[0].thread_id;
  p.run('newSession()');
  assert.equal(p.element('#sendBtn').disabled, false);
  const second = p.send('Second session');
  p.sockets[1].open();
  await second;
  assert.notEqual(p.requests[1].thread_id, originalId);
  p.run(`switchSession('${originalId}')`);
  assert.equal(p.element('#sendBtn').disabled, true);
  p.sockets[1].event('task_result', { result: 'Second answer' });
  assert.equal(p.element('#sendBtn').disabled, true);
  p.sockets[0].event('task_result', { result: 'First answer' });
  await p.send('Follow up in first session');
  assert.equal(p.requests[2].thread_id, originalId);
  assert.doesNotMatch(p.element('#logArea').innerHTML, /Second answer/);
});

test('a result arriving before the HTTP response is not overwritten; full answers remain visible', async () => {
  const p = page();
  p.reply(body => {
    p.sockets[0].event('task_result', { result: 'x'.repeat(900) + '<last paragraph>' });
    return { ok: true, json: async () => ({ status: 'started', thread_id: body.thread_id }) };
  });
  const sending = p.send('Fast task');
  p.sockets[0].open();
  await sending;
  assert.equal(p.element('#sendBtn').disabled, false);
  assert.match(p.element('#logArea').innerHTML, /&lt;last paragraph&gt;/);
});

test('HTTP rejection preserves input and allows retry in the same session', async () => {
  const p = page();
  p.reply(() => ({ ok: false, status: 503 }));
  const sending = p.send('Retry me');
  p.sockets[0].open();
  await sending;
  assert.equal(p.element('#queryInput').value, 'Retry me');
  assert.equal(p.element('#sendBtn').disabled, false);
  assert.match(p.alerts[0], /503/);
  p.reply(body => ({ ok: true, json: async () => ({ status: 'started', thread_id: body.thread_id }) }));
  await p.send('Retry me');
  assert.equal(p.requests[0].thread_id, p.requests[1].thread_id);
});

test('closing a session while connecting does not submit the task', async () => {
  const p = page();
  const sending = p.send('Do not submit');
  p.run('closeAllSessions()');
  await sending;
  assert.equal(p.requests.length, 0);
  assert.equal(p.element('#sendBtn').disabled, false);
  assert.equal(p.run('Object.keys(sessions).length'), 0);
});

test('transport loss does not masquerade as completion or allow overlapping turns', async () => {
  const p = page();
  const sending = p.send('Long task');
  p.sockets[0].open();
  await sending;
  p.sockets[0].close();
  assert.equal(p.run('sessions[activeSessionId].status'), 'running');
  assert.equal(p.element('#sendBtn').disabled, true);
  await p.send('Do not overlap');
  assert.equal(p.requests.length, 1);
  p.run('newSession()');
  assert.equal(p.element('#sendBtn').disabled, false);
});

test('the selected retrieval mode is sent and policy denials are not terminal errors', async () => {
  const p = page();
  p.element('#retrievalMode').value = 'RAG_ONLY';
  const sending = p.send('Only use the knowledge base');
  p.sockets[0].open();
  await sending;
  assert.equal(p.requests[0].retrieval_mode, 'RAG_ONLY');
  p.sockets[0].event('policy_blocked', { error_code: 'WEB_FORBIDDEN' });
  assert.equal(p.run('sessions[activeSessionId].status'), 'running');
  assert.equal(p.element('#sendBtn').disabled, true);
  assert.match(p.element('#logArea').innerHTML, /拦截/);
  p.sockets[0].event('task_result', { result: 'No relevant knowledge' });
  assert.equal(p.element('#sendBtn').disabled, false);
});
