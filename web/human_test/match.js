'use strict';
// Human test, match mode: one human and five frozen AI players, silent or talking. Text comes from i18n.js.
const $ = id => document.getElementById(id);
const T = I18N.t;
const BASE = '/match';
const SIZES = [2, 3, 4, 3, 4];
const MIC = '<svg viewBox="0 0 24 24"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zM6 11a6 6 0 0 0 12 0M12 17v4"/></svg>';
const CROWN = '<svg viewBox="0 0 16 16"><path d="M2 12.5h12l1-7.5-4 3-3-5.5-3 5.5-4-3z"/></svg>';
let S = {active: false}, sel = new Set(), busy = false, timer = null;
let ahead = null;    // the next state, asked for while the visitor still reads the current one: {id, revision, p}
let rec = null;      // the recording in progress: {recorder, timer}
let hearing = false; // a typed sentence or a recording is with the language model
const setup = {role: 'random', seat: 1, baseline: 0, share: null, mode: 'talk'};
// The home page links to one kind of table: /match/?mode=silent or /match/?mode=talk.
const FIXED_MODE = (m => (m === 'silent' || m === 'talk' ? m : null))(new URLSearchParams(location.search).get('mode'));
if (FIXED_MODE) document.body.dataset.title = `mode.${FIXED_MODE}.name`;
let plan = null;  // the team a leader named while speaking, picked for the proposal that follows: {id, quest, team}
// What the visitor is about to say (a stance, see scripts/human_test_talk.py); the suggested team is `sel`.
let draft = newDraft();
function newDraft() { return {claim: null, judge: {}, veto: [], require: [], reply: {}, promise: {}, ignored: [], text: '', typed: false}; }
// Spoken sentences: the agents' messages put into natural speech by a language model (the meaning is checked on the server).
const spoken = () => !!S.llm && $('spoken').checked;

function pref(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : v; } catch (e) { return fallback; } }
function setPref(key, value) { try { localStorage.setItem(key, value); } catch (e) {} }
function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; }
function showErr(msg) { $('err').textContent = msg ? I18N.err(msg) : ''; $('err').hidden = !msg; }

async function api(path, data) {
  const r = await fetch(BASE + path, data === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || 'Error');
  return j;
}
async function move(action) {
  if (busy) return;
  busy = true; showErr('');
  try {
    S = await api('/api/action', {id: S.id, revision: S.revision, action});
    sel.clear(); draft = newDraft(); render();
  } catch (e) { showErr(e.message); } finally { busy = false; }
}
// The agents' next step is asked for as soon as the current one is shown, so that the visitor's reading time
// covers the server's work (a spoken sentence takes a second or two).
function fetchAhead() {
  if (ahead || !S.active || S.done || S.human_turn) return;
  const p = api('/api/advance', {id: S.id, revision: S.revision, polish: spoken(), lang: I18N.lang});
  p.catch(() => {});  // reported when the step is taken
  ahead = {id: S.id, revision: S.revision, p};
}
async function advance() {
  if (busy) return;
  busy = true; showErr('');
  fetchAhead();
  const mine = ahead;
  try {
    const next = mine ? await mine.p : S;
    if (ahead === mine) ahead = null;
    if (mine && S.id === mine.id && S.revision === mine.revision) { S = next; sel.clear(); draft = newDraft(); }
  } catch (e) { if (ahead === mine) ahead = null; showErr(e.message); }
  finally { busy = false; render(); }
}

async function speak(stance) {
  if (busy) return;
  busy = true; showErr('');
  try {
    S = await api('/api/speak', {id: S.id, revision: S.revision, stance, typed: draft.typed, ...(draft.typed && draft.text.trim() ? {text: draft.text.trim()} : {})});
    sel.clear(); draft = newDraft(); render();
  } catch (e) { showErr(e.message); } finally { busy = false; }
}
function heard(r) {  // what the agents would hear in a typed or spoken sentence becomes the draft
  Object.assign(draft, {claim: r.stance.claim, judge: r.stance.judge, veto: r.stance.veto, require: r.stance.require,
                        reply: r.stance.reply || {}, promise: r.stance.promise || {}, ignored: r.ignored, typed: true});
  if (typeof r.text === 'string') draft.text = r.text;
  sel = new Set(r.stance.team || []);
  plan = r.propose ? {id: S.id, quest: S.quest, team: r.propose} : null;
  if (plan) draft.ignored = draft.ignored.map(x => (x.why === 'leader' ? {...x, why: 'leader_kept'} : x));
}
async function parseText() {
  if (busy || !draft.text.trim()) return;
  busy = hearing = true; showErr(''); render();
  try { heard(await api('/api/parse', {text: draft.text, lang: I18N.lang})); }
  catch (e) { showErr(e.message); } finally { busy = hearing = false; render(); }
}
// Voice: tap to record, tap again to stop (30 s at most); the recording is turned into text and read like typed text.
const canRecord = () => !!S.llm && !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
async function toggleMic() {
  if (rec) { rec.recorder.stop(); return; }
  if (busy) return;
  showErr('');
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({audio: true}); }
  catch (e) { showErr(T('mic.denied')); render(); return; }
  const type = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'].find(t => MediaRecorder.isTypeSupported(t));
  const recorder = new MediaRecorder(stream, type ? {mimeType: type, audioBitsPerSecond: 32000} : {});
  const chunks = [], table = S.id;
  recorder.addEventListener('dataavailable', e => { if (e.data.size) chunks.push(e.data); });
  recorder.addEventListener('stop', () => {
    clearTimeout(rec.timer); rec = null;
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(chunks, {type: (recorder.mimeType || type || 'audio/webm').split(';')[0]});
    if (S.id === table && blob.size) hearVoice(blob); else render();
  });
  rec = {recorder, timer: setTimeout(() => recorder.stop(), 30000)};
  recorder.start();
  render();
}
async function hearVoice(blob) {
  busy = hearing = true; showErr(''); render();
  try {
    const r = await fetch(`${BASE}/api/hear?lang=${I18N.lang}`, {method: 'POST', headers: {'Content-Type': blob.type}, body: blob});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Error');
    heard(j);
  } catch (e) { showErr(e.message); } finally { busy = hearing = false; render(); }
}

const isEvilRole = k => k === 'MORGANA' || k === 'ASSASSIN';
const talking = () => S.active && !S.done && (S.phase === 'DISCUSS' || S.phase === 'EVIL_DISCUSS');
const suggesting = () => S.active && !S.done && S.phase === 'DISCUSS' && S.human_turn && S.human !== S.leader;

// A message (the agents' or the visitor's draft) as one sentence in the page language. `teams`: seat -> the team
// it suggested in the same round, so that a reply can say what it means for that team.
function sentence(m, teams = {}) {
  const zh = I18N.lang === 'zh', parts = [];
  if (m.claim) parts.push(T('s.claim.' + m.claim));
  for (const j of m.judge || []) {
    const what = j.judgment === 'good' || j.judgment === 'evil' ? T('side.' + j.judgment) : I18N.role(j.judgment.toUpperCase());
    const basis = j.basis === 'role_knowledge' ? T('s.sight') : j.basis && j.quest ? T('s.basis', {q: j.quest, k: T('basis.' + j.basis)}) : '';
    parts.push(T('s.judge', {n: j.seat, j: zh || !['good', 'evil'].includes(j.judgment) ? what : what.toLowerCase()}) + basis);
  }
  if (m.team && m.team.length) parts.push(T('s.team', {team: I18N.seats(m.team)}));
  for (const n of m.veto || []) parts.push(T('s.veto', {n}));
  for (const n of m.require || []) parts.push(T('s.require', {n}));
  for (const r of m.reply || []) {
    const b = r.basis ? T('rb.' + r.basis) : '';
    parts.push(r.reply === 'endorse' ? T('s.endorse', {n: r.seat}) : T(teams[r.seat] ? 's.disputeTeam' : 's.dispute', {n: r.seat, b}));
  }
  for (const kind of ['fail_evil', 'success_good']) {  // promises of one kind are said together
    const about = (m.promise || []).filter(k => k.promise === kind).map(k => k.seat);
    if (about.length) parts.push(T(`s.promise.${kind}${about.length > 1 ? '.n' : ''}`, {n: I18N.seats(about)}));
  }
  return parts.length ? parts.join(zh ? '；' : '; ') : T('s.none');
}
// What the page shows for a message: the visitor's own words, the spoken sentence in the page language, or the template.
function said(m, teams = {}) {
  const x = m.said || {};
  return x.raw || (spoken() && x[I18N.lang]) || sentence(m, teams);
}
function draftStance() {
  const team = suggesting() && sel.size === S.team_size ? [...sel].sort((a, b) => a - b) : null;
  return {claim: draft.claim, judge: draft.judge, team, veto: draft.veto, require: draft.require, reply: draft.reply, promise: draft.promise};
}
const draftMessage = () => ({...draftStance(), judge: Object.entries(draft.judge).map(([seat, judgment]) => ({seat: +seat, judgment})),
  reply: Object.entries(draft.reply).map(([seat, reply]) => ({seat: +seat, reply})),
  promise: Object.entries(draft.promise).map(([seat, promise]) => ({seat: +seat, promise}))});
// seat -> suggested team, for the messages of one round, and for the round being spoken now
const teamsOf = messages => Object.fromEntries(messages.filter(m => m.team && m.team.length).map(m => [m.speaker, m.team]));
function teamsNow() {
  const last = S.talk && S.talk.length ? S.talk[S.talk.length - 1] : null;
  return last && !last.evil_talk && last.quest === S.quest ? teamsOf(last.messages) : {};
}

const pos = s => (S.active ? (s - S.human + 6) % 6 : s - 1);  // your seat sits at the bottom of the table
function canPick(s) {
  if (!S.active || !S.human_turn || S.done) return false;
  return S.phase === 'PROPOSE' || suggesting() || (S.phase === 'ASSASSINATE' && S.options.some(o => o.target === s));
}

function renderTrack() {
  const qs = S.active ? S.quests : SIZES.map(size => ({size, result: -1}));
  $('quests').innerHTML = qs.map((q, i) => {
    const cls = q.result === 0 ? 'ok' : q.result === 1 ? 'bad' : S.active && !S.done && S.quest === i + 1 ? 'now' : '';
    return `<span class="q ${cls}" title="${T('quest', {q: i + 1})}">${q.result === 0 ? '✓' : q.result === 1 ? '✗' : q.size}</span>`;
  }).join('');
  const rejected = S.active && ['PROPOSE', 'VOTE'].includes(S.phase) ? S.attempt - 1 : 0;
  $('tries').innerHTML = [0, 1, 2, 3, 4].map(i => `<b class="${i < rejected ? 'on' : ''}"></b>`).join('');
}

// The speech just made sits in a bubble next to its seat until the table moves on.
function renderBubbles(ring) {
  ring.querySelectorAll('.bubble').forEach(n => n.remove());
  if (!S.active || !S.talk || S.event.kind !== 'speech') return;
  const m = S.event.message, teams = speechTeams();
  const b = el('div', `bubble b${pos(m.speaker)}`);
  b.textContent = said(m, teams);
  if (!(m.said && m.said.raw) && sentence(m, teams) === T('s.none')) b.classList.add('mute');
  if (S.active && m.speaker === S.human) b.classList.add('mine');
  b.setAttribute('aria-label', `${T('said', {n: who(m.speaker)})}: ${b.textContent}`);
  ring.append(b);
}

function renderRing() {
  const ring = $('ring');
  renderBubbles(ring);
  ring.querySelectorAll('.seat').forEach(n => n.remove());
  const leading = S.active && !S.done && ['DISCUSS', 'PROPOSE', 'VOTE', 'QUEST'].includes(S.phase);
  for (let s = 1; s <= 6; s++) {
    const you = S.active && s === S.human;
    const knownEvil = S.active && S.known_evil.includes(s) && !you;
    const cand = S.active && S.candidates.includes(s);
    const roleKey = S.done ? S.result.role_keys[s - 1] : null;
    const b = el('button', `seat p${pos(s)}`);
    b.type = 'button';
    if (you) b.classList.add('you');
    if (S.active && S.team.includes(s)) b.classList.add('team');
    if (sel.has(s)) b.classList.add('sel');
    if (canPick(s)) b.classList.add('pick');
    if (talking() && s === S.speaker) b.classList.add('speaking');
    if (knownEvil || (roleKey && isEvilRole(roleKey))) b.classList.add('evil');
    let badges = '';
    if (leading && s === S.leader) badges += `<span class="badge tl lead">${CROWN}</span>`;
    if (S.active && S.event.kind === 'vote') badges += S.event.votes[s - 1] ? '<span class="badge tr yes">✓</span>' : '<span class="badge tr no">✗</span>';
    if (S.done && S.event.kind === 'assassination' && S.event.target === s) badges += '<span class="badge br dot">†</span>';
    const broken = S.active && S.broken_words ? S.broken_words[s - 1] : 0;
    if (broken) badges += `<span class="badge bl warn" title="${T('broken', {n: broken})}">!${broken}</span>`;
    let cap = you ? T('you') : 'AI', capCls = '';
    if (roleKey) { cap = I18N.role(roleKey) + (you ? ` · ${T('you')}` : ''); capCls = isEvilRole(roleKey) ? 'evil' : 'good'; }
    else if (knownEvil) { cap = T('side.evil'); capCls = 'evil'; }
    else if (cand) cap = T('me.cands');
    b.innerHTML = `<span class="disc">${s}${badges}</span><span class="cap ${capCls}">${cap}</span>`;
    b.setAttribute('aria-label', `${T('seat', {n: s})} · ${cap}`);
    b.addEventListener('click', () => {
      if (busy || !canPick(s)) return;
      if (S.phase === 'ASSASSINATE') { sel.clear(); sel.add(s); }
      else if (sel.has(s)) sel.delete(s);
      else if (sel.size < S.team_size) sel.add(s);
      render();
    });
    ring.append(b);
  }
}

function renderCenter() {
  let big = 'AVALON', sub = '';
  if (S.active && S.done) {
    big = T(S.result.good_won ? 'win.good' : 'win.evil');
    sub = `${T('reason.' + S.result.reason_key)} · ${T(S.result.human_won ? 'won' : 'lost')}`;
  } else if (S.active) {
    big = S.phase === 'ASSASSINATE' ? T('whoMerlin') : S.phase === 'EVIL_DISCUSS' ? T('evilTalk') : T('quest', {q: S.quest});
    if (talking()) sub = S.human_turn ? T('yourTalk') : T('talkingNow', {n: S.speaker});
    if (S.phase === 'PROPOSE') sub = S.human_turn ? T('picked', {k: sel.size, n: S.team_size}) : T('leaderPicks', {n: S.leader});
    else if (S.phase === 'VOTE') sub = S.attempt === 5 ? T('lastTry') : T('voteTeam');
    else if (S.phase === 'QUEST') sub = S.human_turn ? T('playCard') : T('questOn');
    else if (S.phase === 'ASSASSINATE') sub = S.human_turn ? '' : T('assnWait');
  }
  $('c-big').textContent = big;
  $('c-sub').textContent = sub;
  const ev = S.active ? S.event : null, line = $('event');
  line.className = 'event';
  if (!ev) { line.textContent = ''; return; }
  if (ev.kind === 'start') line.textContent = T('ev.start');
  else if (ev.kind === 'speech') line.textContent = sub ? `${big} · ${sub}` : big;  // the bubble covers the centre of the table
  else if (ev.kind === 'proposal') line.textContent = T('ev.proposal', {l: ev.leader, team: I18N.seats(ev.team || [])});
  else if (ev.kind === 'vote') {
    const y = ev.votes.filter(Boolean).length;
    line.textContent = T('ev.vote', {y, n: 6 - y, r: T(ev.passed ? 'passed' : 'rejected')});
    line.classList.add(ev.passed ? 'ok' : 'bad');
  } else if (ev.kind === 'quest') {
    line.textContent = T(ev.fails ? 'ev.bad' : 'ev.ok', {q: ev.quest + 1, f: ev.fails});
    line.classList.add(ev.fails ? 'bad' : 'ok');
  } else if (ev.kind === 'assassination') line.textContent = T('ev.assn', {n: ev.target});
}

const who = s => (S.active && s === S.human ? T('you') : T('seat', {n: s}));
// The teams of the round the last speech belongs to: always the last round of the log.
function speechTeams() {
  const round = S.talk && S.talk.length ? S.talk[S.talk.length - 1] : null;
  return round && !round.evil_talk ? teamsOf(round.messages) : {};
}
function btn(label, cls, onClick, disabled = false) {
  const b = el('button', `btn ${cls}`); b.type = 'button'; b.textContent = label; b.disabled = disabled;
  b.addEventListener('click', onClick); return b;
}
function renderDock() {
  const dock = $('dock');
  dock.replaceChildren();
  clearTimeout(timer); timer = null;
  document.body.classList.remove('composing');
  if (!S.active) { dock.append(btn(T('newGame'), 'primary wide', openSetup)); return; }
  if (S.done) { dock.append(btn(T('again'), 'primary wide', openSetup), btn(T('replay'), 'ghost', downloadReplay)); return; }
  if (!S.human_turn) {
    dock.append(btn(T('continue') + ' →', 'primary wide', advance));
    fetchAhead();
    // Speech needs time to be read: the pause grows with the sentence just shown (an English sentence has more characters).
    const pause = S.event.kind === 'speech' ? Math.min(9000, 1600 + (I18N.lang === 'zh' ? 70 : 28) * said(S.event.message, speechTeams()).length) : 1500;
    if ($('auto').checked) timer = setTimeout(() => { timer = null; if (!busy && !$('setup').open) advance(); }, pause);
    return;
  }
  document.body.classList.toggle('composing', talking());
  if (talking()) { composer(dock); return; }
  const kind = k => S.options.find(o => o.kind === k);
  if (S.phase === 'PROPOSE') {
    const team = [...sel].sort((a, b) => a - b);
    const opt = S.options.find(o => o.team && o.team.join() === team.join());
    dock.append(btn(T('confirmTeam') + (team.length ? ` · ${I18N.seats(team)}` : ''), 'primary wide', () => move(opt.action), !opt));
  } else if (S.phase === 'VOTE') {
    dock.append(btn('✓ ' + T('approve'), 'good wide', () => move(kind('approve').action)),
                btn('✗ ' + T('reject'), 'evil wide', () => move(kind('reject').action)));
  } else if (S.phase === 'QUEST') {
    dock.append(btn('✓ ' + T('success'), 'good wide', () => move(kind('success').action)));
    if (kind('fail')) dock.append(btn('✗ ' + T('fail'), 'evil wide', () => move(kind('fail').action)));
  } else if (S.phase === 'ASSASSINATE') {
    const target = [...sel][0], opt = S.options.find(o => o.target === target);
    dock.append(btn(target ? T('kill', {n: target}) : T('pickTarget'), 'evil wide', () => move(opt.action), !opt));
  }
}

// The visitor's turn to speak: type a sentence or set the stance by hand; the preview is what the agents hear.
function cycle(label, value, states, onChange) {
  const i = states.findIndex(st => st[0] === value), c = el('button', `chip sm ${states[i][2] || ''}`);
  c.type = 'button'; c.textContent = label + (states[i][1] ? ' · ' + states[i][1] : '');
  c.setAttribute('aria-pressed', String(i > 0));
  c.addEventListener('click', () => { onChange(states[(i + 1) % states.length][0]); draft.typed = false; render(); });
  return c;
}
function composer(dock) {
  const evilTalk = S.phase === 'EVIL_DISCUSS';
  let box = el('div', 'composer');
  const form = el('form', 'say'), input = el('input', 'input');
  input.placeholder = T(evilTalk ? 'say.phEvil' : 'say.ph'); input.maxLength = 300; input.value = draft.text; input.autocomplete = 'off';
  input.addEventListener('input', () => { draft.text = input.value; });
  const go = el('button', 'btn slim'); go.type = 'submit'; go.textContent = T('say.parse');
  form.append(input);
  if (canRecord()) {
    const mic = el('button', 'btn slim mic' + (rec ? ' on' : ''), rec ? '■' : MIC); mic.type = 'button';
    mic.setAttribute('aria-label', T(rec ? 'mic.stop' : 'mic.start')); mic.title = T(rec ? 'mic.stop' : 'mic.start');
    mic.addEventListener('click', toggleMic);
    form.append(mic);
  }
  form.append(go);
  form.addEventListener('submit', e => { e.preventDefault(); parseText(); });
  box.append(form);
  // On a phone the composer covers the error line of the page, so an error is repeated here.
  if (rec || hearing) box.append(el('p', 'note status', T(rec ? 'mic.on' : 'say.hearing')));
  else if (!$('err').hidden) { const p = el('p', 'note bad'); p.textContent = $('err').textContent; box.append(p); }
  else if (S.llm && !draft.typed) box.append(el('p', 'note', T('say.note')));
  // The buttons below set the same stance by hand; with a language model they start folded away.
  const manual = el('details', 'manual'), top = box;
  manual.open = pref('avalon_manual', S.llm ? 'closed' : 'open') === 'open';
  manual.addEventListener('toggle', () => setPref('avalon_manual', manual.open ? 'open' : 'closed'));
  manual.append(el('summary', '', T('say.manual')));
  box = el('div', 'manual-body');
  manual.append(box);
  if (!evilTalk) {
    const claims = el('div', 'chips');
    for (const [value, label] of [[null, T('say.noClaim')], ['servant', I18N.role('SERVANT')], ['percival', I18N.role('PERCIVAL')]]) {
      const c = el('button', 'chip sm'); c.type = 'button'; c.textContent = value ? T('s.claim.' + value) : label;
      c.setAttribute('aria-pressed', String(draft.claim === value));
      c.addEventListener('click', () => { draft.claim = value; draft.typed = false; render(); });
      claims.append(c);
    }
    box.append(claims);
  }
  const judgeStates = evilTalk
    ? [[undefined, ''], ['merlin', I18N.role('MERLIN'), 'evil'], ['percival', I18N.role('PERCIVAL')], ['servant', I18N.role('SERVANT')]]
    : [[undefined, ''], ['good', T('side.good'), 'good'], ['evil', T('side.evil'), 'evil']];
  const intentStates = [['', T('say.anyTeam')], ['veto', T('say.veto'), 'evil'], ['require', T('say.require'), 'good']];
  // Replies go to the players who already spoke in this round; promises are about this round's quest.
  const say = (!evilTalk && S.say) || {reply: [], promise: [], promised: []};
  const spoke = Object.fromEntries(say.reply.map(o => [o.seat, o.team]));
  const promiseStates = [[undefined, T('say.none')], ...say.promise.map(k => [k, T('say.' + k), k === 'fail_evil' ? 'evil' : 'good'])];
  if (say.promised.length) {
    const zh = I18N.lang === 'zh';
    top.append(el('p', 'note due', T('say.promised') + say.promised.map(x => T('s.judge', {n: x.seat, j: zh ? T('side.' + x.judgment) : T('side.' + x.judgment).toLowerCase()})).join(zh ? '；' : '; ')));
  }
  const rows = el('div', 'stances' + (say.reply.length || say.promise.length ? ' wide' : ''));
  for (let s = 1; s <= 6; s++) {
    if (s === S.human || (evilTalk && S.known_evil.includes(s))) continue;
    const row = el('div', 'stance');
    row.append(cycle(T('seat', {n: s}), draft.judge[s], judgeStates, v => { if (v) draft.judge[s] = v; else delete draft.judge[s]; }));
    if (!evilTalk) {
      const intent = draft.veto.includes(s) ? 'veto' : draft.require.includes(s) ? 'require' : '';
      row.append(cycle(T('say.vote'), intent, intentStates, v => {
        draft.veto = draft.veto.filter(x => x !== s); draft.require = draft.require.filter(x => x !== s);
        if (v) draft[v].push(s);
      }));
      if (s in spoke) {
        const replyStates = [[undefined, T('say.none')], ...(spoke[s] ? [['endorse', T('say.endorse'), 'good']] : []), ['dispute', T('say.dispute'), 'evil']];
        row.append(cycle(T('say.reply'), draft.reply[s], replyStates, v => { if (v) draft.reply[s] = v; else delete draft.reply[s]; }));
      }
      if (say.promise.length) row.append(cycle(T('say.promise'), draft.promise[s], promiseStates, v => { if (v) draft.promise[s] = v; else delete draft.promise[s]; }));
    }
    rows.append(row);
  }
  box.append(rows);
  box = top;
  box.append(manual);
  if (suggesting()) box.append(el('p', 'note', T('say.team', {k: sel.size, n: S.team_size})));
  else if (!evilTalk && S.human === S.leader) box.append(el('p', 'note', T('say.leader')));
  const heard = el('p', 'heard'); heard.textContent = T('say.heard') + sentence(draftMessage(), teamsNow());
  box.append(heard);
  for (const x of draft.ignored) if (x.why !== 'other') { const p = el('p', 'ignored'); p.textContent = `“${x.text}” — ${T('why.' + x.why)}`; box.append(p); }
  const partial = suggesting() && sel.size > 0 && sel.size !== S.team_size;
  const actions = el('div', 'actions');
  actions.append(btn(T('say.silent'), 'ghost', () => speak({})), btn(T('say.send'), 'primary wide', () => speak(draftStance()), partial));
  box.append(actions);
  dock.append(box);
}

function renderTalk() {
  const card = $('talk-card');
  card.hidden = !(S.active && S.talk);
  if (card.hidden) return;
  if (!S.talk.length) { $('talk').innerHTML = `<p class="empty">${T('empty')}</p>`; return; }
  $('talk').replaceChildren(...[...S.talk].reverse().map(round => {
    const box = el('div', 'round');
    box.append(el('p', 'round-title', round.evil_talk ? T('evilTalk') : T('quest', {q: round.quest})));
    const teams = teamsOf(round.messages);
    for (const m of round.messages) {
      const line = el('p', 'line'), name = el('b', m.speaker === S.human ? 'you' : '');
      name.textContent = who(m.speaker);
      line.append(name, document.createTextNode(said(m, teams)));
      if (m.said && m.said.raw) {  // under the visitor's own words: what reached the agents
        const sub = el('span', 'sub'); sub.textContent = T('say.heard') + sentence(m, teams);
        line.append(sub);
      }
      box.append(line);
    }
    return box;
  }));
}

function renderMe() {
  const me = $('me');
  if (!S.active) { me.className = 'card me'; me.innerHTML = `<span class="crest">?</span><div><div class="who">—</div></div>`; return; }
  me.className = `card me ${S.side === 'evil' ? 'evil' : ''}`;
  const role = I18N.role(S.role_key), lines = [];
  const evil = S.known_evil.filter(s => s !== S.human);
  if (evil.length) lines.push(`${T('me.evil')} <b>${I18N.seats(evil)}</b>`);
  if (S.candidates.length) lines.push(`${T('me.cands')} <b>${I18N.seats(S.candidates)}</b>`);
  if (!lines.length) lines.push(T('me.none'));
  const cards = S.own_cards.map((c, i) => (c >= 0 ? `${i + 1}${c ? '✗' : '✓'}` : null)).filter(Boolean);
  if (cards.length) lines.push(`${T('me.cards')} <b>${cards.join(' ')}</b>`);
  me.innerHTML = `<span class="crest">${role.charAt(0)}</span><div><div class="who">${role} · ${T('seat', {n: S.human})}</div><div class="info">${lines.join(' · ')}</div></div>`;
}

function renderLog() {
  const h = S.active ? S.history : [];
  if (!h.length) { $('log').innerHTML = `<p class="empty">${T('empty')}</p>`; return; }
  const rows = [...h].reverse().map(p => {
    const q = S.quests[p.quest - 1];
    let res = '…';
    if (p.votes && !p.approved) res = `<span class="muted">${T('rejected')}</span>`;
    else if (p.approved && q.result === 0) res = '<span class="res-ok">✓</span>';
    else if (p.approved && q.result === 1) res = `<span class="res-bad">✗ ${q.fails}</span>`;
    const votes = p.votes ? `<span class="votes">${p.votes.map((v, i) => `<span class="${v ? 'y' : ''}">${i + 1}</span>`).join('')}</span>` : '';
    return `<tr><td>${p.quest}-${p.attempt}</td><td>${p.leader}</td><td><span class="team">${p.team.map(s => `<span>${s}</span>`).join('')}</span></td><td>${votes}</td><td>${res}</td></tr>`;
  }).join('');
  $('log').innerHTML = `<table class="log"><tr><th>${T('col.q')}</th><th>${T('col.lead')}</th><th>${T('col.team')}</th><th>${T('col.votes')}</th><th>${T('col.res')}</th></tr>${rows}</table>`;
}

function render() {
  // A leader who named a team while speaking finds it picked when the proposal comes.
  if (plan && S.active && S.id === plan.id && S.quest === plan.quest && S.phase === 'PROPOSE' && S.human_turn && S.attempt === 1) {
    if (!sel.size) sel = new Set(plan.team);
    plan = null;
  } else if (plan && (!S.active || S.id !== plan.id || S.quest !== plan.quest || !['DISCUSS', 'PROPOSE'].includes(S.phase))) plan = null;
  $('spoken-row').hidden = !(S.llm && S.active && S.mode === 'talk');
  renderTrack(); renderRing(); renderCenter(); renderDock(); renderMe(); renderTalk(); renderLog();
}

// ---- new game dialog
function chips(box, items, key) {
  box.replaceChildren();
  for (const [value, label, cls] of items) {
    const c = el('button', `chip ${cls || ''}`); c.type = 'button'; c.textContent = label;
    c.setAttribute('aria-pressed', String(setup[key] === value));
    c.addEventListener('click', () => { setup[key] = value; renderSetup(); });
    box.append(c);
  }
}
function renderSetup() {
  $('mode-row').hidden = !!FIXED_MODE;
  $('setup-title').textContent = T('setup.title') + (FIXED_MODE ? ` · ${T(`mode.${FIXED_MODE}.name`)}` : '');
  chips($('role-chips'), [['random', T('random')], ...['merlin', 'percival', 'servant', 'morgana', 'assassin'].map(r => [r, I18N.role(r.toUpperCase()), isEvilRole(r.toUpperCase()) ? 'evil' : 'good'])], 'role');
  chips($('mode-chips'), [['talk', T('mode.talk')], ['silent', T('mode.silent')]], 'mode');
  chips($('seat-chips'), [1, 2, 3, 4, 5, 6].map(n => [n, String(n)]), 'seat');
  chips($('ai-chips'), [0, 1, 2].map(n => [n, `seed ${n}`]), 'baseline');
  chips($('share-chips'), [[true, T('share.yes'), 'good'], [false, T('share.no')]], 'share');
}
function openSetup() {
  if (setup.share === null) setup.share = S.share != null ? S.share : ({yes: true, no: false})[pref('avalon_share', '')] ?? null;
  $('cancel').hidden = !S.active;
  $('setup-err').hidden = true;
  renderSetup();
  $('setup').showModal();
}
$('setup-form').addEventListener('submit', async e => {
  e.preventDefault();
  if (busy) return;
  if (setup.share === null) { $('setup-err').textContent = T('share.q'); $('setup-err').hidden = false; return; }
  busy = true; $('start').disabled = true; $('start').textContent = T('loading');
  try {
    setPref('avalon_share', setup.share ? 'yes' : 'no');
    setPref('avalon_mode', setup.mode);
    S = await api('/api/new', {role: setup.role, seat: setup.seat, baseline: setup.baseline, share: setup.share, mode: setup.mode});
    ahead = null; sel.clear(); draft = newDraft(); showErr(''); $('setup').close(); render();
  } catch (err) { $('setup-err').textContent = I18N.err(err.message); $('setup-err').hidden = false; }
  finally { busy = false; $('start').disabled = false; $('start').textContent = T('start'); }
});
$('cancel').addEventListener('click', () => $('setup').close());
$('setup').addEventListener('cancel', e => { if (!S.active) e.preventDefault(); });
$('new-btn').addEventListener('click', openSetup);
$('rules-btn').addEventListener('click', () => $('rules').showModal());
$('rules-ok').addEventListener('click', () => $('rules').close());
setup.mode = FIXED_MODE || (pref('avalon_mode', 'talk') === 'silent' ? 'silent' : 'talk');
$('auto').checked = pref('avalon_autoplay', 'on') === 'on';
$('auto').addEventListener('change', () => { setPref('avalon_autoplay', $('auto').checked ? 'on' : 'off'); renderDock(); });
$('spoken').checked = pref('avalon_spoken', 'on') === 'on';
$('spoken').addEventListener('change', () => { setPref('avalon_spoken', $('spoken').checked ? 'on' : 'off'); render(); });

async function downloadReplay() {
  try {
    const r = await api('/api/replay');
    const url = URL.createObjectURL(new Blob([JSON.stringify(r, null, 2)], {type: 'application/json'}));
    const a = el('a'); a.href = url; a.download = `avalon-${S.mode}-seed${S.baseline}-${S.id}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) { showErr(e.message); }
}

I18N.onChange(() => { render(); if ($('setup').open) renderSetup(); });
render();
api('/api/state').then(s => {
  S = s; render();
  if (!S.active) setup.baseline = S.default_baseline || 0;
  if (!S.active || (FIXED_MODE && S.mode !== FIXED_MODE)) openSetup();  // a running game of the other kind: offer a new one
}).catch(e => showErr(e.message));
