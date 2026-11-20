'use strict';
// Human vs. frozen baseline table. All text comes from i18n.js; the server sends language-neutral keys.
const $ = id => document.getElementById(id);
const T = I18N.t;
const BASE = location.pathname.startsWith('/play') ? '/play' : '';  // served alone or under the study site
const SIZES = [2, 3, 4, 3, 4];
const CROWN = '<svg viewBox="0 0 16 16"><path d="M2 12.5h12l1-7.5-4 3-3-5.5-3 5.5-4-3z"/></svg>';
let S = {active: false}, sel = new Set(), busy = false, timer = null;
const setup = {role: 'random', seat: 1, baseline: 0, share: null};

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
    S = await api(action === null ? '/api/advance' : '/api/action', {id: S.id, revision: S.revision, ...(action === null ? {} : {action})});
    sel.clear(); render();
  } catch (e) { showErr(e.message); } finally { busy = false; }
}

const isEvilRole = k => k === 'MORGANA' || k === 'ASSASSIN';
const pos = s => (S.active ? (s - S.human + 6) % 6 : s - 1);  // your seat sits at the bottom of the table
function canPick(s) {
  if (!S.active || !S.human_turn || S.done) return false;
  return S.phase === 'PROPOSE' || (S.phase === 'ASSASSINATE' && S.options.some(o => o.target === s));
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

function renderRing() {
  const ring = $('ring');
  ring.querySelectorAll('.seat').forEach(n => n.remove());
  const leading = S.active && !S.done && ['PROPOSE', 'VOTE', 'QUEST'].includes(S.phase);
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
    if (knownEvil || (roleKey && isEvilRole(roleKey))) b.classList.add('evil');
    let badges = '';
    if (leading && s === S.leader) badges += `<span class="badge tl lead">${CROWN}</span>`;
    if (S.active && S.event.kind === 'vote') badges += S.event.votes[s - 1] ? '<span class="badge tr yes">✓</span>' : '<span class="badge tr no">✗</span>';
    if (S.done && S.event.kind === 'assassination' && S.event.target === s) badges += '<span class="badge br dot">†</span>';
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
    big = S.phase === 'ASSASSINATE' ? T('whoMerlin') : T('quest', {q: S.quest});
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

function btn(label, cls, onClick, disabled = false) {
  const b = el('button', `btn ${cls}`); b.type = 'button'; b.textContent = label; b.disabled = disabled;
  b.addEventListener('click', onClick); return b;
}
function renderDock() {
  const dock = $('dock');
  dock.replaceChildren();
  clearTimeout(timer); timer = null;
  if (!S.active) { dock.append(btn(T('newGame'), 'primary wide', openSetup)); return; }
  if (S.done) { dock.append(btn(T('again'), 'primary wide', openSetup), btn(T('replay'), 'ghost', downloadReplay)); return; }
  if (!S.human_turn) {
    dock.append(btn(T('continue') + ' →', 'primary wide', () => move(null)));
    if ($('auto').checked) timer = setTimeout(() => { timer = null; if (!busy && !$('setup').open) move(null); }, 1500);
    return;
  }
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

function render() { renderTrack(); renderRing(); renderCenter(); renderDock(); renderMe(); renderLog(); }

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
  chips($('role-chips'), [['random', T('random')], ...['merlin', 'percival', 'servant', 'morgana', 'assassin'].map(r => [r, I18N.role(r.toUpperCase()), isEvilRole(r.toUpperCase()) ? 'evil' : 'good'])], 'role');
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
    S = await api('/api/new', {role: setup.role, seat: setup.seat, baseline: setup.baseline, share: setup.share});
    sel.clear(); showErr(''); $('setup').close(); render();
  } catch (err) { $('setup-err').textContent = I18N.err(err.message); $('setup-err').hidden = false; }
  finally { busy = false; $('start').disabled = false; $('start').textContent = T('start'); }
});
$('cancel').addEventListener('click', () => $('setup').close());
$('setup').addEventListener('cancel', e => { if (!S.active) e.preventDefault(); });
$('new-btn').addEventListener('click', openSetup);
$('rules-btn').addEventListener('click', () => $('rules').showModal());
$('rules-ok').addEventListener('click', () => $('rules').close());
$('auto').checked = pref('avalon_autoplay', 'on') === 'on';
$('auto').addEventListener('change', () => { setPref('avalon_autoplay', $('auto').checked ? 'on' : 'off'); renderDock(); });

async function downloadReplay() {
  try {
    const r = await api('/api/replay');
    const url = URL.createObjectURL(new Blob([JSON.stringify(r, null, 2)], {type: 'application/json'}));
    const a = el('a'); a.href = url; a.download = `avalon-seed${S.baseline}-${S.id}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) { showErr(e.message); }
}

I18N.onChange(() => { render(); if ($('setup').open) renderSetup(); });
render();
api('/api/state').then(s => {
  S = s; render();
  if (!S.active) { setup.baseline = S.default_baseline || 0; openSetup(); }
}).catch(e => showErr(e.message));
