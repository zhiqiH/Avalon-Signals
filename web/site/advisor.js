'use strict';
// Assassin helper: record a real game on the round table, then ask the frozen AI Assassin for Merlin.
const $ = id => document.getElementById(id);
const T = I18N.t;
const BASE = location.pathname.startsWith('/advisor') ? '/advisor' : '';  // served alone or under the study site
const SIZES = [2, 3, 4, 3, 4], NEED = 4;
const CROWN = '<svg viewBox="0 0 16 16"><path d="M2 12.5h12l1-7.5-4 3-3-5.5-3 5.5-4-3z"/></svg>';
let rec = null, draft = {stage: 'team', team: [], votes: {}}, busy = false;

function consent() { try { const v = localStorage.getItem('avalon_share'); return v === 'yes' ? true : v === 'no' ? false : null; } catch (e) { return null; } }
function remember() { try { localStorage.setItem('advisor_game', rec.id); } catch (e) {} }
function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; }
function btn(label, cls, onClick, disabled = false) { const b = el('button', `btn ${cls}`); b.type = 'button'; b.textContent = label; b.disabled = disabled; b.addEventListener('click', onClick); return b; }
function showErr(msg) { $('err').textContent = msg ? I18N.err(msg) : ''; $('err').hidden = !msg; }
const pct = p => `${Math.round(p * 100)}%`;

async function api(path, body) {
  const r = await fetch(BASE + path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || 'Error');
  return j;
}
async function save() { try { rec = await api('/api/save', rec); showErr(''); } catch (e) { showErr(e.message); } render(); }

// Game progress from the record alone (same rules as the server's replay).
function derive(r) {
  const d = {rows: [], quest: 0, attempt: 0, leader: r.first_leader, succ: 0, fail: 0, phase: 'setup', results: []};
  if (r.first_leader == null) return d;
  d.phase = 'propose';
  let fi = 0;
  for (const p of r.proposals) {
    const row = {quest: d.quest, attempt: d.attempt, leader: d.leader, team: p.team, approvals: p.approvals, approved: p.approvals.length >= NEED, fails: null};
    d.rows.push(row); d.leader = d.leader % 6 + 1;
    if (row.approved) {
      if (fi >= r.quest_fails.length) { d.phase = 'quest'; return d; }
      row.fails = r.quest_fails[fi++]; d.results.push(row.fails);
      row.fails > 0 ? d.fail++ : d.succ++; d.quest++; d.attempt = 0;
      if (d.fail === 3) { d.phase = 'evil3'; return d; }
      if (d.succ === 3) { d.phase = 'assassinate'; return d; }
    } else { d.attempt++; if (d.attempt === 5) { d.phase = 'evil5'; return d; } }
  }
  return d;
}
const ended = d => ['assassinate', 'evil3', 'evil5'].includes(d.phase);

function seatClick(s, d) {
  if (busy) return;
  if (d.phase === 'setup') { rec.first_leader = s; save(); return; }
  if (d.phase === 'propose') {
    const size = SIZES[d.quest];
    if (draft.stage === 'team') {
      const i = draft.team.indexOf(s);
      if (i >= 0) draft.team.splice(i, 1); else if (draft.team.length < size) draft.team.push(s);
      if (draft.team.length === size) draft.stage = 'votes';
    } else {
      const v = draft.votes[s];
      if (v === undefined) draft.votes[s] = true; else if (v) draft.votes[s] = false; else delete draft.votes[s];
    }
    render(); return;
  }
  if (d.phase === 'assassinate') {
    if (rec.assassin === s) rec.assassin = null;
    else if (rec.morgana === s) rec.morgana = null;
    else if (rec.assassin == null) rec.assassin = s;
    else if (rec.morgana == null) rec.morgana = s;
    else return;
    delete rec.advice; rec.actual_merlin = null; rec.human_target = null;
    save();
  }
}

function renderTrack(d) {
  $('quests').innerHTML = SIZES.map((size, i) => {
    const f = d.results[i];
    const cls = f == null ? (d.phase === 'propose' || d.phase === 'quest') && d.quest === i ? 'now' : '' : f > 0 ? 'bad' : 'ok';
    return `<span class="q ${cls}">${f == null ? size : f > 0 ? '✗' : '✓'}</span>`;
  }).join('');
  const rejected = d.phase === 'propose' ? d.attempt : 0;
  $('tries').innerHTML = [0, 1, 2, 3, 4].map(i => `<b class="${i < rejected ? 'on' : ''}"></b>`).join('');
}

function renderRing(d) {
  const ring = $('ring');
  ring.querySelectorAll('.seat').forEach(n => n.remove());
  const adv = rec.advice, last = d.rows[d.rows.length - 1];
  const team = d.phase === 'propose' ? draft.team : d.phase === 'quest' ? last.team : [];
  const pickable = d.phase === 'setup' || d.phase === 'propose' || (d.phase === 'assassinate' && !(rec.assassin && rec.morgana && adv));
  for (let s = 1; s <= 6; s++) {
    const b = el('button', `seat p${s - 1}`);
    b.type = 'button';
    if (pickable || (d.phase === 'assassinate' && (s === rec.assassin || s === rec.morgana))) b.classList.add('pick');
    if (team.includes(s)) b.classList.add(d.phase === 'propose' && draft.stage === 'team' ? 'sel' : 'team');
    let badges = '', cap = '', capCls = '';
    if ((d.phase === 'propose' || d.phase === 'quest') && s === (d.phase === 'quest' ? last.leader : d.leader)) badges += `<span class="badge tl lead">${CROWN}</span>`;
    if (d.phase === 'propose' && draft.stage === 'votes' && s in draft.votes) badges += draft.votes[s] ? '<span class="badge tr yes">✓</span>' : '<span class="badge tr no">✗</span>';
    if (d.phase === 'assassinate') {
      if (s === rec.assassin) { cap = I18N.role('ASSASSIN'); capCls = 'evil'; b.classList.add('evil'); }
      if (s === rec.morgana) { cap = I18N.role('MORGANA'); capCls = 'evil'; b.classList.add('evil'); }
      if (adv && adv.mean[s - 1] > 0) badges += `<span class="badge br ${s === adv.pick ? 'dot' : 'pct'}">${pct(adv.mean[s - 1])}</span>`;
      if (adv && s === adv.pick) b.classList.add('sel');
      if (rec.actual_merlin === s) { cap = I18N.role('MERLIN'); capCls = 'good'; }
    }
    b.innerHTML = `<span class="disc">${s}${badges}</span><span class="cap ${capCls}">${cap || '&nbsp;'}</span>`;
    b.setAttribute('aria-label', T('seat', {n: s}));
    b.addEventListener('click', () => seatClick(s, d));
    ring.append(b);
  }
}

function renderCenter(d) {
  let big = '', sub = '';
  if (d.phase === 'setup') { big = T('a.first'); sub = T('a.firstSub'); }
  else if (d.phase === 'propose') {
    big = T('quest', {q: d.quest + 1});
    sub = draft.stage === 'team' ? `${T('a.lead', {l: d.leader, n: SIZES[d.quest]})} · ${draft.team.length}/${SIZES[d.quest]}` : T('a.votesSub');
    if (d.attempt === 4) sub += ` · ${T('lastTry')}`;
  } else if (d.phase === 'quest') { big = T('a.result', {q: d.quest + 1}); sub = I18N.seats(d.rows[d.rows.length - 1].team); }
  else if (d.phase === 'assassinate') {
    if (rec.advice) { big = T('a.pick', {n: rec.advice.pick}); sub = rec.actual_merlin ? '' : T('a.merlin'); }
    else { big = rec.assassin == null ? T('a.assassin') : rec.morgana == null ? T('a.morgana') : T('whoMerlin'); }
  } else { big = T('win.evil'); sub = T(d.phase === 'evil3' ? 'o.evil3' : 'o.evil5'); }
  $('c-big').textContent = big;
  $('c-sub').textContent = sub;
}

function renderDock(d) {
  const dock = $('dock');
  dock.replaceChildren();
  if (d.phase === 'propose' && draft.stage === 'votes') {
    const rest = v => { for (let s = 1; s <= 6; s++) if (!(s in draft.votes)) draft.votes[s] = v; render(); };
    const n = Object.keys(draft.votes).length, y = Object.values(draft.votes).filter(Boolean).length;
    dock.append(btn(`✓ ${T('a.restYes')}`, 'ghost', () => rest(true)), btn(`✗ ${T('a.restNo')}`, 'ghost', () => rest(false)),
      btn(n === 6 ? `${T('save')} · ${y} : ${6 - y}` : T('save'), 'primary wide', () => {
        rec.proposals.push({team: [...draft.team].sort((a, b) => a - b), approvals: Object.keys(draft.votes).filter(s => draft.votes[s]).map(Number).sort((a, b) => a - b)});
        draft = {stage: 'team', team: [], votes: {}}; save();
      }, n !== 6),
      btn(`← ${I18N.seats([...draft.team].sort((a, b) => a - b))}`, 'ghost slim', () => { draft.stage = 'team'; draft.votes = {}; render(); }));
  } else if (d.phase === 'quest') {
    const size = d.rows[d.rows.length - 1].team.length;
    dock.append(btn(`✓ ${T('success')}`, 'good wide', () => { rec.quest_fails.push(0); save(); }));
    for (let f = 1; f <= Math.min(size, 2); f++) dock.append(btn(`✗ ${T('a.nfail', {f})}`, 'evil', () => { rec.quest_fails.push(f); save(); }));
  } else if (d.phase === 'assassinate' && !rec.advice) {
    dock.append(btn(T('a.ask'), 'primary wide', async e => {
      e.target.disabled = true; e.target.textContent = T('a.thinking'); busy = true;
      try { rec = await api('/api/advise', rec); showErr(''); } catch (err) { showErr(err.message); }
      busy = false; render();
    }, !(rec.assassin && rec.morgana)));
  } else if (ended(d) && (d.phase !== 'assassinate' || rec.actual_merlin)) {
    dock.append(btn(T('newGame'), 'primary wide', newGame));
  }
}

function seatChips(label, key, seats) {
  const box = el('div', 'row');
  box.append(el('p', 'label', label));
  const chips = el('div', 'chips');
  for (const s of seats) {
    const c = el('button', 'chip'); c.type = 'button'; c.textContent = s; c.setAttribute('aria-pressed', String(rec[key] === s));
    c.addEventListener('click', () => { rec[key] = rec[key] === s ? null : s; save(); });
    chips.append(c);
  }
  box.append(chips);
  return box;
}
function shareRow() {
  const box = el('div', 'row');
  box.append(el('p', 'label', T('setup.share')));
  const chips = el('div', 'chips');
  for (const [v, label, cls] of [[true, T('share.yes'), 'good'], [false, T('share.no'), '']]) {
    const c = el('button', `chip ${cls}`); c.type = 'button'; c.textContent = label; c.setAttribute('aria-pressed', String(rec.share === v));
    c.addEventListener('click', () => { rec.share = v; save(); });
    chips.append(c);
  }
  box.append(chips);
  return box;
}
function renderResult(d) {
  const card = $('result');
  card.hidden = !ended(d);
  if (card.hidden) return;
  card.replaceChildren();
  const adv = rec.advice;
  if (adv) {
    const bars = el('div', 'bars row');
    for (const s of adv.ranking) {
      const row = el('div', `barrow ${s === adv.pick ? 'top' : ''}`, `<span>${s}</span><span class="tr"><span class="fill"></span></span><span>${pct(adv.mean[s - 1])}</span>`);
      row.querySelector('.fill').style.width = `${(adv.mean[s - 1] * 100).toFixed(1)}%`;
      bars.append(row);
    }
    card.append(el('p', 'label', T('a.pick', {n: adv.pick})), bars);
    card.append(el('p', 'muted small row', `${T('a.models')}: ${Object.entries(adv.seed_picks).map(([k, v]) => `seed ${k} → ${v}`).join(' · ')}`));
    if (adv.variants.length) card.append(el('p', 'muted small row', T('a.uncertain')));
    const good = [1, 2, 3, 4, 5, 6].filter(s => s !== rec.assassin && s !== rec.morgana);
    card.append(seatChips(T('a.merlin'), 'actual_merlin', good), seatChips(T('a.human'), 'human_target', good));
    if (rec.actual_merlin) {
      const line = (who, shot) => `${who} <b class="${shot === rec.actual_merlin ? 'res-ok' : 'res-bad'}">${T(shot === rec.actual_merlin ? 'hit' : 'miss')}</b>`;
      card.append(el('p', 'row', [line('AI', adv.pick), rec.human_target ? line(T('a.human').replace(/[？?]$/, ''), rec.human_target) : ''].filter(Boolean).join(' · ')));
    }
  }
  card.append(shareRow());
}

function renderLog(d) {
  if (!d.rows.length) { $('log').innerHTML = `<p class="empty">${T('empty')}</p>`; return; }
  const rows = [...d.rows].reverse().map(r => {
    const res = !r.approved ? `<span class="muted">${T('rejected')}</span>` : r.fails == null ? '…' : r.fails ? `<span class="res-bad">✗ ${r.fails}</span>` : '<span class="res-ok">✓</span>';
    const votes = `<span class="votes">${[1, 2, 3, 4, 5, 6].map(s => `<span class="${r.approvals.includes(s) ? 'y' : ''}">${s}</span>`).join('')}</span>`;
    return `<tr><td>${r.quest + 1}-${r.attempt + 1}</td><td>${r.leader}</td><td><span class="team">${r.team.map(s => `<span>${s}</span>`).join('')}</span></td><td>${votes}</td><td>${res}</td></tr>`;
  }).join('');
  $('log').innerHTML = `<table class="log"><tr><th>${T('col.q')}</th><th>${T('col.lead')}</th><th>${T('col.team')}</th><th>${T('col.votes')}</th><th>${T('col.res')}</th></tr>${rows}</table>`;
}

function render() {
  if (!rec) return;
  const d = derive(rec);
  renderTrack(d); renderRing(d); renderCenter(d); renderDock(d); renderResult(d); renderLog(d);
  $('undo').disabled = rec.first_leader == null && !draft.team.length;
  if (document.activeElement !== $('note')) $('note').value = rec.note || '';
}

async function newGame() {
  const j = await api('/api/new');
  rec = {id: j.id, created_utc: j.created_utc, first_leader: null, proposals: [], quest_fails: [], assassin: null, morgana: null,
         human_target: null, actual_merlin: null, note: '', share: consent()};
  draft = {stage: 'team', team: [], votes: {}}; showErr(''); remember(); render();  // saved on the first entry
}
async function loadGame(id) { rec = await api('/api/game?id=' + id); draft = {stage: 'team', team: [], votes: {}}; showErr(''); remember(); render(); }

$('undo').addEventListener('click', () => {
  if (draft.team.length || Object.keys(draft.votes).length) { draft = {stage: 'team', team: [], votes: {}}; render(); return; }
  const rows = derive(rec).rows, last = rows[rows.length - 1];
  if (rec.assassin || rec.morgana) { rec.assassin = rec.morgana = rec.actual_merlin = rec.human_target = null; delete rec.advice; }
  else if (last && last.fails != null) rec.quest_fails.pop();
  else if (last) rec.proposals.pop();
  else rec.first_leader = null;
  delete rec.advice; save();
});
$('note').addEventListener('change', () => { rec.note = $('note').value; save(); });
$('new-btn').addEventListener('click', () => {
  const d = derive(rec);
  if (rec.first_leader != null && !ended(d) && !confirm(T('a.newQ'))) return;
  newGame();
});
document.querySelectorAll('[data-close]').forEach(b => b.addEventListener('click', () => b.closest('dialog').close()));

function statLine(s) {
  if (!s.judged) return `<p class="muted small">${T('a.games', {n: s.games})}</p>`;
  let t = `${T('a.stat', {h: s.ai_hits, n: s.judged})} <span class="muted">(${T('chance')})</span>`;
  if (s.human_judged) t += ` · ${T('a.statHuman', {h: s.human_hits, n: s.human_judged})}`;
  const seeds = Object.entries(s.per_seed || {}).map(([k, v]) => `seed ${k}: ${v.hits}/${v.n}`).join(' · ');
  return `<p><b>${t}</b></p><p class="muted small">${T('a.games', {n: s.games})} · ${seeds}</p>`;
}
async function openHistory() {
  const j = await api('/api/games');
  const body = $('hist-body');
  body.innerHTML = `<p class="label">${T('a.all')}</p>${statLine(j.shared)}<p class="label">${T('a.mine')}</p>${statLine(j.stats)}`;
  if (j.games.length) {
    const table = el('table', 'log');
    table.innerHTML = j.games.map(g => `<tr><td>${(g.created_utc || '').slice(5, 10)}</td><td>${T('o.' + g.outcome_key)}</td>
      <td>${g.pick ? `AI ${g.pick}${g.ai_hit === undefined ? '' : g.ai_hit ? ' <span class="res-ok">✓</span>' : ' <span class="res-bad">✗</span>'}` : ''}</td>
      <td>${g.share ? `<span class="res-ok">${T('shared')}</span>` : ''}</td>
      <td><button class="btn ghost slim" data-open="${g.id}">${T('open')}</button><button class="btn ghost slim" data-del="${g.id}">✕</button></td></tr>`).join('');
    body.append(table);
    table.querySelectorAll('[data-open]').forEach(b => b.addEventListener('click', async () => { await loadGame(b.dataset.open); $('hist').close(); }));
    table.querySelectorAll('[data-del]').forEach(b => b.addEventListener('click', async () => {
      if (!confirm(T('a.delQ'))) return;
      await api('/api/delete', {id: b.dataset.del});
      if (rec.id === b.dataset.del) await newGame();
      openHistory();
    }));
  }
  if (!$('hist').open) $('hist').showModal();
}
$('hist-btn').addEventListener('click', openHistory);
$('code-btn').addEventListener('click', async () => {
  const {code} = await api('/api/me');
  $('code-value').textContent = code.match(/.{4}/g).join('-');
  $('code-err').hidden = true; $('code-in').value = '';
  $('code').showModal();
});
$('code-go').addEventListener('click', async () => {
  try { await api('/api/code', {code: $('code-in').value}); $('code').close(); await newGame(); openHistory(); }
  catch (e) { $('code-err').textContent = I18N.err(e.message); $('code-err').hidden = false; }
});

I18N.onChange(render);
(async () => {
  let id = null;
  try { id = localStorage.getItem('advisor_game'); } catch (e) {}
  if (id) { try { await loadGame(id); return; } catch (e) {} }
  newGame();
})();
