'use strict';
const $ = id => document.getElementById(id);
let state = {active:false}, selected = new Set(), busy = false;
const shield = '<svg viewBox="0 0 32 38" aria-hidden="true"><path d="M16 2 28 7v12c0 7-7 13-12 17C11 32 4 26 4 19V7Z"/><path d="M16 10v15M10 17h12"/></svg>';
const phaseNames = {PROPOSE:'组队', VOTE:'表决', QUEST:'任务', ASSASSINATE:'刺杀', DONE:'终局'};
const goals = {梅林:'帮助好人完成三次任务，同时隐藏身份，避免被刺客认出。',派西维尔:'分辨梅林与莫甘娜，帮助好人组建可靠的任务队伍。',忠臣:'根据组队、投票和任务结果推理，争取三次任务成功。',莫甘娜:'隐藏身份，干扰队伍判断，让三次任务失败。',刺客:'破坏任务，并观察谁可能是梅林。必要时用最后一次刺杀翻盘。'};
function text(id, value) { $(id).textContent = value; }
function showError(message) { text('error', message); $('error').hidden = !message; }
async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '请求失败，请重试。');
  return result;
}
function openSetup() {
  $('cancel-setup').hidden = !state.active;
  document.querySelectorAll('input[name=share]').forEach(r=>{r.checked = state.share!=null && (r.value==='yes')===state.share;});
  $('setup-error').hidden = true;
  $('setup-dialog').showModal();
}
function button(label, cls, click, disabled=false) {
  const el = document.createElement('button'); el.className=cls; el.textContent=label; el.disabled=disabled; el.addEventListener('click',click); return el;
}
async function move(action) {
  if (busy) return;
  busy=true; document.body.classList.add('busy'); $('actions').setAttribute('aria-busy','true'); showError('');
  try {
    state=await api(action === null ? '/api/advance' : '/api/action', {id:state.id,revision:state.revision,...(action===null?{}:{action})});
    selected.clear(); render();
  } catch (error) { showError(error.message); }
  finally {busy=false;document.body.classList.remove('busy');$('actions').setAttribute('aria-busy','false');}
}
function renderIdentity() {
  if (!state.active) return;
  $('identity').classList.toggle('evil',state.side==='evil');
  text('role-name',state.role);text('role-meta',`${state.human} 号座位 · ${state.side==='good'?'好人':'坏人'}阵营`);
  const info=$('private-info'); info.replaceChildren();
  let label='你没有额外的身份线索。请观察公开记录。', list=[];
  if (state.known_evil.length) {label='你知道以下座位属于坏人阵营：'; list=state.known_evil;}
  if (state.candidates.length) {label='以下两人是梅林与莫甘娜，你无法区分：';list=state.candidates;}
  const p=document.createElement('div');p.textContent=label;info.append(p);
  if(list.length){const row=document.createElement('div');row.className='knowledge-seats';list.forEach(s=>{const tag=document.createElement('span');tag.textContent=`${s} 号`;row.append(tag);});info.append(row);}
  const cards=state.own_cards.map((c,i)=>c>=0?`第${i+1}轮：${c?'失败':'成功'}`:null).filter(Boolean);
  if(cards.length){const p=document.createElement('p');p.className='knowledge-label';p.textContent=`你的历史出牌 · ${cards.join(' / ')}`;info.append(p);}
  text('role-goal',goals[state.role]);
  text('model-tag',`冻结策略 · Baseline seed ${state.baseline}`);
  text('share-status',state.share?'本局数据将在结束后共享':'本局数据不共享');
}
function renderQuests() {
  const quests=state.active?state.quests:[2,3,4,3,4].map(size=>({size,result:-1}));
  $('quests').innerHTML=quests.map((q,i)=>`<div class="quest ${q.result===0?'success':q.result===1?'fail':state.active&&!state.done&&state.quest===i+1?'current':''}" title="第 ${i+1} 轮：${q.result<0?'尚未完成':q.result===0?'任务成功':q.fails+' 张失败牌'}"><div class="quest-orb">${q.result<0?i+1:q.result===0?'✓':'×'}</div><small>${q.size} 人任务</small></div>`).join('');
  $('score').innerHTML=`好人 ${state.successes||0} <i>:</i> ${state.fails||0} 坏人`;
}
function canPick(s) {
  if(!state.active||!state.human_turn||state.done)return false;
  if(state.phase==='PROPOSE')return true;
  return state.phase==='ASSASSINATE' && state.options.some(o=>o.target===s);
}
function renderSeats() {
  $('seats').replaceChildren();
  for(let s=1;s<=6;s++){
    const human=state.active&&s===state.human, leader=state.active&&!state.done&&['PROPOSE','VOTE','QUEST'].includes(state.phase)&&s===state.leader;
    const onTeam=state.active&&state.team.includes(s), pickable=canPick(s), isSelected=selected.has(s);
    let detail=human?'真人玩家':'身份未知', tone='';
    if(state.active){
      if(state.known_evil.includes(s)){detail='已知坏人';tone='known';}
      if(state.candidates.includes(s)){detail='梅林 / 莫甘娜';tone='known';}
      if(state.event.kind==='vote'){const yes=state.event.votes[s-1];detail=yes?'✓ 赞成':'× 反对';tone=yes?'approve':'reject';}
      if(state.done){detail=state.result.roles[s-1];tone='known';}
    }
    const seat=document.createElement('button');seat.className=`seat ${human?'human':''} ${leader?'leader':''} ${onTeam?'on-team':''} ${isSelected?'selected':''} ${pickable?'pickable':''}`;
    seat.type='button';seat.disabled=!pickable;seat.setAttribute('aria-label',`${s} 号${human?'，你':''}${leader?'，队长':''}，${detail}`);
    seat.setAttribute('aria-pressed',String(isSelected));
    seat.innerHTML=`<div class="avatar">${shield}${leader?'<span class="leader-pin" title="队长">♛</span>':''}${isSelected||onTeam?'<span class="selected-pin">✓</span>':''}</div><div class="seat-name">${s} 号<em>${human?'你':'AI'}</em></div><div class="seat-detail ${tone}">${detail}</div>`;
    seat.addEventListener('click',()=>{
      if(busy)return;
      if(state.phase==='ASSASSINATE'){selected.clear();selected.add(s);}
      else if(selected.has(s)) selected.delete(s);
      else if(selected.size<state.team_size)selected.add(s);
      else {showError(`队伍已满 ${state.team_size} 人，请先取消一个座位。`);return;}
      showError('');renderSeats();renderCenter();renderActions();
    });$('seats').append(seat);
  }
}
function renderCenter() {
  if(!state.active)return;
  text('round-label',state.done?'对局已结束':state.phase==='ASSASSINATE'?'最后的刺杀':`第 ${state.quest} 轮任务`);
  text('attempt-label',state.done?'全部身份已揭晓':['PROPOSE','VOTE'].includes(state.phase)?`第 ${state.attempt} / 5 次提案`:phaseNames[state.phase]);
  text('table-title',state.done?(state.result.good_won?'好人获胜':'坏人获胜'):state.phase==='ASSASSINATE'?'最后一击':`第 ${state.quest} 轮`);
  text('table-subtitle',state.done?state.result.reason:state.phase==='PROPOSE'?(state.human_turn?'点击座位，组建你的队伍':`等待 ${state.leader} 号队长组队`):state.phase==='VOTE'?'队伍已提议，等待表决':state.phase==='QUEST'?'任务成员秘密出牌':'谁才是真正的梅林？');
  text('selection-count',state.human_turn&&state.phase==='PROPOSE'?`已选择 ${selected.size} / ${state.team_size} 人`:state.human_turn&&state.phase==='ASSASSINATE'?(selected.size?`目标：${[...selected][0]} 号`:'选择一个刺杀目标'):state.team.length?`任务队伍 · ${state.team.join('、')} 号`:'THE SILENT TABLE');
}
function renderHistory() {
  const history=state.history||[];text('history-count',history.length);
  if(!history.length){$('history').innerHTML='<div class="history-empty"><div class="empty-glyph">◇</div><p>故事尚未开始</p><small>队伍、投票与任务结果<br>将在这里逐一留下记录</small></div>';return;}
  $('history').innerHTML=[...history].reverse().map(h=>{
    const result=state.quests[h.quest-1], questResult=h.approved&&result.result>=0;
    return `<article class="history-item"><div class="history-meta"><span>任务 ${h.quest} · 提案 ${h.attempt}</span><span>${h.votes?(h.approved?'已通过':'已否决'):'待表决'}</span></div><p>${h.leader} 号队长提议</p><div class="team-chips">${h.team.map(s=>`<span class="team-chip">${s}</span>`).join('')}</div>${h.votes?`<div class="vote-dots">${h.votes.map((yes,i)=>`<span class="${yes?'yes':'no'}" title="${i+1} 号${yes?'赞成':'反对'}">${i+1}</span>`).join('')}</div><div class="vote-summary">${h.votes.filter(Boolean).length} 票赞成 / ${h.votes.filter(v=>!v).length} 票反对</div>`:''}${questResult?`<div class="quest-history-result ${result.result?'failed':''}">${result.result?'× 任务失败':'✓ 任务成功'} · ${result.fails} 张失败牌</div>`:''}</article>`;
  }).join('');
}
function renderActions() {
  const box=$('actions');box.replaceChildren();
  const phases=['PROPOSE','VOTE','QUEST','ASSASSINATE'];
  $('steps').innerHTML=phases.map((p,i)=>`<span class="${state.phase===p?'active':''}">0${i+1} ${phaseNames[p]}</span>`).join('');
  if(!state.active){$('event-banner').hidden=true;box.append(button('选择身份，入座 →','primary-button',openSetup));return;}
  $('event-banner').hidden=false;text('event-banner',state.event.text);
  $('event-banner').classList.toggle('bad',(state.event.kind==='quest'&&state.event.fails>0)||(state.event.kind==='vote'&&!state.event.passed));
  text('turn-badge',state.done?'对局结束':state.human_turn?'轮到你了':'AI 回合');
  text('action-kicker',state.done?'THE STORY ENDS':state.human_turn?'YOUR NEXT MOVE':'WATCH & OBSERVE');
  if(state.done){
    text('action-title',state.result.human_won?'你与同伴取得了胜利。':'这一局，另一方更胜一筹。');text('action-hint',`${state.result.reason}。${state.share?'本局数据已共享，感谢参与！':''}现在可以查看全部身份，或下载完整复盘。`);
    box.append(button('再来一局 →','primary-button',openSetup),button('下载复盘','secondary-button',downloadReplay));return;
  }
  if(!state.human_turn){
    const title=state.phase==='PROPOSE'?`观察 ${state.leader} 号如何组队`:state.phase==='QUEST'?'这次任务，你不在队伍中':'等待刺客的最终选择';
    text('action-title',title);text('action-hint','点击继续后，AI 将完成这一阶段。公开结果会保留在右侧纪事中。');
    box.append(button(state.phase==='PROPOSE'?'继续 · 查看 AI 组队 →':state.phase==='QUEST'?'继续 · 查看任务结果 →':'继续 · 查看刺杀结果 →','primary-button',()=>move(null)));return;
  }
  if(state.phase==='PROPOSE'){
    text('action-title',`你是队长，请选出 ${state.team_size} 位队员`);text('action-hint','点击牌桌上的座位选择，再次点击可取消。你也可以选择自己。');
    const team=[...selected].sort((a,b)=>a-b),option=state.options.find(o=>JSON.stringify(o.team)===JSON.stringify(team));
    box.append(button(`确认队伍${team.length?' · '+team.join('、')+' 号':''}`,'primary-button',()=>move(option.action),!option));
  }else if(state.phase==='VOTE'){
    text('action-title','你信任这支队伍吗？');text('action-hint',`任务队伍：${state.team.join('、')} 号。至少 4 票赞成才通过。${state.attempt===5?'注意：这是最后一次提案，否决将使坏人直接获胜。':'所有玩家投票后，一起公布结果。'}`);
    for(const o of state.options)box.append(button(o.label==='赞成'?'✓ 赞成这支队伍':'× 反对这支队伍',o.label==='赞成'?'primary-button':'danger-button',()=>move(o.action)));
  }else if(state.phase==='QUEST'){
    text('action-title',state.side==='good'?'为任务投出成功牌':'秘密出牌，决定任务走向');text('action-hint',state.side==='good'?'你是好人，只能出成功牌。公开结果不会显示每个人的出牌。':'你可以选择帮助任务成功，也可以投出失败牌。只要一张失败牌，任务就会失败。');
    for(const o of state.options)box.append(button(o.label==='任务成功'?'✓ 投出成功牌':'× 投出失败牌',o.label==='任务成功'?'primary-button':'danger-button',()=>move(o.action)));
  }else{
    text('action-title','谁才是梅林？');text('action-hint','点击一位可刺杀的玩家，再确认。命中梅林，坏人阵营将赢得最终胜利。');
    const target=[...selected][0],option=state.options.find(o=>o.target===target);
    box.append(button(target?`确认刺杀 ${target} 号`:'先选择刺杀目标','danger-button',()=>move(option.action),!option));
  }
}
function render(){renderIdentity();renderQuests();renderSeats();renderCenter();renderHistory();renderActions();}
async function downloadReplay(){try{const r=await api('/api/replay');const url=URL.createObjectURL(new Blob([JSON.stringify(r,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`avalon-seed${state.baseline}-${state.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){showError(e.message);}}
$('new-button').addEventListener('click',openSetup);
$('cancel-setup').addEventListener('click',()=>$('setup-dialog').close());
$('setup-dialog').addEventListener('cancel',e=>{if(!state.active||busy)e.preventDefault();});
$('rules-button').addEventListener('click',()=>$('rules-dialog').showModal());
$('close-rules').addEventListener('click',()=>$('rules-dialog').close());
$('setup-form').addEventListener('submit',async e=>{
  e.preventDefault();if(busy)return;busy=true;$('start-button').disabled=true;text('start-button','正在加载模型，准备牌桌…');$('setup-error').hidden=true;
  try{const share=document.querySelector('input[name=share]:checked');if(!share)throw new Error('请先选择是否共享本局数据。');
    state=await api('/api/new',{role:$('role-select').value,seat:Number($('seat-select').value),baseline:Number($('baseline-select').value),share:share.value==='yes'});selected.clear();showError('');render();$('setup-dialog').close();}
  catch(error){text('setup-error',error.message);$('setup-error').hidden=false;}
  finally{busy=false;$('start-button').disabled=false;text('start-button','入座，开始游戏 →');}
});
render();
api('/api/state').then(s=>{state=s;render();if(!state.active){$('baseline-select').value=String(state.default_baseline);openSetup();}}).catch(e=>showError('无法连接本机牌桌，请确认 Python 服务仍在运行。'+e.message));
