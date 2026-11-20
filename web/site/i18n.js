'use strict';
// Shared language module: Chinese by default, English on request; the choice is kept in localStorage.
const I18N = (() => {
  const dict = {
    zh: {
      home:'首页', rules:'规则', newGame:'新局', code:'记录码', history:'历史', undo:'撤销',
      'share.q':'共享数据用于研究？', 'share.yes':'同意', 'share.no':'不同意', 'share.note':'匿名，不含姓名或 IP', shared:'已共享',
      'role.SERVANT':'忠臣', 'role.MERLIN':'梅林', 'role.PERCIVAL':'派西维尔', 'role.MORGANA':'莫甘娜', 'role.ASSASSIN':'刺客',
      'side.good':'好人', 'side.evil':'坏人', seat:'{n} 号', you:'你', ai:'AI', random:'随机',
      'home.title':'阿瓦隆 · 真人实验', 'home.sub':'六人局 · AI 来自冻结的训练策略',
      'mode.play':'人机对局', 'mode.play.d':'你和 5 位 AI，无发言', 'mode.assist':'刺客辅助', 'mode.assist.d':'记录线下真人局，AI 帮你找梅林',
      go:'进入 →', 'home.hint':'先选择是否共享数据',
      'phase.PROPOSE':'组队', 'phase.VOTE':'投票', 'phase.QUEST':'任务', 'phase.ASSASSINATE':'刺杀',
      quest:'第 {q} 轮', pick:'选 {n} 人', picked:'已选 {k} / {n}', leaderPicks:'{n} 号组队中', voteTeam:'是否赞成？',
      playCard:'出牌', questOn:'任务进行中', whoMerlin:'谁是梅林？', assnWait:'刺客选择中', lastTry:'最后一次提案',
      confirmTeam:'确认队伍', approve:'赞成', reject:'反对', success:'成功', fail:'失败', kill:'刺杀 {n} 号', pickTarget:'先选目标',
      'continue':'继续', auto:'自动', 'win.good':'好人胜', 'win.evil':'坏人胜', won:'你赢了', lost:'你输了',
      'reason.ASSASSIN_MISSED':'刺杀落空', 'reason.ASSASSIN_HIT':'刺中梅林', 'reason.THREE_FAILS':'三次任务失败', 'reason.FIVE_REJECTIONS':'连续五次否决',
      again:'再来一局', replay:'下载复盘',
      'ev.start':'身份已分配', 'ev.proposal':'{l} 号提名 {team}', 'ev.vote':'{y} 赞成 · {n} 反对 · {r}', passed:'通过', rejected:'否决',
      'ev.ok':'第 {q} 轮成功', 'ev.bad':'第 {q} 轮失败 · {f} 张失败牌', 'ev.assn':'刺杀 {n} 号',
      'me.evil':'坏人', 'me.cands':'梅林 / 莫甘娜', 'me.none':'只有公开信息', 'me.cards':'你的出牌',
      log:'记录', empty:'暂无', 'col.q':'轮', 'col.lead':'队长', 'col.team':'队伍', 'col.votes':'投票', 'col.res':'结果',
      'setup.title':'新的一局', 'setup.role':'身份', 'setup.seat':'座位', 'setup.ai':'对手模型', 'setup.share':'共享本局数据',
      start:'开始', cancel:'取消', loading:'加载中…', ok:'知道了',
      'rules.1':'好人：梅林、派西维尔、2 名忠臣；坏人：莫甘娜、刺客。',
      'rules.2':'队长组队，全员投票，4 票赞成通过；连续 5 次否决坏人胜。',
      'rules.3':'队员秘密出牌，1 张失败即任务失败。五轮人数 2 · 3 · 4 · 3 · 4。',
      'rules.4':'3 次失败坏人胜；3 次成功后刺客猜梅林，猜中则坏人胜。',
      'rules.5':'梅林知道坏人；坏人互相认识；派西维尔看到梅林和莫甘娜，但分不清。',
      'a.title':'刺客辅助', 'a.first':'首任队长？', 'a.firstSub':'座位按队长轮转顺序编号', 'a.lead':'{l} 号组队 · 选 {n} 人',
      'a.votes':'点座位记录投票', 'a.votesSub':'点一下赞成，再点反对', 'a.restYes':'其余赞成', 'a.restNo':'其余反对', save:'记录',
      'a.result':'第 {q} 轮结果', 'a.nfail':'{f} 失败', 'a.assassin':'刺客是？', 'a.morgana':'莫甘娜是？', 'a.ask':'AI 判断', 'a.thinking':'计算中…',
      'a.pick':'AI 刺杀 {n} 号', 'a.models':'三个模型', 'a.merlin':'真梅林？', 'a.human':'人类刺客刺了？', hit:'命中', miss:'未中',
      'a.evilWon':'坏人已获胜', 'a.variant':'第 {q} 轮失败牌来自{who}，概率 {p}', 'a.uncertain':'失败牌归属不确定，已对各种情况取平均',
      note:'备注', 'a.mine':'我的对局', 'a.all':'全体共享', 'a.stat':'AI 命中 {h} / {n}', 'a.statHuman':'人类 {h} / {n}', chance:'随机 25%',
      open:'打开', del:'删除', 'a.delQ':'删除这局记录？', 'a.newQ':'当前局未结束，开始新局？', 'a.codeTip':'在另一台设备输入此码，可查看同一批记录',
      'a.codePh':'输入记录码', 'a.switch':'切换', 'a.games':'{n} 局',
      'o.playing':'进行中', 'o.evil5':'五次否决', 'o.evil3':'三次失败', 'o.assassinate':'刺杀',
    },
    en: {
      home:'Home', rules:'Rules', newGame:'New', code:'Code', history:'History', undo:'Undo',
      'share.q':'Share data for research?', 'share.yes':'Yes', 'share.no':'No', 'share.note':'Anonymous · no names or IP', shared:'Shared',
      'role.SERVANT':'Servant', 'role.MERLIN':'Merlin', 'role.PERCIVAL':'Percival', 'role.MORGANA':'Morgana', 'role.ASSASSIN':'Assassin',
      'side.good':'Good', 'side.evil':'Evil', seat:'Seat {n}', you:'You', ai:'AI', random:'Random',
      'home.title':'Avalon · Human Study', 'home.sub':'Six players · AI from a frozen trained policy',
      'mode.play':'Play vs AI', 'mode.play.d':'You and five AI players, no chat', 'mode.assist':'Assassin Helper', 'mode.assist.d':'Log a real game; the AI finds Merlin',
      go:'Enter →', 'home.hint':'Choose a data option first',
      'phase.PROPOSE':'Team', 'phase.VOTE':'Vote', 'phase.QUEST':'Quest', 'phase.ASSASSINATE':'Assassination',
      quest:'Quest {q}', pick:'Pick {n}', picked:'{k} / {n} picked', leaderPicks:'Seat {n} is picking', voteTeam:'Approve this team?',
      playCard:'Play a card', questOn:'Quest in progress', whoMerlin:'Who is Merlin?', assnWait:'The Assassin is choosing', lastTry:'Last proposal',
      confirmTeam:'Confirm team', approve:'Approve', reject:'Reject', success:'Success', fail:'Fail', kill:'Assassinate {n}', pickTarget:'Pick a target',
      'continue':'Continue', auto:'Auto', 'win.good':'Good wins', 'win.evil':'Evil wins', won:'You won', lost:'You lost',
      'reason.ASSASSIN_MISSED':'Assassin missed', 'reason.ASSASSIN_HIT':'Merlin assassinated', 'reason.THREE_FAILS':'Three quests failed', 'reason.FIVE_REJECTIONS':'Five rejections',
      again:'Play again', replay:'Replay file',
      'ev.start':'Roles dealt', 'ev.proposal':'Seat {l} proposes {team}', 'ev.vote':'{y} for · {n} against · {r}', passed:'passed', rejected:'rejected',
      'ev.ok':'Quest {q} succeeded', 'ev.bad':'Quest {q} failed · {f} fail', 'ev.assn':'Assassin shot seat {n}',
      'me.evil':'Evil', 'me.cands':'Merlin / Morgana', 'me.none':'Public info only', 'me.cards':'Your cards',
      log:'Record', empty:'Nothing yet', 'col.q':'#', 'col.lead':'Lead', 'col.team':'Team', 'col.votes':'Votes', 'col.res':'Result',
      'setup.title':'New game', 'setup.role':'Role', 'setup.seat':'Seat', 'setup.ai':'Opponent model', 'setup.share':'Share this game',
      start:'Start', cancel:'Cancel', loading:'Loading…', ok:'Got it',
      'rules.1':'Good: Merlin, Percival, two Servants. Evil: Morgana, Assassin.',
      'rules.2':'The leader picks a team and everyone votes; 4 approvals pass. Five rejections in a row: evil wins.',
      'rules.3':'Team members play cards in secret; one Fail fails the quest. Team sizes 2 · 3 · 4 · 3 · 4.',
      'rules.4':'Three fails: evil wins. After three successes the Assassin names Merlin; a hit wins for evil.',
      'rules.5':'Merlin sees evil; evil know each other; Percival sees Merlin and Morgana but not which is which.',
      'a.title':'Assassin Helper', 'a.first':'First leader?', 'a.firstSub':'Number seats in leader order', 'a.lead':'Seat {l} picks {n}',
      'a.votes':'Tap seats to record votes', 'a.votesSub':'Tap once: approve · twice: reject', 'a.restYes':'Rest approve', 'a.restNo':'Rest reject', save:'Save',
      'a.result':'Quest {q} result', 'a.nfail':'{f} fail', 'a.assassin':'Assassin?', 'a.morgana':'Morgana?', 'a.ask':'Ask the AI', 'a.thinking':'Thinking…',
      'a.pick':'AI shoots seat {n}', 'a.models':'Three models', 'a.merlin':'Who was Merlin?', 'a.human':'Human assassin shot?', hit:'Hit', miss:'Miss',
      'a.evilWon':'Evil already won', 'a.variant':'Quest {q} fail by {who}: {p}', 'a.uncertain':'Fail-card owner unknown; averaged over the cases',
      note:'Note', 'a.mine':'My games', 'a.all':'All shared', 'a.stat':'AI hits {h} / {n}', 'a.statHuman':'Human {h} / {n}', chance:'chance 25%',
      open:'Open', del:'Delete', 'a.delQ':'Delete this game?', 'a.newQ':'This game is not finished. Start a new one?', 'a.codeTip':'Enter this code on another device to see the same games',
      'a.codePh':'Enter a code', 'a.switch':'Switch', 'a.games':'{n} games',
      'o.playing':'In progress', 'o.evil5':'Five rejections', 'o.evil3':'Three fails', 'o.assassinate':'Assassination',
    },
  };
  // Server messages are Chinese; translate the ones a visitor can actually meet.
  const serverEn = [
    [/^请先开始一局新游戏/, 'Start a new game first.'], [/^牌桌已更新/, 'The table changed. Refresh and try again.'],
    [/^当前行动不合法/, 'That move is not allowed.'], [/^请先选择你的行动/, 'Make your move first.'],
    [/^当前请点击继续/, 'Press Continue for the AI.'], [/^对局会话已过期/, 'Session expired. Refresh to start again.'],
    [/^当前牌桌人数已满/, 'All tables are busy. Try again later.'], [/^请选择是否共享/, 'Choose whether to share this game.'],
    [/^第 (\d+) 轮任务有 (\d+) 张失败牌/, 'Quest $1 had $2 fail card(s), more than the evil players on it. Check the Assassin and Morgana seats.'],
    [/^记录还没有走到刺杀阶段/, 'Not at the assassination yet.'], [/^刺客和莫甘娜不能是同一个座位/, 'The Assassin and Morgana must be different seats.'],
    [/^记录码应为/, 'A code has 16 characters (0–9, a–f).'], [/^找不到这局记录/, 'Game not found.'],
  ];
  let lang = 'zh';
  try { if (localStorage.getItem('avalon_lang') === 'en') lang = 'en'; } catch (e) {}
  const listeners = [];
  function t(key, p = {}) {
    const s = dict[lang][key] ?? dict.zh[key] ?? key;
    return s.replace(/\{(\w+)\}/g, (m, k) => (k in p ? p[k] : m));
  }
  function apply(root = document) {
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    root.querySelectorAll('[data-t]').forEach(el => { el.textContent = t(el.dataset.t); });
    root.querySelectorAll('[data-t-ph]').forEach(el => { el.placeholder = t(el.dataset.tPh); });
    root.querySelectorAll('[data-t-label]').forEach(el => { el.setAttribute('aria-label', t(el.dataset.tLabel)); el.title = t(el.dataset.tLabel); });
    if (document.body.dataset.title) document.title = t(document.body.dataset.title);
    document.querySelectorAll('[data-lang-toggle]').forEach(b => { b.textContent = lang === 'zh' ? 'EN' : '中'; });
  }
  function set(next) {
    lang = next;
    try { localStorage.setItem('avalon_lang', lang); } catch (e) {}
    apply();
    listeners.forEach(f => f());
  }
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-lang-toggle]').forEach(b => b.addEventListener('click', () => set(lang === 'zh' ? 'en' : 'zh')));
    apply();
  });
  return {
    t, apply, set, onChange: f => listeners.push(f),
    get lang() { return lang; },
    role: key => t('role.' + key),
    seats: list => (list.length ? list.join(lang === 'zh' ? '、' : ', ') : '—'),
    err: msg => { if (lang === 'zh') return msg; for (const [re, en] of serverEn) if (re.test(msg)) return msg.replace(re, en).replace(/[。，].*$/, ''); return msg; },
  };
})();
