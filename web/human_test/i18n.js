'use strict';
// Shared language module: Chinese by default, English on request; the choice is kept in localStorage.
const I18N = (() => {
  const dict = {
    zh: {
      home:'首页', rules:'规则', newGame:'新对局', code:'记录码', history:'历史', undo:'撤销',
      'share.q':'是否同意将数据用于研究？', 'share.yes':'同意', 'share.no':'不同意', 'share.note':'匿名记录，不含姓名或 IP', shared:'已共享',
      'role.SERVANT':'忠臣', 'role.MERLIN':'梅林', 'role.PERCIVAL':'派西维尔', 'role.MORGANA':'莫甘娜', 'role.ASSASSIN':'刺客',
      'side.good':'好人', 'side.evil':'坏人', seat:'{n} 号', you:'你', ai:'AI', random:'随机',
      'home.title':'Avalon Signals · 真人测试', 'home.sub':'六人阿瓦隆 · AI 为冻结的强化学习策略',
      'mode.match':'人机对战', 'mode.match.d':'一名真人与五个 AI 对局，可选发言或静默', 'mode.analysis':'刺杀辅助', 'mode.silent.name':'静默版', 'mode.silent.d':'一名真人与五个 AI 对局，全程不发言', 'mode.talk.name':'发言版', 'mode.talk.d':'每轮组队前轮流发言，可打字或语音；AI 说到做到', said:'{n} 说', 'bubble.more':'点气泡看全文', 'mode.analysis.d':'录入线下对局，由 AI 刺客推断梅林',
      go:'进入 →', 'home.hint':'请先选择是否同意',
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
      'setup.title':'新的一局', 'setup.role':'身份', 'setup.seat':'座位', 'setup.ai':'AI 模型', 'setup.share':'共享本局数据',
      start:'开始', cancel:'取消', loading:'加载中…', ok:'知道了',
      'rules.1':'好人：梅林、派西维尔、2 名忠臣；坏人：莫甘娜、刺客。',
      'rules.2':'队长组队，全员投票，4 票赞成通过；连续 5 次否决坏人胜。',
      'rules.3':'队员秘密出牌，1 张失败即任务失败。五轮人数 2 · 3 · 4 · 3 · 4。',
      'rules.4':'3 次失败坏人胜；3 次成功后刺客猜梅林，猜中则坏人胜。',
      'rules.5':'梅林知道坏人；坏人互相认识；派西维尔看到梅林和莫甘娜，但分不清。',
      'rules.6':'发言桌：每轮组队前每人发言一次，队长最后说，可以回复先发言的人。AI 说到做到：指控谁就反对含他的队，推荐或同意哪支队就赞成那支队，承诺的条件成立就照说。',
      'rules.7':'你不受这些约束，但 AI 只听得懂明确的表态；说了不做会被公开记一次“言行不符”（座位上的 ! 标记），AI 会因此更怀疑你。',
      'rules.8':'你可以打字或按话筒说话。语言模型只做翻译：把你的话转成 AI 听得懂的表态，把 AI 的表态说成口语，不参与决策。文字和录音会发送给 OpenAI 处理。同意共享数据时，你说的话会以文字随对局一起记录，用来检查翻译是否准确；录音不保存。',
      spoken:'口语', 'mic.start':'语音输入', 'mic.stop':'停止录音', 'mic.on':'正在录音，再点一次话筒结束（最长 30 秒）', 'mic.denied':'无法使用麦克风，请在浏览器里允许，或改用打字',
      'say.hearing':'识别中…', 'say.note':'直接说或打字即可，AI 只听明确的表态；内容会发送给 OpenAI 做翻译', 'say.manual':'手动设置表态', 'why.other':'没有明确表态，未传达给 AI', 'why.group':'AI 只听得懂对每个人单独的表态，“其中一个”这类说法没有传达',
      'setup.mode':'牌桌', 'mode.talk':'发言', 'mode.silent':'静默', talk:'发言', evilTalk:'刺杀前商议', yourTalk:'轮到你发言', talkingNow:'{n} 号发言中',
      broken:'言行不符 {n} 次', 's.claim.servant':'我是忠臣', 's.claim.percival':'我是派西维尔', 's.claim.merlin':'我是梅林', 's.judge':'{n} 号是{j}', 's.basis':'（依据第 {q} 轮{k}记录）', 's.sight':'（依据身份信息）',
      'basis.quest':'任务', 'basis.vote':'投票', 'basis.proposal':'组队', 's.team':'推荐 {team} 号', 's.veto':'有 {n} 号的队我不赞成', 's.require':'没有 {n} 号的队我不赞成', 's.none':'（不发言）',
      's.endorse':'同意 {n} 号，他推荐的队我赞成', 's.dispute':'不同意 {n} 号{b}', 's.disputeTeam':'不同意 {n} 号{b}，他推荐的队我不赞成',
      'rb.defense':'（他指认了我）', 'rb.judgment':'（我们对同一人判断相反）', 'rb.team':'（他推荐的队里有我指认的人）',
      's.promise.fail_evil':'{n} 号若上本轮任务且失败，下轮我说他是坏人', 's.promise.success_good':'{n} 号若上本轮任务且成功，下轮我说他是好人',
      's.promise.fail_evil.n':'本轮任务若失败，{n} 号谁在队里，下轮我说谁是坏人', 's.promise.success_good.n':'本轮任务若成功，{n} 号谁在队里，下轮我说谁是好人',
      'say.none':'无', 'say.reply':'回复', 'say.endorse':'同意', 'say.dispute':'不同意', 'say.promise':'承诺', 'say.fail_evil':'失败则坏', 'say.success_good':'成功则好',
      'say.promised':'你上轮的承诺已触发：',
      'why.not_spoken':'只能回复本轮已发言的玩家', 'why.no_team':'他没有推荐队伍，无从同意', 'why.promise_both':'对一名玩家只能承诺一种情况', 'why.promise_closed':'这一轮不能做这个承诺',
      'say.ph':'例如：3号是坏人，推荐1、2、5号', 'say.phEvil':'例如：5号是梅林', 'say.parse':'识别', 'say.heard':'AI 听到的是：', 'say.send':'发言', 'say.silent':'不发言',
      'say.noClaim':'不报身份', 'say.vote':'投票条件', 'say.anyTeam':'无', 'say.veto':'有他不投', 'say.require':'没他不投', 'say.team':'点座位可推荐一支队（已选 {k} / {n}）',
      'why.soft':'语气不确定，没有传达给 AI', 'why.unknown':'没听懂，没有传达给 AI', 'why.size':'队伍人数不对', 'why.merlin_claim':'暂不支持自称梅林', 'why.role_name':'具体身份只在刺杀前商议时说', 'why.leader':'队长不用推荐，直接组队',
      'a.title':'刺杀辅助', 'a.first':'首任队长？', 'a.firstSub':'座位按队长轮转顺序编号', 'a.lead':'{l} 号组队 · 选 {n} 人',
      'a.votes':'点座位记录投票', 'a.votesSub':'点一下赞成，再点反对', 'a.restYes':'其余赞成', 'a.restNo':'其余反对', save:'记录',
      'a.result':'第 {q} 轮结果', 'a.nfail':'{f} 失败', 'a.assassin':'刺客是？', 'a.morgana':'莫甘娜是？', 'a.ask':'运行分析', 'a.thinking':'分析中…',
      'a.pick':'AI 推断梅林：{n} 号', 'a.models':'各模型', 'a.merlin':'实际梅林', 'a.human':'人类刺客选择', hit:'正确', miss:'错误',
      'a.evilWon':'坏人已获胜', 'a.variant':'第 {q} 轮失败牌来自{who}，概率 {p}', 'a.uncertain':'失败牌归属不确定，已对各种情况取平均',
      note:'备注', 'a.mine':'我的记录', 'a.all':'全部共享记录', 'a.stat':'AI 准确率 {h} / {n}', 'a.statHuman':'人类 {h} / {n}', chance:'随机基线 25%',
      open:'打开', del:'删除', 'a.delQ':'删除这局记录？', 'a.newQ':'当前局未结束，开始新局？', 'a.codeTip':'在另一台设备输入此码，可查看同一批记录',
      'a.codePh':'输入记录码', 'a.switch':'切换', 'a.games':'{n} 局',
      'o.playing':'进行中', 'o.evil5':'五次否决', 'o.evil3':'三次失败', 'o.assassinate':'刺杀',
    },
    en: {
      home:'Home', rules:'Rules', newGame:'New game', code:'Code', history:'History', undo:'Undo',
      'share.q':'Allow your data to be used for research?', 'share.yes':'Agree', 'share.no':'Decline', 'share.note':'Anonymous; no names or IP addresses', shared:'Shared',
      'role.SERVANT':'Servant', 'role.MERLIN':'Merlin', 'role.PERCIVAL':'Percival', 'role.MORGANA':'Morgana', 'role.ASSASSIN':'Assassin',
      'side.good':'Good', 'side.evil':'Evil', seat:'Seat {n}', you:'You', ai:'AI', random:'Random',
      'home.title':'Avalon Signals · Human Test', 'home.sub':'Six-player Avalon · AI players are frozen RL policies',
      'mode.match':'Human vs. AI', 'mode.match.d':'One human and five AI players, talking or silent', 'mode.analysis':'Assassination Aid', 'mode.silent.name':'Silent table', 'mode.silent.d':'One human and five AI players, no talk at all', 'mode.talk.name':'Talking table', 'mode.talk.d':'Everyone speaks before each team is picked; type or use your voice. The AI keeps its word', said:'{n} says', 'bubble.more':'Tap a bubble for the full text', 'mode.analysis.d':'Enter an offline game; the AI Assassin infers Merlin',
      go:'Enter →', 'home.hint':'Please choose Agree or Decline first',
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
      'setup.title':'New game', 'setup.role':'Role', 'setup.seat':'Seat', 'setup.ai':'AI model', 'setup.share':'Share this game',
      start:'Start', cancel:'Cancel', loading:'Loading…', ok:'Got it',
      'rules.1':'Good: Merlin, Percival, two Servants. Evil: Morgana, Assassin.',
      'rules.2':'The leader picks a team and everyone votes; 4 approvals pass. Five rejections in a row: evil wins.',
      'rules.3':'Team members play cards in secret; one Fail fails the quest. Team sizes 2 · 3 · 4 · 3 · 4.',
      'rules.4':'Three fails: evil wins. After three successes the Assassin names Merlin; a hit wins for evil.',
      'rules.5':'Merlin sees evil; evil know each other; Percival sees Merlin and Morgana but not which is which.',
      'rules.6':'Talk table: before each team is picked everyone speaks once, the leader last, and may answer earlier speakers. The AI keeps its word: it rejects teams with a player it accused, approves the team it suggested or agreed with, and says what it promised once the condition holds.',
      'rules.7':'You are not bound, but the AI only hears clear statements. Acting against your own words is counted in public (the ! mark on your seat) and makes the AI suspect you.',
      'rules.8':'You can type or tap the microphone and speak. A language model only translates: your words into statements the AI understands, and the AI\'s statements into spoken sentences. It makes no decisions. Text and recordings are sent to OpenAI for this. If you agreed to share data, your words are kept as text with the game, to check the translation; recordings are not kept.',
      spoken:'Spoken', 'mic.start':'Voice input', 'mic.stop':'Stop recording', 'mic.on':'Recording — tap the microphone again to finish (30 s at most)', 'mic.denied':'No access to the microphone. Allow it in the browser, or type instead',
      'say.hearing':'Reading…', 'say.note':'Just speak or type; the AI hears clear statements only. Your words are sent to OpenAI for translation', 'say.manual':'Set the statements by hand', 'why.other':'no clear statement, not passed on', 'why.group':'the AI only hears a statement about each player; “one of them” was not passed on',
      'setup.mode':'Table', 'mode.talk':'Talking', 'mode.silent':'Silent', talk:'Talk', evilTalk:'Before the assassination', yourTalk:'Your turn to speak', talkingNow:'Seat {n} is speaking',
      broken:'Acted against own words {n} time(s)', 's.claim.servant':'I am a Servant', 's.claim.percival':'I am Percival', 's.claim.merlin':'I am Merlin', 's.judge':'Seat {n} is {j}', 's.basis':' (from the {k} record of quest {q})', 's.sight':' (from my role)',
      'basis.quest':'quest', 'basis.vote':'vote', 'basis.proposal':'proposal', 's.team':'I suggest {team}', 's.veto':'I reject any team with seat {n}', 's.require':'I reject any team without seat {n}', 's.none':'(says nothing)',
      's.endorse':'I agree with seat {n} and approve their team', 's.dispute':'I disagree with seat {n}{b}', 's.disputeTeam':'I disagree with seat {n}{b} and reject their team',
      'rb.defense':' (they accused me)', 'rb.judgment':' (we judge a player in opposite ways)', 'rb.team':' (their team has a player I accuse)',
      's.promise.fail_evil':'If seat {n} goes on this quest and it fails, next round I call them evil', 's.promise.success_good':'If seat {n} goes on this quest and it succeeds, next round I call them good',
      's.promise.fail_evil.n':'If this quest fails, next round I call evil whoever of seats {n} was on it', 's.promise.success_good.n':'If this quest succeeds, next round I call good whoever of seats {n} was on it',
      'say.none':'none', 'say.reply':'Reply', 'say.endorse':'agree', 'say.dispute':'disagree', 'say.promise':'Promise', 'say.fail_evil':'fail → evil', 'say.success_good':'success → good',
      'say.promised':'Your promise from last round is due: ',
      'why.not_spoken':'you can only answer players who already spoke this round', 'why.no_team':'they suggested no team to agree with', 'why.promise_both':'one promise per player', 'why.promise_closed':'that promise is not open in this round',
      'say.ph':'e.g. 3 is evil, suggest 1, 2 and 5', 'say.phEvil':'e.g. 5 is Merlin', 'say.parse':'Read', 'say.heard':'The AI hears: ', 'say.send':'Say it', 'say.silent':'Say nothing',
      'say.noClaim':'No claim', 'say.vote':'Vote', 'say.anyTeam':'free', 'say.veto':'not with them', 'say.require':'only with them', 'say.team':'Tap seats to suggest a team ({k} / {n} picked)',
      'why.soft':'hedged, not passed on to the AI', 'why.unknown':'not understood, not passed on', 'why.size':'wrong team size', 'why.merlin_claim':'claiming Merlin is not supported yet', 'why.role_name':'role names are only for the talk before the assassination', 'why.leader':'the leader just picks the team',
      'a.title':'Assassination Aid', 'a.first':'First leader?', 'a.firstSub':'Number seats in leader order', 'a.lead':'Seat {l} picks {n}',
      'a.votes':'Tap seats to record votes', 'a.votesSub':'Tap once: approve · twice: reject', 'a.restYes':'Rest approve', 'a.restNo':'Rest reject', save:'Save',
      'a.result':'Quest {q} result', 'a.nfail':'{f} fail', 'a.assassin':'Assassin?', 'a.morgana':'Morgana?', 'a.ask':'Run analysis', 'a.thinking':'Analyzing…',
      'a.pick':'AI estimate of Merlin: seat {n}', 'a.models':'Per model', 'a.merlin':'Actual Merlin', 'a.human':'Human Assassin\'s choice', hit:'Correct', miss:'Incorrect',
      'a.evilWon':'Evil already won', 'a.variant':'Quest {q} fail by {who}: {p}', 'a.uncertain':'Fail-card owner unknown; averaged over the cases',
      note:'Note', 'a.mine':'My records', 'a.all':'All shared records', 'a.stat':'AI accuracy {h} / {n}', 'a.statHuman':'Human {h} / {n}', chance:'chance 25%',
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
    [/^当前牌桌人数已满/, 'All tables are busy. Try again later.'], [/^请先发言/, 'Speak first, or choose to say nothing.'],
    [/^现在不是你的发言时间/, 'It is not your turn to speak.'], [/^推荐的队伍需要 (\d+) 人/, 'A suggested team needs $1 players.'],
    [/^队长不推荐队伍/, 'The leader does not suggest a team.'], [/^只能回复本轮已经发过言的玩家/, 'You can only answer players who already spoke this round.'],
    [/^语音识别暂时不可用/, 'Voice input is not available right now. Please type.'], [/^没有听清/, 'Nothing was heard. Please say it again.'],
    [/^录音太长或为空/, 'The recording is empty or too long.'], [/^录音格式或长度不支持/, 'This recording cannot be read.'],
    [/^本局的识别次数已用完/, 'No translations left in this game. Use the buttons below.'],
    [/^(\d+) 号没有推荐队伍/, 'Seat $1 suggested no team to agree with.'], [/^这一轮不能做这个承诺/, 'That promise is not open in this round.'], [/^模型文件无法读取/, 'The model files could not be read.'], [/^请选择是否共享/, 'Choose whether to share this game.'],
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
