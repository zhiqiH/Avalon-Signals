# References 索引（核心 10 篇）

项目的参考文献 PDF。方括号编号对应 `Avalon_Research_Design.docx`（研究设计）中的引用编号；“研究 §n”“实现 §n”分别指《研究设计》与《实现方案》的章节。文件名带 `0` 前缀的是最初的三篇。

类型：**背景** = 问题设定、领域现状与对比对象；**方法** = 项目会直接采用或据此做设计决策的算法、协议与测量方法。

## 一览

| 编号 | 文件 | 出处 | 类型 | 用在哪里 |
|---|---|---|---|---|
| [1] | 0Finding Friend and Foe in Multi-Agent Games.pdf | Serrino et al., NeurIPS 2019 | 背景＋方法参照 | 研究 §2 文献定位；规则基线与 RQ3 probe |
| [2] | 0Is it Percival time yet.pdf | Nakamura, SIGBOVIK 2017 | 背景 | RQ2 指标；身份声明槽的人类参照 |
| [3] | 0STRATEGIST.pdf | Light et al., ICLR 2025 | 背景 | 研究 §2 创新性定位；实现 §10 语言层 |
| [5] | LLM-Based Agent Society Investigation - Collaboration and Confrontation in Avalon Gameplay.pdf | Lan et al., EMNLP 2024 | 背景 | RQ2 行为分类；RQ4 真实性分析 |
| [6] | RLupus - Cooperation through Emergent Communication in the Werewolf Social Deduction Game.pdf | Brandizzi et al., Intelligenza Artificiale 2021 | 背景＋方法 | 最接近的先例；实现 §6 玩具检查；槽位开关设计 |
| [10] | Proximal Policy Optimization Algorithms.pdf | Schulman et al., arXiv 2017 | 方法 | 实现 §4 核心更新算法 |
| [11] | The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.pdf | Yu et al., NeurIPS 2022 | 方法 | 实现 §3–4 多智能体 PPO 实现 |
| [7] | Learning to Communicate with Deep Multi-Agent Reinforcement Learning.pdf | Foerster et al., NeurIPS 2016 | 方法 | 实现 §3 消息头的 RIAL 式训练 |
| [8] | Emergent Communication through Negotiation.pdf | Cao et al., ICLR 2018 | 方法（理论依据） | 研究 §4 依据锚定的设计理由；通信风险 |
| [9] | On the Pitfalls of Measuring Emergent Communication.pdf | Lowe et al., AAMAS 2019 | 方法（测量） | 实现 §6 通信诊断；RQ3、RQ4 |

## 逐篇要点

### [1] DeepRole：Finding Friend and Foe in Multi-Agent Games
- **重点**：Algorithm: DeepRole 一节中的 *CFR with deductive logic* 与 *Value network*；附录 *State space calculation*。
- **用途**：五人阿瓦隆、不用语言即可达到很强水平，是 "Silent Avalon 可行" 的先例。它用演绎推理排除与公开记录矛盾的角色分配，这个思路可以直接做规则基线和 RQ3 的身份 probe。本项目的六人配置只有 6!/2! = 360 种角色分配，精确的贝叶斯身份信念完全算得动。
- **差异**：CFR 搜索加价值网络，不是 PPO；没有通信对照。

### [2] Is it Percival time yet?
- **重点**：4.4 *Percival claims*、4.5 *The first mission fail*、4.6 *Evil coordination failures*。
- **用途**：人类对局的描述统计，可作为 RQ2 指标的来源；派西维尔何时公开身份，直接对应身份声明槽 ①。
- **局限**：61 局、单一玩家群体，不能作因果推断。

### [3] STRATEGIST
- **重点**：附录 *Resistance: Avalon Game Description*（规则与讨论阶段）、*Avalon Agent Implementation Details*（Dialogue Analyzer / Action Planner / Dialogue Generation）、*Details on Learning the Value Heuristic via RL*、附录 O *Human Evaluation Details*。
- **用途**：附录 O 有六人局有对话、无对话的人类实验，但角色不含派西维尔与莫甘娜，也不是受控训练，这是本项目创新性定位的依据。它把发言的"分析、规划、生成"拆开，可作为《实现方案》§10 语言层的参照。

### [5] Lan et al.：LLM-Based Agent Society Investigation
- **重点**：3.1 *Social Behaviors in Avalon*；第 6 节 *Social Behaviors of AI Agents*（Leadership、Persuasion、Camouflage、Teamwork and Confrontation、Sharing、Vacillation）；第 4 节智能体框架。
- **用途**：与本项目相同、含派西维尔和莫甘娜的六人局。第 6 节的行为分类可借给 RQ2；Camouflage 与 Sharing 对应 RQ4 的真实性分析（冒充、隐瞒、如实分享）。
- **差异**：LLM 智能体，规则细节与规模不同，胜率不可直接比较。

### [6] RLupus
- **重点**：第 3 节 *Problem Statement*、第 4 节 *General solution framework*、5.3 环境与信号向量、5.4 *Policies*、第 6 节结果与 Table 1、附录 A.3（与 AIWolf 协议的比较）。
- **用途**：最接近的 MARL 先例。
  - **算法**：用 PPO 训练村民，狼人是规则策略。这正是《实现方案》§6 "玩具检查"的设计：坏人固定为规则策略，只训练好人通信。
  - **通信效果**：九人局中，村民胜率从无通信的 0.044 最高提到 0.45（信号长度 9、取值范围 2）；但长度 9、取值范围 9 时只有 0.19，可见带宽和效果并非单调关系。这对应本项目的槽位开关与取值范围消融：多开槽位不一定更好。
  - **结构化协议**：附录 A.3 对比的 AIWolf 协议把发言拆成知识/意图、过往行动、赞同/反对等类型，并带有 "because" 等推理算子，是五槽结构（含判断依据）的直接先例。

### [10] PPO：Proximal Policy Optimization Algorithms
- **重点**：第 3 节 *Clipped Surrogate Objective*；第 5 节 *Algorithm*（含截断版 GAE）。
- **用途**：核心更新规则。消息的联合对数概率进入同一个 clipped 比率。GAE 对应课程第 12 章 λ-return，写论文时可借此和课程内容挂钩。

### [11] MAPPO / IPPO：The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
- **重点**：3.2 *MAPPO and IPPO*；第 5 节影响性能的因素（Value Normalization、Input Representation to Value Function、Training Data Usage、PPO Clipping、Batch Size）；附录 C.2 *Parameter Sharing*、C.4 *Hyperparameters*；4.5 *Hanabi Testbed*。
- **用途**：《实现方案》§3–4 的实现手册。同阵营共享参数、critic 输入选择（个体信息还是全局信息）、超参数起点都可以从这里取。Hanabi 是部分可观察的合作博弈，和本项目的信息结构最接近。
- **注意**：论文针对纯合作任务；阿瓦隆是两队对抗，效果需要实验确认。

### [7] RIAL / DIAL：Learning to Communicate with Deep Multi-Agent Reinforcement Learning
- **重点**：第 4 节 *Setting*、5.1 *RIAL*、5.2 *DIAL*、6.1 *Model Architecture*、6.4 *Effect of Channel Noise*。
- **用途**：把消息当作策略的离散动作，用 RL 和其他动作一起训练，智能体之间共享参数，这就是 RIAL。本项目沿用这一思路，把 DQN 换成 PPO，并把一条消息拆成五个槽位。DIAL 需要让梯度穿过信道，适用于全员合作，而阿瓦隆的听者里有对手，所以不采用；写论文时可以借此解释方法选择。

### [8] Emergent Communication through Negotiation
- **重点**：2.2 *Communication channels*、2.3 *Agent sociality and reward schemes*、第 3–4 节实验、第 5 节对语言信道的分析。
- **用途**：自利智能体能利用有内在含义的信道（proposal 信道），却学不会利用无内在含义的 cheap-talk 信道；合作型智能体两者都能用。由此得到两条设计理由：
  - 组队建议 ④ 与提案共用队伍编码，判断依据 ③ 必须指向真实事件，都是为了让信道带有内在含义；
  - 需要设置"通信学习诊断"，不能默认通信会被学会。

### [9] On the Pitfalls of Measuring Emergent Communication
- **重点**：3.1 *A categorization of metrics*（positive signaling / positive listening）、3.3 *Causal influence of communication*、6.2 *Recommendations*。
- **用途**：通信学习诊断与 RQ4 的"听者影响"指标直接取自这里。核心提醒是：消息与身份相关，不代表听者用了这条消息。

## 建议阅读顺序

| 阶段 | 阅读 | 对应工作 |
|---|---|---|
| 写环境 | STRATEGIST 规则附录、DeepRole 第 2–3 节、Percival time | 写环境与规则测试 |
| 实现训练 | PPO 第 3、5 节；MAPPO 第 3、5 节与附录 C；RIAL 第 5 节 | 实现 PPO、参数共享与消息头 |
| 通信诊断 | RLupus 全文；Cao 第 2–4 节；Lowe 第 3 节 | 玩具检查与通信诊断 |
| 分析与写作 | Lan 第 6 节；各篇 Related Work | RQ2 分析与论文综述部分 |

## 已移到废纸篓的 16 篇（2026-10-20）

按 "只保留核心 10 篇" 的决定移除。在清空废纸篓前，可以在 Finder 废纸篓中右键选 "放回原处" 恢复。《研究设计》仍引用其中的 AvalonBench [4]，《实现方案》仍引用 AvalonBench、Jaques et al. 与 Policy Distillation；引用本身不需要本地 PDF。

A Unified Game-Theoretic Approach to MARL（PSRO）· AvalonBench · Bayesian Action Decoder · Classifying Ambiguous Identities in Hidden-Role Stochastic Games（IDRL）· Deep Recurrent Q-Learning for POMDPs · Hidden Agenda · Generalized Advantage Estimation · Is Independent Learning All You Need in SMAC（IPPO）· Language Agents with RL for Werewolf · Learning to Deceive in Hidden Role Games · Other-Play · Playing Social Deduction Games with RFT LLMs · Policy Distillation · Social Influence as Intrinsic Motivation · The Hanabi Challenge · Unbiased Asymmetric RL under Partial Observability

## 未下载，需要手动获取

| 引用 | 原因 | 获取方式 |
|---|---|---|
| 研究 [13] The Resistance: Avalon Rulebook | 文档中的链接是第三方镜像站 | 从出版商 Indie Boards & Cards 官网或实体游戏获取 |
| 实现 [7] Avalon-LLM 代码库 | 是代码仓库，不是论文 | https://github.com/jonathanmli/Avalon-LLM |
| 研究 [12] CICERO, Science 2022 | Science 正式版需要订阅 | 通过 SMU 图书馆访问 https://www.science.org/doi/10.1126/science.ade9097 |
