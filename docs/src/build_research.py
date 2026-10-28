"""Generates Avalon_Research_Design.docx: research questions, setting and experimental design."""
from docgen import DocBuilder

d = DocBuilder(
    title="六人阿瓦隆中显式交流的作用",
    subtitle="Research Design: Structured Communication in Avalon",
    meta_line="CS7329 Reinforcement Learning ｜ 研究设计 ｜ 2026年10月28日 ｜ 配套：《实现方案》",
    subject="Research Design: Structured Communication in Avalon",
    keywords="Avalon, MARL, Reinforcement Learning, Communication, CS7329",
)

d.h("1 研究问题")
d.p("《阿瓦隆》是隐藏身份游戏：好人人数占优但缺少信息，坏人彼此知情但人数少，线下对局中大量信息靠讨论传递。本项目要回答：在只靠公开行为（组队、投票、任务结果）进行的沉默对局之上，加入结构化的显式交流，会怎样改变阵营胜率、策略行为和信息传递？其中哪类信息真正起了作用？")
d.p("为什么用强化学习：每名玩家只看到部分信息，需要从对局历史推断身份并做序贯决策；奖励只在终局出现；最优的组队、投票和发言方式事先未知。这类问题适合用 self-play 的策略梯度方法学习策略，再在受控条件下比较不同的通信设置。", boldlead="为什么用强化学习：")
d.table(
    ["问题", "需要回答的内容", "主要证据"],
    [
        ["RQ1\nGame Balance", "相对沉默基线，加入交流后哪一阵营获益更多？", "好人最终胜率、三次任务成功概率、条件刺杀成功率"],
        ["RQ2\nStrategic Behavior", "相对沉默基线，组队、投票、破坏与身份隐藏如何调整？", "按角色和局面分层的行为指标"],
        ["RQ3\nImplicit Signaling", "沉默基线中，公开行为是否承担信息传递？加入显式交流后，这部分隐式信号被补充还是被挤出？", "信息增益、接收者干预、跨种子验证"],
        ["RQ4\nMessage Components", "从沉默基线逐项加入的五类信息中，哪些贡献了交流的效果？", "逐项加入的训练消融、测试时关闭槽位、槽位真实性与听者影响"],
    ], [1.12, 2.75, 2.84]
)
d.p("以沉默为基线：沉默对局的决策更少，也不存在“消息被忽略”的 cheap-talk 学习难题，更容易训练成稳定的基线，并可作为各通信配置的起点。研究因此表述为“交流带来了什么”。不预设方向：交流可能帮助好人协调，也可能帮助坏人欺骗或找到梅林；不同信息的作用方向也可能相反，例如身份声明可能帮助派西维尔，也可能把梅林暴露给刺客。", boldlead="以沉默为基线：")

d.h("2 文献定位")
d.table(
    ["工作", "做法", "与本项目的关系"],
    [
        ["DeepRole [1]", "五人阿瓦隆；CFR、演绎推理与价值网络；不使用语言", "智能体可以完成沉默对局的先例，支持以沉默为基线"],
        ["Is it Percival time yet [2]", "61 局人类对局的描述统计", "指标来源：派西维尔声明、首次任务失败、坏人协调失败；不能做因果推断"],
        ["STRATEGIST [3]\nAvalonBench [4]", "LLM 结合搜索；LLM 阿瓦隆基准", "前者附录 O 有六人有对话与无对话的人类实验，但不含派西维尔与莫甘娜，也不是受控训练；后者显示通用 LLM 的阿瓦隆水平有限"],
        ["Lan et al. [5]", "含派西维尔与莫甘娜的六人局，LLM 智能体", "伪装、分享、说服等行为分类可借给 RQ2 与 RQ4"],
        ["RLupus [6]", "狼人杀；用 PPO 训练村民的离散通信信道", "最接近的先例：村民胜率从无通信的 0.044 最高提到 0.45，且信道带宽与效果并非单调关系"],
        ["RIAL [7]\nCao et al. [8]", "把消息当作 RL 动作训练；自利智能体难以利用 cheap talk", "通信的训练方式与主要风险"],
        ["Lowe et al. [9]", "涌现通信的测量方法", "区分“消息包含信息”与“听者利用了消息”"],
        ["PPO [10]\nMAPPO [11]", "clipped 策略梯度；多智能体 PPO 的实现经验", "训练算法"],
    ], [1.45, 2.40, 2.86]
)
d.p("拟贡献：", boldlead="拟贡献：")
d.bullets([
    "以沉默对局为基线，在规则、算法和预算上受控、逐级加入通信的实验；",
    "用可独立开关的五槽协议，把“交流”分解为可逐项加入的信息类型；",
    "把胜负分解为任务与刺杀两条路径，并用干预区分“行为包含信息”与“行为被他人利用”。",
])
d.p("课程关联：问题是部分可观察随机博弈，是课程部分可观察讲义的多智能体推广；方法涉及策略梯度与 actor–critic（第 13 章）、PPO（深度强化学习讲义）、GAE 与 λ-return（第 12 章），以及把完整的观察历史编码为信息状态（部分可观察讲义中的 information state）。[C1, C2]", boldlead="课程关联：")

d.h("3 游戏设定")
d.p("固定六人配置：两名平民（Loyal Servants）、梅林（Merlin）、派西维尔（Percival）、莫甘娜（Morgana）、刺客（Assassin）。好人阵营为梅林、派西维尔与两名平民，坏人阵营为莫甘娜与刺客。规则以规则书为准。[13]")
d.table(
    ["项目", "规范"],
    [
        ["任务与表决", "五轮任务人数为 2、3、4、3、4；至少四票赞成才通过，平票否决；同一任务连续五次提案被否决，坏人获胜。"],
        ["任务执行", "好人只能出 Success；坏人可选 Success 或 Fail；每轮一张 Fail 即失败；只公开失败票数。"],
        ["终局判定", "三次任务失败，坏人胜；三次成功后刺客刺杀，命中梅林则坏人胜，否则好人胜。"],
        ["身份信息", "梅林知道两名坏人；派西维尔看到梅林与莫甘娜两名候选者，但分不清谁是谁；两名坏人互相知情；平民没有额外信息。"],
        ["随机化", "每局随机分配角色与初始队长；不使用湖中仙女等附加机制。"],
    ], [1.10, 5.61]
)
d.p("所有玩家只看到公共历史和本角色允许的私有信息；角色真值只用于裁判与离线分析。")

d.h("4 通信设计")
d.p("发言时机：每次队长公布提案后、投票前有一轮公开讨论，从队长开始每人发一条结构化消息，后发言者能看到先发言者的内容。三次任务成功后，两名坏人在刺杀前各发言一次，只开放阵营判断槽，含义改为“我认为某人是梅林”。", boldlead="发言时机：")
d.table(
    ["槽位", "线下对应", "离散取值", "约束"],
    [
        ["① 身份声明", "我自称什么角色", "不声明、忠臣、派西维尔、梅林", "可虚假；历次声明公开可见"],
        ["② 阵营判断", "我认为谁好谁坏", "对其余五名玩家各选：不评、好、坏", "可虚假"],
        ["③ 判断依据", "我为什么这样判断", "对每名被评价的玩家选一条：任务、投票或组队记录（附轮次）、身份视野声明、无依据", "依附于 ②；公开记录类依据必须指向真实事件，但可以选择性引用"],
        ["④ 组队建议", "我点哪些人上车", "不建议，或一个合法队伍", "与提案共用队伍编码"],
        ["⑤ 行动倾向", "我赞成或反对，希望谁回应", "不表态、赞成、反对；可点名一名尚未发言的玩家", "不约束实际投票"],
    ], [1.0, 1.25, 2.3, 2.16]
)
d.p("设计原则：", boldlead="设计原则：")
d.bullets([
    ("独立开关：", "每个槽位单独开关。开启的槽位都含“不说”，所以每句话都可以不说；关闭的槽位固定为“不说”。"),
    ("允许说谎：", "内容可以虚假，裁判只检查合法性，不检查真假。"),
    ("让信道带有内在含义：", "公开记录类依据只能引用真实发生的事件，组队建议与提案共用队伍编码。理由有三：线下歪曲公开记录会被当场识破；可验证的依据与不可验证的身份视野声明分开后，可以检验“有依据的指控”是否更有说服力；Cao 等人的结果表明，有内在含义的信道比纯 cheap talk 更容易被利用。[8]"),
    ("范围：", "五槽消息相当于 CICERO 中的 intent，只决定“说什么”。[12] 一条完整消息覆盖线下约 2 分钟、约 400 tokens 发言中的主要信息，但不含措辞、语气与长篇推理。"),
    ("含义需实测：", "槽位名称不保证智能体学到的含义与名称一致，含义由第 6 节的真实性与听者影响指标确定。"),
])

d.h("5 实验设计")
d.table(
    ["配置", "身份", "判断", "依据", "组队", "倾向", "作用"],
    [
        ["S", "关", "关", "关", "关", "关", "沉默基线（对照组）：PPO self-play"],
        ["M", "关", "开", "开", "关", "关", "最小通信：谁好谁坏、为什么"],
        ["M+Claim", "开", "开", "开", "关", "关", "逐项加入：身份声明"],
        ["M+Team", "关", "开", "开", "开", "关", "逐项加入：组队建议"],
        ["M+Intent", "关", "开", "开", "关", "开", "逐项加入：行动倾向"],
        ["F", "开", "开", "开", "开", "开", "完整结构化通信"],
    ], [0.95, 0.6, 0.6, 0.6, 0.6, 0.6, 2.76], keep=True
)
d.p("核心层为 S、M、F；扩展层为在 M 上逐项加入一个槽位的三个配置。")
d.table(
    ["实验", "训练配置", "评估配置", "回答的问题"],
    [
        ["S → S", "沉默", "沉默", "对照组"],
        ["M → M", "最小通信", "最小通信", "加入谁好谁坏、为什么的作用"],
        ["F → F", "完整通信", "完整通信", "加入完整结构化通信的作用"],
        ["M+k → M+k", "逐项加入配置", "同一配置", "槽位 k 在最小通信之上的增量"],
        ["F → S\nM → S", "F 或 M", "冻结后全部槽位“不说”", "训练好的通信策略对信道的依赖"],
        ["F → F−k", "完整通信", "冻结后关闭槽位 k", "完整通信对槽位 k 的依赖"],
        ["S → S mask", "沉默", "遮蔽部分历史线索", "沉默基线中隐式信号是否被利用（RQ3）"],
    ], [1.0, 1.35, 1.75, 2.61]
)
d.bullets([
    ("主比较：", "F → F 相对 S → S。M → M 相对 S → S、F → F 相对 M → M 构成从沉默到完整通信的阶梯。"),
    ("两类消融互补：", "逐项加入衡量增量，测试时关闭衡量依赖；两者结论不一致，说明槽位之间存在交互。测试时关闭可能含有分布外效应，不等于重新学会在缺少该槽位时游戏。"),
    ("不做 S → F：", "沉默策略从未见过消息，评估时打开信道没有意义。"),
    ("判断依据的识别：", "② 与 ③ 总是同时开启，③ 的作用只能通过在 M、F 上测试时关闭 ③ 来衡量，得到的是依赖程度，不是训练适应后的作用。"),
    ("控制：", "各配置使用相同的网络结构、输出头、超参数与训练局数，只用掩码改变可用槽位。训练随机种子是重复单位，核心层每个配置 3 个种子。通信配置从同一种子的沉默 checkpoint 分支训练，以便配对比较；但这可能让通信策略沿用沉默时的打法，因此另设从头训练的对照。"),
])

d.h("6 指标与证据标准")
d.h("RQ1 游戏平衡", 2)
d.p("主指标为 ΔGood＝P(Good wins | F → F) − P(Good wins | S → S)。正值表示在本实验设定下交流更有利于好人，负值表示更有利于坏人；阶梯上的相邻差值同样报告。50% 只是胜率对称的参考点，不代表这款非对称游戏应当天然公平。")
d.p("跨种子评估：self-play 会学到只在同一训练群体内有效的约定。沉默基线中，好人对同种子坏人的胜率为 0.39，对其他种子坏人为 0.54。因此每个配置都报告 3 × 3 的好人 i 对坏人 j 胜率矩阵，ΔGood 分别在同种子对局（对角线）与跨种子对局（非对角线）上报告，用来区分交流本身的作用与种子特有的约定。跨种子只用于评估，不参与训练。", boldlead="跨种子评估：")
d.p("胜率分解：P(Good wins)＝P(三次任务成功) × [1 − P(刺杀命中梅林 | 三次任务成功)]，另报告三次任务失败与五次否决导致的坏人胜比例。刺杀成功率的分母只能是进入刺杀阶段的对局；在四名好人中随机猜梅林的命中率为 25%，作为刺杀参考线。任务成功率按任务序号报告，未发生的任务不能记作失败。", boldlead="胜率分解：")
d.h("RQ2 策略行为", 2)
d.table(
    ["行为", "量化方式", "解释限制"],
    [
        ["组队与投票", "提议及通过队伍中的好人比例；各角色赞成率；对含坏人队伍的赞成率", "按轮次与否决次数分层；最后机会的赞成不一定是误判"],
        ["任务破坏", "坏人在队时的 Fail 率；双坏人同队时 0、1、2 张 Fail 的分布", "放过任务可能是伪装"],
        ["梅林隐藏", "只用公开历史训练身份 probe，评估识别梅林的 log loss", "probe 认不出不等于刺客一定被骗"],
        ["辨别与混淆", "派西维尔对两名候选者的组队与投票偏好；莫甘娜对这种偏好的影响", "被支持不等于被认作梅林"],
        ["刺客推断", "条件刺杀准确率；对历史投票或组队记录的敏感度", "进入刺杀阶段的样本有选择偏差"],
    ], [1.0, 3.30, 2.41]
)
d.h("RQ3 隐式信号", 2)
d.p("以沉默基线 S 为主：没有显式信道时，公开行为是唯一可能的信息传递途径。S 中已观察到只在同种子内有效的隐式约定，例如刺客按座次或按最后一名队长猜梅林，对其他种子的梅林则接近随机；跨种子评估用来区分通用信号与种子特有的约定。证据分三层，结论只能达到实际完成的那一层：")
d.bullets([
    ("信息性：", "加入个人投票序列后，对隐藏身份的预测是否更准。只说明行为包含信息。"),
    ("接收者使用：", "遮蔽其他玩家的历史投票身份后，接收者的行动与收益是否改变，并设置 placebo 对照。说明行为线索被利用。"),
    ("策略性信号：", "在不改变表决结果的投票组合之间，信息持有者是否系统性地选择能影响队友的方式，并随搭档变化而调整。属于探索目标。"),
])
d.p("最低要求是完成前两层。显式信道开启后，再以 S 的信息性为参照，比较 M、F 中投票的信息量，判断隐式信号被补充还是被挤出。已有研究表明，单看相关性或任务成绩容易误判是否存在通信。[9]")
d.h("RQ4 消息组件", 2)
d.table(
    ["分析", "量化方式", "解释限制"],
    [
        ["槽位使用", "各槽位选择“不说”的比例随训练的变化，按角色与阵营分层", "使用率高不等于有用"],
        ["真实性", "身份声明与真实角色的一致率；被评为“坏”的玩家确为坏人的比例；梅林是否隐瞒、坏人是否冒充", "含义以行为为准，不以名称为准"],
        ["依据作用", "三类依据（公开记录、身份视野、无依据）对听者投票的影响", "只能通过测试时关闭 ③ 衡量"],
        ["听者影响", "替换单个槽位后，听者投票与组队分布的变化", "影响大不等于对己方有利"],
        ["胜率贡献", "逐项加入的增量与测试时关闭的依赖，按阵营报告", "两者不一致说明存在交互"],
    ], [1.0, 3.30, 2.41]
)

d.h("7 结论边界")
d.bullets([
    "结论只适用于所定义的槽位、取值范围、角色配置和训练所得策略。结构化消息不等于自由讨论；要声称自然语言交流的作用，需另做自然语言实验并单独报告。",
    "有限预算下的 self-play 策略不是纳什均衡，本研究测量的是特定学习系统的经验表现，不以求解均衡为目标：两队博弈中队友之间不能共享私有信息，其均衡在理论上就比两人零和博弈难求 [16]；以 PPO 加 MMD 均衡正则 [14] 训练的对照在 150 万局时近似可利用度 [15] 为 0.60，普通 self-play 为 0.65，并未明显接近均衡。同一训练群体内的胜率只说明该群体内的平衡，需要跨种子评估加以区分。",
    "通信没有被学会时，报告为学习失败，不推广为“交流没有价值”；胜率差很小时，报告区间与可以排除的效应大小；RQ3 只有相关性时，保留为探索结果。",
])

d.h("参考文献")
d.ref(1, "Serrino, J., Kleiman-Weiner, M., Parkes, D. C., & Tenenbaum, J. B. (2019). Finding Friend and Foe in Multi-Agent Games. NeurIPS 2019.", "https://arxiv.org/abs/1906.02330")
d.ref(2, "Nakamura, Y. (2017). Is it Percival time yet? A preliminary analysis of Avalon gameplay and strategy. SIGBOVIK 2017.", "https://sigbovik.org/2017/proceedings.pdf")
d.ref(3, "Light, J., Cai, M., Chen, W., Wang, G., Chen, X., Cheng, W., Yue, Y., & Hu, Z. (2025). Strategist: Self-improvement of LLM Decision Making via Bi-Level Tree Search. ICLR 2025.", "https://openreview.net/forum?id=gfI9v7AbFg")
d.ref(4, "Light, J., Cai, M., Shen, S., & Hu, Z. (2023). AvalonBench: Evaluating LLMs Playing the Game of Avalon. NeurIPS 2023 Foundation Models for Decision Making Workshop.", "https://arxiv.org/abs/2310.05036")
d.ref(5, "Lan, Y., Hu, Z., Wang, L., Wang, Y., Ye, D., Zhao, P., Lim, E.-P., Xiong, H., & Wang, H. (2024). LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay. EMNLP 2024, 128–145.", "https://aclanthology.org/2024.emnlp-main.7/")
d.ref(6, "Brandizzi, N., Grossi, D., & Iocchi, L. (2021). RLupus: Cooperation through Emergent Communication in The Werewolf Social Deduction Game. Intelligenza Artificiale, 15(2), 55–70.", "https://arxiv.org/abs/2106.05018")
d.ref(7, "Foerster, J. N., Assael, Y. M., de Freitas, N., & Whiteson, S. (2016). Learning to Communicate with Deep Multi-Agent Reinforcement Learning. NeurIPS 2016.", "https://arxiv.org/abs/1605.06676")
d.ref(8, "Cao, K., Lazaridou, A., Lanctot, M., Leibo, J. Z., Tuyls, K., & Clark, S. (2018). Emergent Communication through Negotiation. ICLR 2018.", "https://arxiv.org/abs/1804.03980")
d.ref(9, "Lowe, R., Foerster, J., Boureau, Y.-L., Pineau, J., & Dauphin, Y. (2019). On the Pitfalls of Measuring Emergent Communication. AAMAS 2019.", "https://arxiv.org/abs/1903.05168")
d.ref(10, "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.", "https://arxiv.org/abs/1707.06347")
d.ref(11, "Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. NeurIPS 2022 Datasets and Benchmarks Track.", "https://arxiv.org/abs/2103.01955")
d.ref(12, "Meta Fundamental AI Research Diplomacy Team (FAIR), Bakhtin, A., Brown, N., et al. (2022). Human-level play in the game of Diplomacy by combining language models with strategic reasoning. Science, 378(6624), 1067–1074.", "https://www.science.org/doi/10.1126/science.ade9097")
d.ref(13, "The Resistance: Avalon Rulebook. Indie Boards & Cards.", "https://cdn.1j1ju.com/medias/a6/dc/c1-the-resistance-avalon-rulebook.pdf")
d.ref(14, "Sokota, S., D'Orazio, R., Kolter, J. Z., Loizou, N., Lanctot, M., Mitliagkas, I., Brown, N., & Kroer, C. (2023). A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games. ICLR 2023.", "https://arxiv.org/abs/2206.05825")
d.ref(15, "Timbers, F., Bard, N., Lockhart, E., Lanctot, M., Schmid, M., Burch, N., Schrittwieser, J., Hubert, T., & Bowling, M. (2022). Approximate Exploitability: Learning a Best Response. IJCAI 2022.", "https://arxiv.org/abs/2004.09677")
d.ref(16, "Celli, A., & Gatti, N. (2018). Computational Results for Extensive-Form Adversarial Team Games. AAAI 2018.", "https://arxiv.org/abs/1711.06930")
d.p("[C1] CS7329 Fall 2026. Current Topics Presentation and Paper (Graduate Students). 课程项目要求。")
d.p("[C2] Hahsler, M. CS 7329 Reinforcement Learning 课程大纲与 Introduction_to_Reinforcement_Learning 课程仓库。")

d.save("Avalon_Research_Design.docx")
