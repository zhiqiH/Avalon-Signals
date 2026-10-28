"""Generates Avalon_Implementation_Spec.docx: engineering details for the research design."""
from docgen import DocBuilder

d = DocBuilder(
    title="六人阿瓦隆中显式交流的作用",
    subtitle="Implementation Spec: 工程实现方案",
    meta_line="CS7329 Reinforcement Learning ｜ 实现方案 ｜ 2026年10月28日 ｜ 配套：《研究设计》",
    subject="Implementation Spec: Structured Communication in Avalon",
    keywords="Avalon, MARL, PPO, Implementation, CS7329",
)

d.h("0 范围一览")
d.p("本文件只记录怎么实现。配置名称（S、M、M+k、F）与实验条件（如 F → S）的定义见《研究设计》第 5 节。")
d.table(
    ["模块", "必须实现", "可选"],
    [
        ["环境", "六人状态机、信息过滤、规则与信息泄露测试、随机与规则对手", "PettingZoo 接口包装"],
        ["消息协议", "五槽编码、开关掩码、依据真实性掩码、消息元组日志", "模板渲染成文本"],
        ["策略与训练", "信息状态向量上的 MLP actor–critic、带掩码的统一动作空间、PPO 与 GAE、self-play 对手池、沉默优先的分支训练", "消息头的行为克隆初始化；从头训练的 F 对照"],
        ["配置与种子", "S、M、F，各 3 个种子", "M+Claim、M+Team、M+Intent；增至 5 个种子"],
        ["诊断与评估", "三项通信诊断、冻结评估、跨种子评估、按种子配对的 bootstrap", "近似可利用度（诊断）"],
        ["扩展", "—", "自然语言实验；蒸馏（后续工作）"],
    ], [1.15, 3.36, 2.20]
)

d.h("1 环境")
d.bullets([
    ("自写状态机：", "AvalonBench 的引擎只支持五人局且未标明许可证，只作规则与接口参考。[7]"),
    ("阶段顺序：", "分配角色 → 队长提案 → 讨论（仅通信配置）→ 全员投票 → 通过则执行任务，否则换队长 → 三胜或三败判定 → 刺杀前讨论（仅通信配置）→ 刺杀 → 结算。"),
    ("动作空间：", "所有阶段共用一个 30 维动作空间，并按行动者的相对座位编码（0 为自己），因此同一个网络可以服务任意座位：前 20 维为提案，枚举当前任务人数下的全部合法队伍（C(6,2)=15、C(6,3)=20、C(6,4)=15）；之后依次为投票（赞成、反对）、任务牌（只有坏人可选 Fail，好人强制 Success）、刺杀目标（两名坏人以外的四名玩家）。"),
    ("同时动作：", "投票与任务牌先全部收集再一起揭示，执行顺序不能泄露未公开的选择。"),
    ("信息过滤：", "每名玩家的观察只含公共历史、自身角色与本角色允许的私有信息。派西维尔的两名候选者用无序集合表示；座位、角色与初始队长每局随机。"),
    ("真值隔离：", "完整状态只供裁判与离线分析，和策略输入分开存储。"),
])
d.p("必须写成单元测试的项目：", boldlead="必须写成单元测试的项目：")
d.bullets([
    "规则：各轮任务人数、四票通过与平票否决、连续五次否决、单张 Fail 即失败、三胜三败、刺杀判定；",
    "信息泄露：逐角色检查观察中不含不该知道的身份；派西维尔候选者的排列顺序不随真实身份变化；",
    "消息掩码：关闭槽位强制“不说”；③ 只能指向真实事件；点名只限尚未发言者；队长的 ④ 固定为其提案。",
])

d.h("2 消息协议")
d.table(
    ["槽位", "输出头", "取值数", "掩码规则"],
    [
        ["① 身份声明", "1 个分类头", "4", "—"],
        ["② 阵营判断", "5 个分类头，每名其他玩家一个", "每个 3", "刺杀阶段只开放此槽，取值改为是否梅林候选"],
        ["③ 判断依据", "每名被评价玩家各一个依据类型头与一个轮次头", "类型 5，轮次至多 5", "② 为“不评”时屏蔽；公开记录类只能选真实发生的事件"],
        ["④ 组队建议", "1 个分类头，共用提案的队伍编码", "至多 21", "队长固定为其提案"],
        ["⑤ 行动倾向", "投票倾向头与点名头", "3；至多 6", "点名只限本轮尚未发言的玩家"],
    ], [1.0, 2.35, 1.15, 2.21]
)
d.bullets([
    ("“不说”：", "每个槽位的第一个取值即“不说”（不声明、不评、不建议、不表态、不点名）。"),
    ("关闭与不说要区分：", "关闭的槽位强制“不说”，不进入损失；观察中编码为“关闭”，与主动选择“不说”区分开。"),
    ("训练后关闭信道：", "F → S 与 M → S 实现为强制所有槽位“不说”，不引入训练中从未出现的新 NULL 符号，以减轻分布外输入。"),
    ("存储与渲染：", "消息以槽位元组存储；可用模板渲染成句子以便阅读，例如“我是派西维尔。我认为 3 号是坏人，因为第 2 轮任务失败时他在队里。我建议 1、2、5 号上车。我反对当前队伍，请 4 号回应。”"),
])

d.h("3 策略与价值网络")
d.bullets([
    ("结构：", "IPPO。好人、坏人各一套 actor–critic，actor 与 critic 为两个独立的 MLP（默认两层 256 维）；同阵营所有座位共享参数，角色作为输入。架构在正式比较前冻结。"),
    ("为什么不用循环网络：", "一局的公开历史有上限（至多 5 个任务 × 5 次提案），把它连同私有信息与自己的任务牌完整编码成固定长度向量（沉默配置为 615 维），就是一个完整的信息状态，前馈网络即可，不需要 GRU。这样也省去了 BPTT 和 hidden state 管理，RQ3 的遮蔽干预可以直接修改向量字段。通信配置的消息记录同样按提案定长编码；若维度过大，再考虑在提案序列上使用循环网络或注意力。"),
    ("输入：", "信息状态向量：角色、本角色允许的私有信息、当前阶段与队伍、完整的提案与投票历史（投票编码为 +1 赞成、−1 反对、0 未投）、任务结果与自己的任务牌，以及通信配置下按槽位编码的消息记录。所有按座位索引的字段都旋转到行动者的相对座位；各字段在向量中的位置有命名索引（LAYOUT）。"),
    ("输出：", "一个 30 维策略头配合合法动作掩码：非法动作的 logit 置为极小值，熵只在合法动作上计算；等价于各阶段各用一个动作头。critic 输出该玩家的状态价值。"),
    ("消息的概率：", "一条消息的对数概率为已开启子动作对数概率之和，与其他动作进入同一个 PPO 比率。所有配置保留全部输出头，只用掩码区分，以保证网络容量一致。"),
    ("RIAL 式训练：", "消息头与行动头共享编码器，用同一个终局奖励训练；不用让梯度穿过信道的 DIAL，因为听者中有对手。[3]"),
    ("Critic：", "只使用该玩家可获得的信息。若日后改用集中式 critic，需单独说明训练时用到的特权信息，并保证 actor 看不到。"),
])

d.h("4 奖励与更新")
d.bullets([
    ("奖励：", "己方获胜 +1、失败 −1，中间奖励为 0；γ＝1，用 GAE 估计 advantage。[1]"),
    ("损失汇总：", "两阵营分别汇总损失，避免四名好人对两名坏人造成隐含的权重差异。"),
    ("不计入损失：", "关闭槽位、强制“不说”和其他强制动作不计入 actor 损失；padding 不参与梯度。"),
    ("不做奖励塑形：", "不给“诚实发言”“成功传信”“隐瞒身份”等额外奖励。"),
    ("超参数：", "在独立 pilot 中确定后，所有配置共用，不能按测试胜率分别挑选。价值归一化、批大小、clip 范围与参数共享方式以 MAPPO 的经验为起点。[2]"),
])

d.h("5 训练流程")
d.bullets([
    ("第 1 步 沉默基线：", "先训练 S，直到对规则对手的胜率稳定（关卡一：最后约 10 万局与之前约 10 万局的均值相差 < 0.02）。S 不过关时不训练任何通信配置，先排查环境、奖励、掩码与 credit assignment。"),
    ("第 2 步 分支：", "每个种子把 S 训练到总预算 B 的一半，作为分支点。S 从分支点继续训练剩余一半；M、F 以及扩展层的 M+k 从同一分支点加入消息头，再训练相同局数。新增的消息输入权重初始化为零，使分支时的行为与 S 完全相同；消息输出头从均匀分布起步。"),
    ("第 3 步 对手池：", "pilot 中尝试每局 50% 当前对手、25% 历史对手、25% 规则对手，正式实验前锁定比例。分支后对手池只收录本配置的 checkpoint；对手的轨迹不参与 on-policy 更新。"),
    ("预算：", "B＝150 万局（2,930 次迭代，每次 512 局）。S 在 100 万局时仍有种子未稳定，150 万局时对规则对手的六项检查全部稳定。分支点为约 75 万局的 checkpoint_01460（747,520 局）。核心层 S、M、F 各 3 个种子，分支设计下前半段共享；扩展层 M+k 各 3 个种子。"),
    ("从头训练对照（可选）：", "至少保留一个从头训练的 F 种子，检查热启动是否压低了交流的作用。若分支导致消息头完全不被使用，改为所有配置从头训练、预算相同。"),
])

d.h("6 通信学习诊断")
d.bullets([
    ("玩具检查：", "参照 RLupus 的设置，把坏人固定为规则策略，只训练好人在 M 下通信。[6] 若梅林的阵营判断仍不能提高好人胜率，优先排查实现错误。"),
    ("发送端：", "计算各槽位取值与发言者真实身份、被评价者真实阵营之间的互信息，按角色分层。"),
    ("接收端：", "在冻结 checkpoint 上替换某条消息的某个槽位，测量听者投票与组队分布的 KL 变化，即 causal influence。[4, 5]"),
    ("关卡二：", "M 的接收端影响接近零时不进入正式训练，先减少开启的槽位或缩小取值范围。"),
    ("可选初始化：", "先用“诚实规则发言者”的行为克隆初始化消息头，再用 RL 微调；若采用，所有通信配置统一使用，并在论文中说明。"),
])

d.h("7 评估与统计")
d.bullets([
    ("冻结评估：", "每个训练种子在其全部评估条件下各评估 1,000 局。核心层每组种子 10 个条件：S → S；M → M、M → S、M 关闭 ③；F → F、F → S，以及分别关闭身份、依据、组队、倾向的四个 F → F−k。3 个种子合计约 3 万局。"),
    ("配对：", "各条件使用相同的角色分配与初始队长表，策略采样用独立随机流；评估中没有训练更新与跨局记忆。"),
    ("RQ3 遮蔽干预：", "遮蔽其他玩家的历史投票身份，但保留票数、提案结果与当前合法动作，直接修改信息状态向量中的对应字段（投票置 0，即“未知”）；用同长度占位作 placebo；用训练时见过遮蔽输入的辅助条件检验分布外影响。"),
    ("probe 数据划分：", "按完整对局和训练种子划分训练集与测试集，避免同一对局的片段同时出现在两边。"),
    ("统计：", "先报告每个种子的指标，再汇总配对差值、效应量与 95% 区间：按训练种子配对做分层 bootstrap，局内事件按整局归组。单个模型的胜率可另给 Wilson 区间，但不能替代训练间的不确定性。好人最终胜率是预先指定的主指标。"),
    ("跨种子评估：", "每个配置都用 scripts/crossplay.py 报告 3 × 3 的好人 i 对坏人 j 矩阵（各 1,000 局配对对局），分别汇总同种子对角线与跨种子非对角线；ΔGood 同时在两者上报告。只做评估，不做跨种子训练。"),
    ("近似可利用度（可选诊断）：", "冻结一方，从该 checkpoint 自身对应阵营的网络出发，用相同的 PPO 设置训练 10 万局最优反应，可利用度＝BR 好人胜率＋BR 坏人胜率－1（scripts/best_response.py、exploitability.py）。[11] 只作诊断，不作训练目标或关卡。"),
])

d.h("8 数据记录")
d.p("每局至少记录：game id、训练与评估 seed、配置与槽位开关、checkpoint、座位与角色映射、合法私有观察、提案、消息元组（及可选的渲染文本）、公开投票、私有任务动作、公开任务结果、刺杀目标与最终胜负。私有真值日志与策略输入分开保存；同时保存配置文件、依赖版本、代码版本与分析脚本。")

d.h("9 算力")
d.bullets([
    ("本地优先：", "远程 Windows 游戏本（Ryzen 9 8945HX 16 核、RTX 5070 Ti Laptop 12 GB）。实测网络较小时 CPU 训练更快，GPU 同步还曾导致挂起，因此训练用 CPU：每个进程 4 线程，多个种子并行。PyTorch 选用支持 RTX 50 系的 CUDA 12.8 版本，以备更大的网络。"),
    ("预算：", "训练时间约为 配置数 × 种子数 × B × 每局训练耗时（分支设计下前半段共享），另加调参与评估。实测 3 个种子并行时每 50 万局约 2–2.5 小时，完整的 150 万局约 6 小时。超出预算时先减小 B，保留种子数。"),
    ("HPC：", "只用于并行独立种子与扩展评估，核心部分不依赖 HPC 配额。"),
    ("环境记录：", "开跑前记录显存、内存、功耗限制与软件版本。"),
])

d.h("10 自然语言扩展（可选）")
d.bullets([
    ("语言层：", "用模板或 LLM 把五槽消息渲染成自然语言，或把自然语言解析回五槽；这一层不需要懂阿瓦隆策略。[9]"),
    ("实验：", "冻结的本地 9B 模型或固定版本的低成本 API 模型扮演六名玩家，上下文彼此隔离。发言以 JSON 输出五个槽位，关闭的槽位从格式中删除，另可附不超过 400 tokens 的文本。NL-S、NL-M、NL-F 对应 S、M、F。先跑 20 局 pilot，再每个条件 100–300 局；非法输出按固定的重试与回退规则处理并报告频率，不丢弃整局。结果单独报告。"),
    ("资源：", "9B 权重在 FP16 下约 18 GB、4-bit 下约 4.5 GB，不含 KV cache 与运行开销；12 GB 显存可尝试量化推理，不做 LLM 的 RL 训练。API 预算上限 50 美元。"),
    ("蒸馏（后续工作）：", "通用 LLM 的阿瓦隆水平有限：AvalonBench 中 ChatGPT 扮演好人、对阵规则 bot 坏人时胜率为 22.2%，规则 bot 好人为 38.2%。[8] 商业 API 的条款也常限制用其输出训练模型。更可行的路线是以本项目训练出的 RL 策略为教师做 policy distillation[10]，或让开源模型自博弈、只保留胜局做行为克隆；12 GB 显存下，1–4B 模型的 LoRA 或 QLoRA 微调可行。"),
])

d.h("参考文献")
d.ref(1, "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.", "https://arxiv.org/abs/1707.06347")
d.ref(2, "Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. NeurIPS 2022 Datasets and Benchmarks Track.", "https://arxiv.org/abs/2103.01955")
d.ref(3, "Foerster, J. N., Assael, Y. M., de Freitas, N., & Whiteson, S. (2016). Learning to Communicate with Deep Multi-Agent Reinforcement Learning. NeurIPS 2016.", "https://arxiv.org/abs/1605.06676")
d.ref(4, "Lowe, R., Foerster, J., Boureau, Y.-L., Pineau, J., & Dauphin, Y. (2019). On the Pitfalls of Measuring Emergent Communication. AAMAS 2019.", "https://arxiv.org/abs/1903.05168")
d.ref(5, "Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P. A., Strouse, D. J., Leibo, J. Z., & de Freitas, N. (2019). Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning. ICML 2019.", "https://arxiv.org/abs/1810.08647")
d.ref(6, "Brandizzi, N., Grossi, D., & Iocchi, L. (2021). RLupus: Cooperation through Emergent Communication in The Werewolf Social Deduction Game. Intelligenza Artificiale, 15(2), 55–70.", "https://arxiv.org/abs/2106.05018")
d.ref(7, "Light, J. et al. Avalon-LLM：AvalonBench 与 STRATEGIST 代码库；引擎位于 avalonbench_dev，README 说明目前只支持五人局。", "https://github.com/jonathanmli/Avalon-LLM")
d.ref(8, "Light, J., Cai, M., Shen, S., & Hu, Z. (2023). AvalonBench: Evaluating LLMs Playing the Game of Avalon. NeurIPS 2023 Foundation Models for Decision Making Workshop.", "https://arxiv.org/abs/2310.05036")
d.ref(9, "Meta Fundamental AI Research Diplomacy Team (FAIR), Bakhtin, A., Brown, N., et al. (2022). Human-level play in the game of Diplomacy by combining language models with strategic reasoning. Science, 378(6624), 1067–1074.", "https://www.science.org/doi/10.1126/science.ade9097")
d.ref(10, "Rusu, A. A., Colmenarejo, S. G., Gulcehre, C., Desjardins, G., Kirkpatrick, J., Pascanu, R., Mnih, V., Kavukcuoglu, K., & Hadsell, R. (2016). Policy Distillation. ICLR 2016.", "https://arxiv.org/abs/1511.06295")
d.ref(11, "Timbers, F., Bard, N., Lockhart, E., Lanctot, M., Schmid, M., Burch, N., Schrittwieser, J., Hubert, T., & Bowling, M. (2022). Approximate Exploitability: Learning a Best Response. IJCAI 2022.", "https://arxiv.org/abs/2004.09677")

d.save("Avalon_Implementation_Spec.docx")
