# Structured Communication in Six-Player Avalon

Graduate term project for **CS 7329 Reinforcement Learning** at Southern Methodist University (Fall 2026).

What do agents gain from talking in a hidden-role game? This project trains multi-agent reinforcement learning agents to play six-player *The Resistance: Avalon*. They first play **in silence**, using only public actions such as team proposals, votes and mission results. Then they play with a **structured five-slot message channel**. Comparing the two shows what explicit communication adds to win rates, to strategic behavior, and to the information carried by public actions.

This README is the experiment plan and design. It will be replaced by a results write-up once all experiments are finished.

## 1 Research Questions

| | Question | Main evidence |
|---|---|---|
| **RQ1** Game balance | Relative to the silent baseline, which team gains more when communication is added? | Good win rate; probability of three successful missions; assassination accuracy given three successes |
| **RQ2** Strategic behavior | How do team building, voting, sabotage and identity concealment change? | Behavior metrics by role and game state |
| **RQ3** Implicit signaling | In silent play, do public actions carry information? Once explicit messages exist, is that implicit signal complemented or crowded out? | Information gain, receiver interventions, cross-seed checks |
| **RQ4** Message components | Which of the five message types account for the effect of communication? | Slot-by-slot training ablations, test-time slot removal, message truthfulness and listener influence |

**Why reinforcement learning.** Each player sees only part of the state and must infer identities from the game history while making sequential decisions. Reward arrives only at the end of a game, and good ways to propose, vote and speak are not known in advance. Self-play policy gradient methods can learn such policies, and different communication settings can then be compared under controlled conditions.

**Why silence is the baseline.** Silent play has fewer decisions and no cheap-talk problem of messages being ignored, so it trains into a stable baseline that every communication configuration can start from. The question is therefore what communication *adds*, without assuming a direction. Talk may help good players coordinate, but it may also help evil players deceive or find Merlin. Different message types may even pull in opposite directions: a role claim can help Percival, and it can also expose Merlin to the Assassin.

## 2 Related Work

| Work | Approach | Relation to this project |
|---|---|---|
| DeepRole [1] | Five-player Avalon; CFR with deductive reasoning and a value network; no language | Precedent for strong silent play, which supports silence as the baseline |
| Is it Percival time yet? [2] | Descriptive statistics of 61 human games | Source of metrics: Percival claims, the first failed mission, evil coordination failures; not causal |
| Strategist [3] | LLM with tree search | Its appendix has six-player human games with and without discussion, but without Percival and Morgana and without controlled training |
| Lan et al. [4] | LLM agents in six-player Avalon with Percival and Morgana | Behavior categories (camouflage, sharing, persuasion) for RQ2 and RQ4 |
| RLupus [5] | Werewolf; PPO-trained discrete signals for villagers | Closest precedent: villager win rate rose from 0.044 without communication to at most 0.45, and more bandwidth did not always help |
| RIAL [6], Cao et al. [7] | Messages trained as RL actions; self-interested agents struggle to use cheap talk | How messages are trained, and the main risk |
| Lowe et al. [8] | Measuring emergent communication | Separating "a message carries information" from "a listener uses it" |
| PPO [9], MAPPO [10] | Clipped policy gradient; practical multi-agent PPO | Training algorithm and starting hyperparameters |
| Hidden-role games [11], PSRO [12] | Equilibrium concepts for hidden-role games; best responses as a measure of distance from equilibrium | Why this project measures a learning system instead of targeting an equilibrium |

**Intended contributions.**
- A controlled ladder from silent play to full communication, with rules, algorithm and budget held fixed.
- A five-slot protocol whose slots switch on and off independently, so "communication" splits into message types that can be added one at a time.
- Win rates decomposed into the mission path and the assassination path, with interventions that separate "behavior carries information" from "others use that behavior".

## 3 Game Setup

**Roles.** Two Loyal Servants, Merlin, Percival, Morgana and the Assassin. Good is Merlin, Percival and the two Servants; evil is Morgana and the Assassin. The rules follow the game description in the Strategist appendix [3].

| Item | Rule |
|---|---|
| Missions and votes | Mission sizes 2-3-4-3-4. A team needs at least four approvals; a tie rejects. Five rejected proposals in one mission give the win to evil. |
| Missions | Good players must play Success; evil players choose Success or Fail. One Fail fails the mission; only the number of Fail cards is revealed. |
| End of game | Three failed missions: evil wins. Three successful missions: the Assassin names one player, and evil wins if that player is Merlin. |
| Private knowledge | Merlin sees both evil players. Percival sees Merlin and Morgana as two candidates without knowing which is which. The evil players know each other. Servants know nothing extra. |
| Randomization | Roles and the first leader are random in every game; no Lady of the Lake or other add-ons. |

Players see only the public history and the private knowledge their role allows. True roles are used only by the referee and for offline analysis.

## 4 Communication Design

**Timing** follows common tabletop play.
- Every mission opens with one round of public discussion before its first proposal. Players speak once each in seat order, starting left of the leader, and the leader speaks last. Then the leader proposes and everyone votes.
- After a rejected proposal, leadership passes on and the next leader proposes at once, without discussion.
- After the third successful mission, Morgana and then the Assassin each speak once to agree on who Merlin is, before the assassination.
- There is no dialogue: nobody can address or question another player.

**Five slots.** Each slot can be switched on or off independently, and every slot can say nothing. Masks enforce the rules below; full details are in `src/comm/protocol.py`.

| Slot | Content | Values | Rules |
|---|---|---|---|
| ① Role claim | Who I claim to be | none, Servant, Percival, Merlin | No evil role can be claimed; claims may be false |
| ② Judgment | What I think each other player is | per player: none, good, evil, Servant, Merlin, Percival, Morgana, Assassin | Any subset of players; together with the claim, no role more often than the deck holds it (at most two players judged evil, four judged good) |
| ③ Rationale | Why I judge so | per judged player: a mission, vote or proposal record (with its mission number), role knowledge, or no basis | Cites only finished missions and points the same way as the judgment; role knowledge must match what the claimed role can see |
| ④ Team suggestion | Who should go on the mission | none, or any team of the current size | Nobody the message judges evil; the leader makes no suggestion, since its proposal follows at once |
| ⑤ Action intent | How I will vote, and what I will do if the mission fails | per player: none; **veto** (I reject any team with them this round); **require** (I reject any team without them); **bind** (they and I go together: I reject any team with exactly one of us); **blame** (if this mission fails with them on it, I reject their teams from then on) | No veto on a player judged good, no require or bind with a player judged evil; some team of the current size, and my own suggestion if I make one, must satisfy all vote conditions |

How a record points:
- **mission record:** the player was on mission *q*, which failed (evil) or succeeded (good);
- **vote record:** the player approved mission *q*'s team that then failed, or rejected one that then succeeded (evil); the reverse points to good;
- **proposal record:** the player led mission *q*'s team, which failed (evil) or succeeded (good);
- **role knowledge:** "good" or "evil" needs a Merlin claim or no claim; "Merlin" or "Morgana" needs a Percival claim or no claim.

In the evil talk before the assassination, only judgments of good players as Servant, Merlin or Percival are open (with a record rationale, without role knowledge).

**Design principles.**
- **Lies and partial statements are allowed, contradictions are not.** The referee never checks truth, but one message must be internally consistent. For example, it cannot call two players Merlin, cite a successful mission as proof that someone is evil, or suggest a team containing a player it accuses.
- **Messages concern the current round only.** A player may change its story later; all past messages stay public, so changing one's story is itself a signal.
- **Two chains.** Claim → judgment → rationale is a chain of reasoning. Judgment → team suggestion → intent is a chain of action: intents announce and explain one's votes and one's reaction to a failed mission. Talk never binds actions: votes may contradict an announced intent.
- **Grounded rationales.** Record-based rationales must cite real events and point the same way as the judgment. There are three reasons. At a real table a distorted record is caught at once. Separating verifiable records from unverifiable role knowledge lets the analysis test whether grounded accusations persuade more. And channels with intrinsic meaning are easier to exploit than pure cheap talk [7]. Evil players can still mislead by citing true records selectively.
- **Scope.** A message only decides *what* to say. It covers the main content of about two minutes of table talk, but not wording, tone or long arguments.
- **Meaning is measured, not assumed.** Slot names do not guarantee that agents use them in that sense; truthfulness and listener-influence metrics (Section 7) decide what a slot means.

## 5 Experimental Design

**Configurations.**

| Config | ① | ② | ③ | ④ | ⑤ | Purpose |
|---|:-:|:-:|:-:|:-:|:-:|---|
| S | | | | | | Silent baseline (control): self-play PPO |
| M | | ✓ | ✓ | | | Minimal communication: who is good or evil, and why |
| M+Claim | ✓ | ✓ | ✓ | | | Adds role claims |
| M+Team | | ✓ | ✓ | ✓ | | Adds team suggestions |
| M+Intent | | ✓ | ✓ | | ✓ | Adds action intent |
| F | ✓ | ✓ | ✓ | ✓ | ✓ | Full structured communication |

The core layer is S, M and F; the extension layer is the three M+k configurations.

**Experiments.**

| Experiment | Trained as | Evaluated as | Question |
|---|---|---|---|
| S → S | silent | silent | Control |
| M → M | M | M | Effect of saying who is good or evil, and why |
| F → F | F | F | Effect of full structured communication |
| M+k → M+k | M+k | same | What slot *k* adds on top of M |
| F → S, M → S | F or M | frozen, every slot forced to say nothing | How much a trained communicating policy depends on the channel |
| F → F−k | F | frozen, slot *k* switched off | How much F depends on slot *k* |
| S → S masked | silent | parts of the history hidden | Whether the silent baseline uses implicit signals (RQ3) |

- **Main comparison:** F → F against S → S. M → M against S → S and F → F against M → M form the ladder from silence to full communication.
- **Two complementary ablations.** Adding a slot measures its increment; removing it at test time measures dependence. If the two disagree, the slots interact. Test-time removal can include out-of-distribution effects and is not the same as relearning to play without that slot.
- **No S → F:** a silent policy has never seen messages, so opening the channel at test time is meaningless.
- **Identifying the rationale:** ② and ③ are always switched on together, so ③ can only be measured by switching it off at test time in M and F. That measures dependence, not the effect after training adapts.
- **Controls:** all configurations share the network, output heads, hyperparameters and training budget; masks alone change which slots are available. The training seed is the unit of replication, with 3 seeds per configuration. Communication configurations branch from the same seed's silent checkpoint so that comparisons are paired. Since a branched policy may keep its silent habits, an optional from-scratch F run checks whether warm-starting suppresses communication.

## 6 Training

**Environment.**
- A custom six-player state machine (six players and five-slot talk both need custom rules). Phase order: roles → discussion (communication configurations only) → proposal → vote → mission or next leader → three successes or three failures → evil talk (communication configurations only) → assassination → result.
- One flat 30-way action space for all phases, in seat-relative coordinates (0 is the acting player), so one network serves every seat:
  - 20 proposal actions, enumerating every legal team of the current size (C(6,2) = 15, C(6,3) = 20, C(6,4) = 15);
  - then approve/reject, Success/Fail (only evil players may play Fail), and the assassination target (the four players other than the two evil ones).
- Votes and mission cards are collected first and revealed together, so execution order leaks nothing.
- Each player's observation holds only the public history, its own role and its role's private knowledge. Percival's two candidates are an unordered set. Seats, roles and the first leader are random. The full state is kept apart from policy inputs and used only by the referee and for analysis.

**Observation.** The silent observation is a 615-dimensional information-state vector:
- the player's role and private knowledge;
- the phase, mission, attempt, leader and current team;
- the complete proposal and vote history (votes as +1 approve, −1 reject, 0 not cast; at most 5 missions × 5 proposals);
- mission results and the player's own mission cards.

Every seat-indexed field is rotated into the acting player's frame, and a named index (`LAYOUT`) locates each field. Because the history is bounded, this vector is a complete information state, so plain MLPs suffice: no recurrence, no BPTT, and RQ3 interventions can edit the vector directly.

Communication configurations add a 4,729-dimensional message block: 13 global features plus 36 fixed message slots of 131 features each (one per mission and speaker, plus the evil talk). The actor and the critic each compress every message slot with a shared linear encoder (131 → 32, tanh), and the result is concatenated with the silent vector.

**Network.**
- Independent PPO with one actor–critic per team, shared by all seats of that team; the role is an input. The actor and the critic are separate MLPs (two layers of 256 units).
- One 30-way action head with legal-action masking: illegal logits are set to a large negative value, and entropy is computed over legal actions only.
- Communication configurations add 22 message heads (140 logits) in the speaker's frame:
  - claim (4 values);
  - a judgment per other player (8 values);
  - a rationale type (5 values) and a mission number (5 values) per other player;
  - a team (up to 21 values, shared with the proposal encoding);
  - an intent per other player (5 values).
- Sampling runs in two stages. The claim and judgments come first. Rationales, team and intents follow, with their logits conditioned on the first stage through a zero-initialized linear map, and consistency masks applied throughout.
- The log-probability of a message is the sum over the heads used, and it enters the same PPO ratio as game actions. Every configuration keeps all heads and differs only in masks, so network capacity is identical.
- Messages are trained RIAL-style [6] with the game reward. DIAL is not used, because listeners include opponents. The critic sees only the player's own information.

**Reward and update.**
- Win +1, loss −1, no intermediate reward; γ = 1 with GAE (λ = 0.95) [9]. There is no reward shaping for honesty, signaling or concealment.
- The two teams' losses are summed separately, so the 4-vs-2 team sizes never weight each other.
- Switched-off slots, forced silence and other forced actions do not enter the actor loss.
- Hyperparameters are shared by every configuration and were set from the PPO and MAPPO defaults [9, 10]: learning rate 3e-4, 4 epochs, minibatch 2,048, clip 0.2, value coefficient 0.5, entropy coefficient 0.01, gradient norm 0.5, normalized advantages, 512 games per iteration.

**Opponents.** In each training game, both teams use the current policies (50%), or one team is replaced by a past snapshot (25%) or by the rule-based player (25%). A snapshot is added every 10 iterations, and the pool keeps 20 per team. Only current policies learn. Rule players say nothing and ignore messages, so they are a fixed reference.

**Training pipeline.**
1. **Silent baseline.** Train S until its win rates against rule players are stable (gate 1: the mean over the last ~100k games differs from the previous ~100k by less than 0.02). No communication configuration is trained until S passes.
2. **Branching.** The budget is B = 1,500,160 games (2,930 iterations). Each seed's S checkpoint at half the budget (`checkpoint_01460.pt`, 747,520 games) is the branch point. M, F and the M+k configurations add the message heads there and train the remaining 752,640 games (1,470 iterations). New message inputs start at zero weight, so a branched network plays exactly like S at first; message heads start uniform.
3. **Optional from-scratch control:** at least one F seed trained from scratch with the same budget.

**Communication diagnostics.**
- **Toy check:** as in RLupus [5], fix evil to the rule player and train only good players to talk in M. If Merlin's judgments still do not raise good's win rate, look for an implementation error first.
- **Sender side:** mutual information between slot values and the speaker's true role, or the judged player's true team, by role.
- **Listener side:** in a frozen checkpoint, replace one slot of one message and measure the change in listeners' vote and proposal distributions (causal influence) [8].
- **Gate 2:** if listener influence in M is near zero, reduce the open slots or their value ranges before full training.

## 7 Evaluation and Metrics

**Frozen evaluation.**
- Each seed plays 1,000 games per condition.
- Core conditions per seed:
  - S → S;
  - M → M, M → S, and M without ③;
  - F → F, F → S, and four F → F−k (without claim, rationale, team or intent).
- All conditions share the role assignments and first leaders, and policy sampling uses a separate stream fixed by the evaluation seed, so results are exactly reproducible. There are no training updates and no memory across games.

**RQ1 Game balance.**
- **Primary metric:** ΔGood = P(good wins | F → F) − P(good wins | S → S). Positive means communication favors good in this setting; adjacent differences on the ladder are reported too. 50% is only a symmetry reference, not a target for an asymmetric game.
- **Cross-seed evaluation.** Self-play learns conventions that only work inside one training population. In S, good wins 0.39 against its own seed's evil but 0.54 against other seeds' evil. Every configuration therefore reports a 3 × 3 matrix of good *i* against evil *j* (1,000 paired games per cell), and ΔGood is given on same-seed games (the diagonal) and cross-seed games (off the diagonal). Seeds are crossed only in evaluation, never in training.
- **Decomposition:** P(good wins) = P(three successes) × [1 − P(Assassin hits Merlin | three successes)]. Evil wins by three failures and by five rejections are reported separately. Assassination accuracy is computed only over games that reach the assassination; a random guess among the four good players hits 25%. Mission success is reported by mission number, and missions that never happen are not counted as failures.

**RQ2 Strategic behavior.**

| Behavior | Measure | Caveat |
|---|---|---|
| Team building and voting | Share of good players on proposed and approved teams; approval rates by role; approval of teams with evil players | Stratify by mission and rejection count; approving a last-chance team is not necessarily a mistake |
| Sabotage | Fail rate of evil players on missions; with both evil players on a team, the distribution of 0, 1 and 2 Fails | Letting a mission pass may be camouflage |
| Merlin concealment | Log loss of an identity probe trained on public history only | A probe failing does not mean the Assassin is fooled |
| Percival's dilemma | Percival's team and vote preferences between the two candidates; how Morgana shifts them | Support is not the same as being taken for Merlin |
| Assassination | Accuracy given three successes; sensitivity to vote and proposal history | Games that reach the assassination are a selected sample |

**RQ3 Implicit signaling.** S is the main object: without an explicit channel, public actions are the only way to pass information. S already shows conventions that work only within one seed; for example, an Assassin that targets a seat or the last leader is near chance against other seeds' Merlins. Evidence comes in three levels, and conclusions go only as far as the level reached:
1. **Informativeness:** adding individual vote histories improves the prediction of hidden roles. This shows only that behavior carries information.
2. **Receiver use:** hiding who cast past votes (counts, outcomes and legal actions kept) changes receivers' actions and payoffs, against a placebo mask of the same size and an auxiliary condition trained with masked inputs. This shows the cue is used.
3. **Strategic signaling** (exploratory): among vote combinations with the same outcome, informed players choose the ones that influence teammates and adapt to their partners.

The first two levels are the minimum. Once the explicit channel exists, the informativeness of votes in M and F is compared with S to judge whether implicit signals are complemented or crowded out. Correlations or task scores alone can mislead about whether communication exists [8].

**RQ4 Message components.**

| Analysis | Measure | Caveat |
|---|---|---|
| Slot use | Share of "say nothing" per slot over training, by role and team | Frequent use does not mean usefulness |
| Truthfulness | Claims matching true roles; share of accused players who are evil; whether Merlin hides and evil players impersonate | Meaning comes from behavior, not slot names |
| Rationale | Effect of record, role-knowledge and no-basis rationales on listeners' votes | Only measurable by switching ③ off at test time |
| Listener influence | Change in listeners' vote and proposal distributions when one slot is replaced | Large influence is not necessarily good for the speaker's team |
| Win contribution | Increment from adding a slot and dependence from removing it, by team | Disagreement means the slots interact |

**Statistics.** Per-seed metrics come first. Paired differences, effect sizes and 95% intervals then come from a stratified bootstrap over training seeds, with events grouped by game. Wilson intervals for a single model's win rate are given as well, but they do not replace between-seed uncertainty. The good win rate is the pre-specified primary metric. Identity probes split train and test sets by whole games and seeds.

**Optional diagnostic: approximate exploitability.** Freeze one team and train a 100,000-game PPO best response against it, starting from the checkpoint's own network for that team. Exploitability = BR-good win rate + BR-evil win rate − 1 [12]. It is a diagnostic only, never a training target or a gate.

**Logging.** `evaluate.py --save-games` writes every self-play game as one JSON line: roles by seat, proposals with leaders, teams and approvals, mission cards and results, the assassination target, the winner and reason, the open slots, and every message in speaking order. The evaluation JSON names the checkpoint, seed and silenced slots. Each training run saves its configuration, metrics per iteration, and the code version and host it ran on.

## 8 Scope of Conclusions

- Conclusions hold only for the defined slots, value ranges, roles and learned policies. Structured messages are not free discussion; claims about natural language need a separate natural-language experiment.
- Finite-budget self-play is not a Nash equilibrium, and this project measures a learning system without trying to solve for one. Teammates who cannot share private information make the equilibrium harder to find than in two-player zero-sum games [11]. A control trained with PPO plus magnetic mirror descent reached approximate exploitability [12] of 0.60 at 1.5M games, against 0.65 for plain self-play, which is no meaningful move toward equilibrium. Win rates inside one training population describe only that population, hence the cross-seed evaluation.
- If communication is not learned, that is reported as a learning failure, not as "talk has no value". Small win-rate differences are reported with intervals and the effect sizes they rule out. If RQ3 reaches only correlation, it stays exploratory.

## 9 Optional Extension: Natural Language

- **Language layer:** templates or an LLM render five-slot messages as text, or parse text back into slots; this layer needs no Avalon strategy [3].
- **Experiment:** frozen LLMs play all six seats with isolated contexts, speaking JSON with the open slots plus at most 400 tokens of text. NL-S, NL-M and NL-F mirror S, M and F: a 20-game pilot, then 100–300 games per condition. Invalid outputs follow fixed retry and fallback rules, and their frequency is reported. Results are reported separately.

## Repository Structure

```
.
├── src/                     The code, imported as `src` (scripts and tests run from the repository root)
│   ├── env/                 Rules, game state machine, action space, information-state encoding
│   ├── agents/              Actor–critic policy, random and rule-based players
│   ├── comm/                Five-slot message protocol: slots, consistency masks, observation features
│   ├── training/            Trajectory recording and GAE, PPO update, opponent pool, trainer, best response
│   ├── evaluation/          Paired frozen evaluation, win-rate decomposition, message statistics
│   ├── runner.py            Plays many games in lockstep with batched decisions
│   └── config.py            YAML-backed training configuration
├── configs/                 baseline_silent.yaml (S), comm_M, comm_M_claim, comm_M_team, comm_M_intent, comm_F, pilot_silent.yaml
├── scripts/                 train.py, evaluate.py, crossplay.py, analyze_messages.py, best_response.py, exploitability.py, plot_training.py, benchmark.py
├── tests/                   Rules, information leakage, observation encoding, GAE, message protocol, message statistics, short training runs
├── results/                 Curated figures and tables
├── references/              Paper PDFs (local only, not tracked)
└── runs/                    Training outputs (local only, not tracked)
```

## Quick Start

Python 3.10 or newer and pip 25.1 or newer; the environment is named after the project on every machine. Dependencies are listed in `pyproject.toml`. On a CUDA machine, install the matching PyTorch build first.

```bash
conda create -y -n avalon-signals --override-channels -c conda-forge python=3.12
conda activate avalon-signals
pip install --group default
pytest                                                     # rules, leakage, protocol, GAE and short training runs
python scripts/benchmark.py --games 2048                   # simulation and update throughput on this machine
python scripts/train.py --config configs/pilot_silent.yaml # 2,048-game pilot
python scripts/train.py --config configs/baseline_silent.yaml --seed 0 1 2
python scripts/train.py --config configs/comm_F.yaml --seed 0 1 2 --device cuda   # branches from S checkpoint_01460
python scripts/evaluate.py --checkpoint runs/comm_F/seed0/latest.pt --games 1000
python scripts/evaluate.py --checkpoint runs/comm_F/seed0/latest.pt --silence all        # F -> S
python scripts/evaluate.py --checkpoint runs/comm_F/seed0/latest.pt --silence rationale  # F -> F without (3)
python scripts/crossplay.py runs/comm_F/seed{0,1,2}/latest.pt                            # 3 x 3 cross-seed matrix
python scripts/evaluate.py --checkpoint runs/comm_F/seed0/latest.pt --save-games runs/comm_F/seed0/games.jsonl
python scripts/analyze_messages.py runs/comm_F/seed*/games.jsonl                        # message statistics (RQ4)
python scripts/best_response.py --checkpoint runs/baseline_silent/seed0/latest.pt --side evil   # optional diagnostic
```

- **Outputs:** each run writes `config.yaml`, `metrics.jsonl` (one JSON object per iteration), `run_info.jsonl` and checkpoints (every 20 iterations) to `runs/<name>/seed<k>/`.
- **Changing settings:** override any config key with `--set`, for example `--set ppo.lr=1e-4`.
- **Several seeds:** `--seed 0 1 2` trains the seeds in parallel processes; each writes its console output to `runs/<name>/seed<k>/train.log`. `--threads` sets the torch CPU threads per run (default with several seeds: an even share of the cores).
- **Continuing a run:** `--resume` continues from the run's `latest.pt`, for example after a crash or to extend the budget.
- **Starting from saved weights:** `--init-from` loads network weights only, for warm starts and the branch training of the communication configurations.

**Reference lines.** These come from 3,000 paired games with the default scripted players; trained policies should beat them.

| Matchup (good vs evil) | Good win rate |
|---|---|
| random vs random | ≈ 0.26 |
| rule vs rule | ≈ 0.14 |
| rule vs random | ≈ 0.65 |
| random vs rule | ≈ 0.01 |

## Compute

The networks are small and rollouts run on the CPU, so throughput comes mostly from running several seeds in parallel; the GPU only speeds up the PPO update.
- **Silent configurations:** on an Apple M5 CPU, self-play collects about 500 games/s. S trains with the update on the CPU.
- **Communication configurations:** these add up to 32 speaking decisions per game and a 4,729-dimensional message block to every observation, so their PPO update runs on the GPU. On a laptop with a Ryzen 9 8945HX and an RTX 5070 Ti:
  - six runs in parallel (M and F × 3 seeds) took about 15–16 s per iteration including evaluation and checkpoints, so a 1,470-iteration branch run takes about 6.5 h;
  - with the update on the CPU instead, an iteration took about 26 s;
  - rollouts are the bottleneck: GPU utilization stays near 16% and peak memory near 4 GB.
- **GPU stability:** with NVIDIA driver 581.42 that GPU faulted when several runs updated on it at once. With driver 621.01 six parallel runs had no faults. A run that crashes continues from its latest checkpoint with `--resume`.

## Key References

1. Serrino et al. [Finding Friend and Foe in Multi-Agent Games](https://arxiv.org/abs/1906.02330). NeurIPS 2019.
2. Nakamura. [Is it Percival time yet? A preliminary analysis of Avalon gameplay and strategy](https://sigbovik.org/2017/proceedings.pdf). SIGBOVIK 2017.
3. Light et al. [Strategist: Self-improvement of LLM Decision Making via Bi-Level Tree Search](https://openreview.net/forum?id=gfI9v7AbFg). ICLR 2025.
4. Lan et al. [LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay](https://aclanthology.org/2024.emnlp-main.7/). EMNLP 2024.
5. Brandizzi et al. [RLupus: Cooperation through Emergent Communication in The Werewolf Social Deduction Game](https://arxiv.org/abs/2106.05018). Intelligenza Artificiale, 2021.
6. Foerster et al. [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676). NeurIPS 2016.
7. Cao et al. [Emergent Communication through Negotiation](https://arxiv.org/abs/1804.03980). ICLR 2018.
8. Lowe et al. [On the Pitfalls of Measuring Emergent Communication](https://arxiv.org/abs/1903.05168). AAMAS 2019.
9. Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). 2017.
10. Yu et al. [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955). NeurIPS 2022.
11. Carminati et al. [Hidden-Role Games: Equilibrium Concepts and Computation](https://arxiv.org/abs/2308.16017). arXiv 2308.16017v5, 2026.
12. Lanctot et al. [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832). NeurIPS 2017.

These are the only works the project cites. Paper PDFs are kept locally and are not redistributed here.
