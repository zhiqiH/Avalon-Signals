# Structured Communication in Six-Player Avalon

Graduate term project for **CS 7329 Reinforcement Learning** at Southern Methodist University (Fall 2026).

What do agents gain from talking in a hidden-role game? This project trains multi-agent reinforcement learning agents to play six-player *The Resistance: Avalon*. They first play **in silence**, using only public actions such as team proposals, votes and mission results. Then they play with a **structured message channel** in which a statement needs a basis on the public record and binds what its speaker may do next. The talk is built so that a person at the table can read it and hold the speaker to it. The project measures what such talk changes: who wins, and what it gives away.

This README is the design of the project and its state on 2026-11-24 (Section 8). It will be replaced by a results write-up once all experiments are finished.

## 1 Research Questions

| | Question | Main evidence |
|---|---|---|
| **RQ1** Game balance | Relative to the silent baseline, which team gains when the players can talk? | Good win rate within and across training seeds; probability of three successful missions; assassination accuracy given three successes |
| **RQ2** Implicit signaling | In silent play, do public actions carry hidden information that other players use? | Cross-seed play and a partner who votes and proposes "with a foreign accent"; the question is whether such a channel exists, not how to decode it |
| **RQ3** Concealment | What does talk give away about Merlin, and how do the agents learn to hide it? | Assassination accuracy by who speaks for each seat; which kinds of statement let an observer pick Merlin; how often Merlin says what he knows to be false |

An earlier plan asked which message slot accounts for the effect of communication, with one training configuration per slot. It was dropped: the slots do not work apart from each other once words bind deeds, and the project's aim moved from splitting "communication" into parts to getting one talking agent whose speech makes sense at a table.

**Why reinforcement learning.** Each player sees only part of the state and must infer identities from the game history while making sequential decisions. Reward arrives only at the end of a game, and good ways to propose, vote and speak are not known in advance. Self-play policy gradient methods can learn such policies, and silent and talking play can then be compared under controlled conditions.

**Why silence is the baseline.** Silent play has fewer decisions and no cheap-talk problem of messages being ignored, so it trains into a stable baseline that the talking agents start from. The question is therefore what communication *adds*, without assuming a direction. Talk may help good players coordinate, but it may also help evil players deceive or find Merlin.

**Readable talk is a requirement of the design, not a research question.** That statements rest on the public record and that speakers act on them is enforced by the protocol and checked before any result is read (Section 7, acceptance checks); so is play next to partners with plain human logic. Win rate is an observation, not the goal: a higher win rate reached through talk that only an agent's own training partners understand would count against it.

## 2 Related Work

| Work | Approach | Relation to this project |
|---|---|---|
| DeepRole [1] | Five-player Avalon; CFR with deductive reasoning and a value network; no language | Precedent for strong silent play, which supports silence as the baseline |
| Is it Percival time yet? [2] | Descriptive statistics of 61 human games | Source of metrics: Percival claims, the first failed mission, evil coordination failures; not causal |
| Strategist [3] | LLM with tree search | Its appendix has six-player human games with and without discussion, but without Percival and Morgana and without controlled training |
| Lan et al. [4] | LLM agents in six-player Avalon with Percival and Morgana | Behavior categories (camouflage, sharing, persuasion) for RQ3 |
| RLupus [5] | Werewolf; PPO-trained discrete signals for villagers | Closest precedent: villager win rate rose from 0.044 without communication to at most 0.45, and more bandwidth did not always help |
| RIAL [6], Cao et al. [7] | Messages trained as RL actions; self-interested agents struggle to use cheap talk | How messages are trained, and the main risk |
| Lowe et al. [8] | Measuring emergent communication | Separating "a message carries information" from "a listener uses it" |
| PPO [9], MAPPO [10] | Clipped policy gradient; practical multi-agent PPO | Training algorithm and starting hyperparameters |
| Hidden-role games [11], PSRO [12] | Equilibrium concepts for hidden-role games; best responses as a measure of distance from equilibrium | Why this project measures a learning system instead of targeting an equilibrium |

**Intended contributions.**
- A message protocol for a hidden-role game in which meaning is fixed by rules instead of being learned: a judgment needs a record that points its way, its justification is derived, and what a player says limits how it may propose and vote.
- A training recipe that keeps learned talk readable: a hand-written rule teacher is distilled into the speaker, listeners learn before speakers do, and two anchors hold the meaning of talk in place while reinforcement learning chooses what to say.
- Measurements of what such talk does: win rates decomposed into the mission path and the assassination path, what statements expose about Merlin, and play with logic partners and humans.

## 3 Game Setup

**Roles.** Two Loyal Servants, Merlin, Percival, Morgana and the Assassin. Good is Merlin, Percival and the two Servants; evil is Morgana and the Assassin. The rules follow the game description in the Strategist appendix [3].

| Item | Rule |
|---|---|
| Missions and votes | Mission sizes 2-3-4-3-4. A team needs at least four approvals; a tie rejects. Five rejected proposals in one mission give the win to evil. |
| Vote guard for good players | Two votes that can only lose are not legal for a good player. On the fifth proposal of a mission it must approve. With two missions failed it must reject a team that, by its own knowledge, certainly holds an evil player. Evil players vote freely. |
| Missions | Good players must play Success; evil players choose Success or Fail. One Fail fails the mission; only the number of Fail cards is revealed. |
| End of game | Three failed missions: evil wins. Three successful missions: the Assassin names one player, and evil wins if that player is Merlin. |
| Private knowledge | Merlin sees both evil players. Percival sees Merlin and Morgana as two candidates without knowing which is which. The evil players know each other. Servants know nothing extra. |
| Randomization | Roles and the first leader are random in every game; no Lady of the Lake or other add-ons. |

Players see only the public history and the private knowledge their role allows. True roles are used only by the referee and for offline analysis.

## 4 Communication Design

The talking agents are called **Comm**, the silent ones **Silent**. Comm's protocol is set in `configs/comm.yaml`; the full rules are in `src/comm/protocol.py`.

**Timing** follows common tabletop play.
- Every mission opens with one round of public discussion before its first proposal. Players speak once each in seat order, starting left of the leader, and the leader speaks last. Then the leader proposes and everyone votes.
- After a rejected proposal, leadership passes on and the next leader proposes at once, without discussion.
- After the third successful mission, Morgana and then the Assassin each speak once to agree on who Merlin is, before the assassination.
- There is no free dialogue, but a speaker can answer the players who spoke before it in the same round (slot ⑥).

**Seven slots.** Every slot can say nothing. Masks enforce the rules below.

| Slot | Content | Values | Rules |
|---|---|---|---|
| ① Role claim | Who I claim to be | none, Servant, Percival | Nobody claims Merlin or an evil role; claims may be false; a claim stands for the rest of the game |
| ② Judgment | What I think each other player is | per player: none, good, evil | **A judgment needs a basis:** a player can be judged good (or evil) only if some record points that way. Before the first mission is finished nothing is on record, so nobody can judge anybody. At most two players judged evil and four judged good |
| ③ Rationale | Why I judge so | per judged player: a mission, vote or proposal record with its mission number | Derived, not chosen: the strongest record that points the judgment's way |
| ④ Team suggestion | Who should go on the mission | none, or a team of the current size | Optional. The team holds the speaker and nobody the speaker accuses, and it must be among the cleanest teams by the public record that are open to the speaker. The leader makes no suggestion: its proposal follows |
| ⑤ Action intent | How I will vote this round | per player: none; **veto** (I reject any team with them); **require** (I reject any team without them) | No veto on a player judged good, no require with an accused player; some team must satisfy all conditions |
| ⑥ Reply | What I say to an earlier speaker of this round | per earlier speaker: none; **endorse** (I agree, and I approve the team you suggested); **dispute** (I disagree) | Endorsing needs a suggested team and no conflict with one's own judgments. A dispute needs a basis, which is derived: the other player accused me, we judged somebody in opposite ways, or their team holds a player I accuse |
| ⑦ Conditional promise | What I will say after this round's mission | per player: none; **fail-evil** (if they go and the mission fails, I will call them evil); **success-good** (if they go and it succeeds, I will call them good) | Closed where no discussion can follow the outcome; at most two fail-evil and three success-good promises per message |

How a record points:
- **mission record:** the player was on mission *q*, which failed (evil) or succeeded (good);
- **vote record:** the player approved mission *q*'s team that then failed, or rejected one that then succeeded (evil); the reverse points to good. A forced fifth proposal leaves no vote record;
- **proposal record:** the player led mission *q*'s team, which failed (evil) or succeeded (good).

In the evil talk before the assassination the two evil players judge good players as Servant, Merlin or Percival; the Assassin names exactly one player as Merlin and then has to strike that player.

**Words bind deeds.** What a player says limits what it may do, for both teams, through the action masks:

| Said | Bound deed |
|---|---|
| Accused a player | Does not propose them, and rejects any team that holds them, until it calls them good or names a new list of suspects |
| Announced an intent | Proposes and approves only teams that meet it in this round |
| Suggested a team | Approves exactly that team |
| Endorsed a speaker | Approves that speaker's team; a leader who endorses proposes it |
| Disputed a speaker | Rejects that speaker's team and does not propose it |
| Judged all other members of a team good | Approves that team |
| Made a promise | Makes that judgment in the next round if the condition holds |
| Named Merlin (Assassin) | Strikes that player |

Voters are free on the forced fifth proposal of a mission, and the good players' vote guard (Section 3) comes first. A leader approves its own proposal and proposes one of the cleanest teams that its words leave it.

**No unfounded change of mind.** If a player's most recent earlier judgment of someone pointed the other way, a judgment in the new direction needs a record from a mission finished since then.

**Design principles.**
- **Meaning is set by rules, choice is left to learning.** A sentence means what it obliges its speaker to do, and a judgment can only be made where the record supports it, with the justification derived by a fixed rule. So a statement cannot be turned into a private code. What reinforcement learning decides is what a human player decides: whether to speak, about whom, which team to back, and how much to hide.
- **Lies are allowed, contradictions are not.** The referee never checks truth. Evil players accuse good ones and vouch for each other, and Merlin may accuse a player he knows to be good to cover himself; how much he does so is learned. One message cannot contradict itself, and deeds cannot contradict words.
- **Nobody has to speak.** Every slot may stay empty, team suggestions included. A team that says less also gives less away.
- **Human seats are free.** A human at the table is bound by none of this: no rule limits what the human says or does. Deeds against one's own words are counted in public for such a seat (a mark every player sees) instead of being prevented.
- **Scope.** A message decides *what* is said. It covers the main content of about two minutes of table talk, but not wording, tone or long arguments.

**How the protocol got here.** It was reached in steps: five slots whose words bound nothing, then five slots with words that bind deeds, then the seven slots above, with the team suggestion made optional and without a rule that forbade Merlin to accuse a player he knew to be good (with promises, that rule let the evil team rule out as Merlin whoever promised to accuse a good player). The earlier versions, their configurations and their results are in the git history (tag `f3-2026-11-24`).

## 5 Experimental Design

**Configurations.**

| Config | Slots | Words and deeds | Purpose |
|---|---|---|---|
| Silent (`configs/baseline_silent.yaml`) | none | | Silent baseline: self-play PPO, frozen at 10 million games per seed |
| Comm (`configs/comm.yaml`) | ①–⑦, every slot optional | bound | The talking agents, trained from the frozen baseline |

**Evaluations of a trained talking agent** (frozen policies, no learning, 3 seeds).

| Evaluation | What is played | Question |
|---|---|---|
| Same-seed self-play | Good and evil of one seed | Balance inside one training population (RQ1) |
| Cross-seed play | Good of one seed against evil of another | Balance without conventions shared by training partners (RQ1) |
| Silence at test time | The same policies with every slot forced to say nothing | How much the trained policy depends on the channel |
| Word and deed check | Whole games, every statement compared with the deeds that follow | Acceptance check: talk is consistent with action |
| Private-code checks | Whether the form of a message tells the speaker's team; distance from the rule teacher | Acceptance check: talk stays readable |
| Leak probes | Which statements let an observer pick Merlin; assassination accuracy by who speaks for each seat | What talk exposes (RQ3) |
| Logic partners | One to three good seats played by a deduction player with the rule teacher's speech | Acceptance check: play next to a partner with plain human logic |
| Human test | One human and five agents in the browser, the human unbound | Acceptance check: play with people |
| Silent play with a foreign partner | Silent with one good seat from another seed | Whether public actions carry a private signal (RQ2) |

- **Controls:** Silent and Comm share the game rules, network size, algorithm and hyperparameters. The training seed is the unit of replication, with 3 seeds. Each seed's Comm network starts from the same seed's Silent network, with the new parts at zero weight, so it starts by playing exactly like the baseline.
- **No test-time opening of the channel:** a silent policy has never seen messages, so letting it talk at test time is meaningless.
- **No slot ablations:** see Section 1.

## 6 Training

**Environment.**
- A custom six-player state machine (six players and structured talk both need custom rules). Phase order: roles → discussion (Comm only) → proposal → vote → mission or next leader → three successes or three failures → evil talk (Comm only) → assassination → result.
- One flat 30-way action space for all phases, in seat-relative coordinates (0 is the acting player), so one network serves every seat:
  - 20 proposal actions, enumerating every legal team of the current size (C(6,2) = 15, C(6,3) = 20, C(6,4) = 15);
  - then approve/reject, Success/Fail (only evil players may play Fail), and the assassination target (the four players other than the two evil ones).
- Votes and mission cards are collected first and revealed together, so execution order leaks nothing.
- Each player's observation holds only the public history, its own role and its role's private knowledge. Percival's two candidates are an unordered set. Seats, roles and the first leader are random. The full state is kept apart from policy inputs and used only by the referee and for analysis.

**Observation.** The silent observation is a 669-dimensional information-state vector:
- the player's role and private knowledge;
- the phase, mission, attempt, leader and current team;
- the complete proposal and vote history (votes as +1 approve, −1 reject, 0 not cast; at most 5 missions × 5 proposals);
- mission results and the player's own mission cards;
- deduction fields (54): which of the 15 possible evil pairs are still consistent with the mission results (a mission with *f* Fail cards had at least *f* evil members), first from public information and then also from the player's private knowledge, plus the seats that are therefore certainly evil or certainly good in each view.

Every seat-indexed field is rotated into the acting player's frame, and a named index (`LAYOUT`) locates each field. Because the history is bounded, this vector is a complete information state, so plain MLPs suffice: no recurrence, no BPTT, and RQ2 interventions can edit the vector directly (the deduction fields must then be recomputed from the edited mission history).

**Why the deduction fields.** They add no information; they do the logical bookkeeping that DeepRole [1] does explicitly. Without them PPO does not learn this bookkeeping: states with a logically certain evil player are rare (about 5% of good votes) and a single vote decides the outcome in only about 16% of them, so the terminal reward gives little signal for them. Since Comm starts from Silent, a baseline without this bookkeeping would let communication look useful merely for passing on deductions.

Comm adds a message block (5,919 dimensions): 123 global features (15 for the state of the discussion and 108 for the public counts of deeds against words, three thresholds per count and player) plus 36 fixed message slots of 161 features each (one per mission and speaker, and one per speaker of the evil talk), rotated into the observer's frame.

**Network.**
- Independent PPO with one actor–critic per team, shared by all seats of that team; the role is an input. The actor and the critic are separate MLPs (two layers of 256 units).
- One 30-way action head with legal-action masking: illegal logits are set to a large negative value, and entropy is computed over legal actions only.
- Comm adds message heads in the speaker's frame (37 heads, 180 logits):
  - claim (4 values);
  - a judgment per other player (8 values);
  - a rationale type (5 values) and a mission number (5 values) per other player;
  - a team (up to 21 values, shared with the proposal encoding);
  - an intent per other player (3 values);
  - a reply (3 values) and its basis (4 values) per other player;
  - a promise per other player (3 values).
- Sampling runs in two stages. The claim and judgments come first. Rationales, team and intents follow, with their logits conditioned on the first stage through a zero-initialized linear map, and consistency masks applied throughout. A rationale head has exactly one legal value (the derived one), so it adds nothing to the log-probability.
- The log-probability of a message is the sum over the heads used, and it enters the same PPO ratio as game actions. Derived heads (rationale, reply basis) have exactly one legal value and carry no choice.
- Messages are trained RIAL-style [6] with the game reward. DIAL is not used, because listeners include opponents. The critic sees only the player's own information.

**Reward and update.**
- Win +1, loss −1, no intermediate reward; γ = 1 with GAE (λ = 0.95) [9]. There is no reward shaping for honesty, signaling or concealment.
- The two teams' losses are summed separately, so the 4-vs-2 team sizes never weight each other.
- Switched-off slots, forced silence and other forced actions do not enter the actor loss.
- Hyperparameters are shared by Silent and Comm and were set from the PPO and MAPPO defaults [9, 10]: learning rate 3e-4, 4 epochs, minibatch 2,048, clip 0.2, value coefficient 0.5, entropy coefficient 0.01, gradient norm 0.5, normalized advantages, 512 games per iteration.

**Opponents.** In each training game, both teams use the current policies (50%), or one team is replaced by a past snapshot (25%) or by the rule-based player (25%). A snapshot is added every 10 iterations, and the pool keeps 20 per team. Only current policies learn. Rule players say nothing and ignore messages, so they are a fixed reference.

**Training pipeline.**
1. **Silent baseline.** Train Silent until its win rates against rule players are stable (gate 1: the mean over the last ~100k games differs from the previous ~100k by less than 0.02). Comm is not trained until Silent passes.
2. **Frozen baseline.** Silent is frozen at 10,000,384 games (19,532 iterations) per seed: `baseline/seed{0,1,2}/checkpoint_19532.pt`, tracked in git with the run's configuration, metrics and run info. Its win rates against rule players are flat over the last several million games, and its evaluation is in `results/tables/baseline_silent_final.json`. Remaining weaknesses were measured by replacing one behavior at test time:
   - masking the good players' residual deduction errors moves the good win rate by less than 1 point;
   - the learned Assassin hits Merlin 0.32 of the time in same-seed games, against 0.36 for a vote-reading heuristic, so the heuristic would cost good about 3 points. Across seeds the heuristic falls to 0.18, below chance, because Merlin learned decoy votes against vote readers;
   - an evil team that never approves a clean team at good's match point would cost good 2–4 points.
3. **Branching.** Comm adds its message inputs and heads to each seed's frozen Silent network. New inputs and heads start at zero weight, so the branched network plays exactly like the baseline at first. The actor changes only through low-rank updates (LoRA, rank 8); the new message parts and the critic train fully.
4. **Staged communication training** (`scripts/run_comm_pipeline.sh`), three stages:
   1. **Distillation** (4,000 games per seed, minutes). A hand-written rule teacher (`src/agents/teacher.py`) speaks; its messages are fitted into the message heads by supervised learning. The network now talks like the teacher, plays as before and does not listen yet. Reinforcement learning from a blank speaker ended in talk that only training partners understood.
   2. **Listening** (153,600 games). The speaker is frozen and only listening and acting are trained. With fixed talk of known meaning, listeners have a stable target. If both sides changed from the start, listeners would first learn to ignore talk and speakers would then have no reason to mean anything.
   3. **Anchored talk** (512,000 games, and more in the same way if wanted). Speakers learn too, without an entropy bonus on messages. Two anchors hold the meaning of talk in place: a KL penalty towards the distilled speaker, whose weight falls from 0.1 to 0.03 and stays there, and the rule teacher itself speaking for 40% of the seats, so that listeners keep hearing talk with the teacher's meaning and no code can rely on who is speaking.
   In stages 2 and 3 a quarter of the games put a logic partner (`src/agents/logic_agent.py`), a player that deduces from the record as a person would, in one or two good seats.
5. **What the rule teacher is.** It gives every speaker "a good player's view" of the table (public suspicion plus what the mission results make certain) and speaks from it: accuse on about two records against a player, vouch on two records for, suggest the cleanest team, endorse a team close to one's own best, promise what one would then say anyway. Evil players speak from the same view except about their teammate, so the teacher favors neither team. It speaks for Merlin and Percival as for a Servant, by the public record alone: an earlier teacher let Percival lean towards his two candidates and Merlin use what he knows, and the learned Assassin came to read both (Section 8).
6. **How the networks of Section 8 were trained.** Not in one pass of the pipeline, but in three steps that followed the protocol's history (Section 4), 1,843,200 games per seed on top of the baseline:
   - a pass with the first five slots (153,600 games of listening, 512,000 of anchored talk);
   - a second pass from those networks with all seven slots (again 153,600 and 512,000 games), the parts for replies and promises starting at zero weight;
   - 512,000 more games of anchored talk after the teacher stopped showing roles, with the speaker anchor distilled again from that teacher and the KL weight at 0.03.

   One pass from the baseline with `configs/comm.yaml` is the recipe for a new run; it has not been run in that form.

**Communication diagnostics** (`scripts/comm_diagnostics.py`, `scripts/dialect_probe.py`, `scripts/trace_game.py`).
- **Talk, babble, silence:** the same frozen policies with learned talk, with random legal messages, and with every slot silenced.
- **Private code:** how well a probe tells the speaker's team from the form of its messages alone.
- **Transcripts:** whole games with roles, statements and their basis, proposals, votes, and every deed checked against the speaker's words.

## 7 Evaluation and Metrics

**Frozen evaluation.**
- Each seed plays 2,000 games per condition (1,000 per pair in cross-seed play).
- Conditions per seed: Silent; Comm with talk, with every slot silenced, and with one to three logic partners.
- All conditions share the role assignments and first leaders, and policy sampling uses a separate stream fixed by the evaluation seed, so results are exactly reproducible. There are no training updates and no memory across games.

**RQ1 Game balance.**
- **Primary metric:** ΔGood = P(good wins | Comm) − P(good wins | Silent). Positive means communication favors good in this setting. 50% is only a symmetry reference, not a target for an asymmetric game.
- **Cross-seed evaluation.** Self-play learns conventions that only work inside one training population. In Silent, good wins 0.50 against its own seed's evil but 0.67 against other seeds' evil (0.46 and 0.65 without the vote guard, which Silent was trained without; `results/tables/baseline_silent_guard.json`). Both configurations therefore report a 3 × 3 matrix of good *i* against evil *j* (1,000 paired games per cell), and ΔGood is given on same-seed games (the diagonal) and cross-seed games (off the diagonal). Seeds are crossed only in evaluation, never in training.
- **Decomposition:** P(good wins) = P(three successes) × [1 − P(Assassin hits Merlin | three successes)]. Evil wins by three failures and by five rejections are reported separately. Assassination accuracy is computed only over games that reach the assassination; a random guess among the four good players hits 25%. Mission success is reported by mission number, and missions that never happen are not counted as failures.

**RQ2 Implicit signaling.** Silent is the object: without an explicit channel, public actions are the only way to pass information. The evidence is the difference between same-seed and cross-seed play (good wins 0.50 against its own seed's evil and 0.67 against another seed's), and a test in which one good Servant comes from another seed: if its teammates treat a good player who votes and proposes "with a foreign accent" like an evil one, their reading of public actions is a convention of the training population. The question is whether such a channel exists; decoding it is out of scope.

**RQ3 Concealment.**

| Analysis | Measure | Caveat |
|---|---|---|
| Assassination | Accuracy given three successes, with talk and with silence | Games that reach the assassination are a selected sample |
| Which statements expose Merlin | A probe that picks Merlin among the four good players from counts of each kind of statement, with groups of statements left out | The Assassin also sees deeds, so a probe on statements is not the Assassin's view |
| Who speaks | Assassination accuracy when the rule teacher speaks for some seats (Merlin, Percival, Servants) instead of the learned speaker | Shows which habits of plain speech give Merlin away |
| Bluffing | How often Merlin accuses a player he knows to be good, or promises to | Learned, not prescribed |

**Acceptance checks.** These are requirements a talking agent has to meet before its results are read, not research questions.

| Check | Measure | Caveat |
|---|---|---|
| Word and deed | Per game and role: deeds against one's own words (none are possible for bound seats; this checks the implementation and counts what free seats do) | Consistency is necessary for sense, not sufficient |
| Whether talk is right | Share of accused players who are evil and of vouched players who are good, by role | Good players are allowed to be wrong, and Merlin to lie |
| Distance from the teacher | KL divergence of the learned speaker from the distilled one over training; use of each slot by team | Moving away is expected where the agents learn to hide; the question is what changes |
| Private code | Information about the speaker's team in the form of a message; cross-seed against same-seed win rates | A probe failing does not prove there is no code |
| Transcripts | Whole games read by a person | Small sample |

**Play with people** (acceptance check). Win rates and transcripts with one to three logic partners in good seats, and the human test: one human and five agents, the human's seat free (Section 4), games recorded only with consent. A language model is to translate between sentences and messages in both directions (Section 9); until then the table uses fixed sentence templates and a rule-based reader.

**Statistics.** Per-seed metrics come first. Paired differences, effect sizes and 95% intervals then come from a stratified bootstrap over training seeds, with events grouped by game. Wilson intervals for a single model's win rate are given as well, but they do not replace between-seed uncertainty. The good win rate is the pre-specified primary metric. Identity probes split train and test sets by whole games and seeds.

**Optional diagnostic: approximate exploitability.** Freeze one team and train a 100,000-game PPO best response against it, starting from the checkpoint's own network for that team. Exploitability = BR-good win rate + BR-evil win rate − 1 [12]. It is a diagnostic only, never a training target or a gate.

**Logging.** `evaluate.py --save-games` writes every self-play game as one JSON line: roles by seat, proposals with leaders, teams and approvals, mission cards and results, the assassination target, the winner and reason, the open slots, and every message in speaking order. The evaluation JSON names the checkpoint, seed and silenced slots. Each training run saves its configuration, metrics per iteration, and the code version and host it ran on.

## 8 State on 2026-11-24 and Scope

**Results so far** (3 seeds, 2,000 games per seed and condition; tables in `results/tables/comm/`, figures in `results/figures/`).

| | Silent | Comm (seeds 0 / 1 / 2) |
|---|---|---|
| Good win, same seed | 0.50 (mean) | 0.392 / 0.400 / 0.354 |
| Good win, cross-seed (mean) | 0.67 | 0.555 |
| Three successful missions | | 0.574 / 0.597 / 0.553 |
| Assassin hits Merlin after three successes | 0.32 | 0.318 / 0.329 / 0.360 |
| Good win with every slot silenced at test time | | 0.538 / 0.578 / 0.417 |
| Good win with one logic partner | | 0.312 / 0.329 / 0.247 |
| Deeds against one's own words | | none |

- **Balance (RQ1).** With Comm, good wins less often than in Silent, in same-seed and in cross-seed play. The same networks with every slot silenced win more often as good than with talk.
- **Merlin covers himself (RQ3).** 71–76% of his accusations are right, and from the statements alone an observer picks him among the four good players 0.39–0.46 of the time (chance is 0.25).
- **The teacher used to give Merlin away (RQ3).** While the rule teacher still showed role habits and spoke for 40% of the seats, the Assassin's accuracy in training games rose to 0.44–0.49, against 0.32–0.39 where every seat spoke for itself. Since the teacher speaks by the record alone, the two agree (`results/figures/comm_training_play.png`). With the final networks the Assassin hits 0.35 at tables where today's teacher speaks for 40% of the seats and 0.34 where it speaks for none, but 0.40 with the earlier teacher, and 0.47 in the games where that teacher speaks for Percival (`results/tables/comm/teacher/`). The Assassin still reads a Percival who vouches for Merlin.
- **More training of the same kind changes little.** Over the last 512,000 games of anchored talk the good win rate stayed at 0.36–0.38 and the Assassin's accuracy at 0.33–0.36, pooled over seeds (`results/figures/comm_by_checkpoint.png`, `results/tables/comm/by_checkpoint/`; file numbers are iterations of 512 games).
- **Seeds differ.** Good wins 0.38 against its own seed's evil and 0.56 against another seed's, as in Silent. The tests of RQ2 have so far been run on Silent only.

The checkpoint paths inside the result tables name the local run directories of the training steps (Section 6).

**Not done yet.** The RQ2 tests on Comm; the language layer (Section 9); the human test with talk.

**Scope of conclusions.**
- Conclusions hold only for the defined slots, value ranges, roles and learned policies. Structured messages are not free discussion; claims about natural language need a separate natural-language experiment.
- Finite-budget self-play is not a Nash equilibrium, and this project measures a learning system without trying to solve for one. Teammates who cannot share private information make the equilibrium harder to find than in two-player zero-sum games.
- The readability of talk is partly imposed and partly learned: the rules fix what a statement means and what it obliges; learning decides what is said. The claim is "readable by construction, strategic by learning", not that reinforcement learning discovered human meaning.

## 9 Language Layer for the Human Test

The policy trained by reinforcement learning is the player; the structured message is its only interface. A language model is to serve as mouth and ears, and needs no Avalon strategy [3]:
- **Human to agents:** a typed or spoken sentence is turned into a message. Only clear statements are passed on; hedged ones ("3 seems a bit off") stay in the visible transcript and do not reach the agents.
- **Agents to human:** a message is rendered as a short natural sentence. The sentence is parsed back and compared with the message; on a mismatch a fixed template is shown instead.
- **Accuracy first:** translation is measured offline in both directions before it is used with people.

The browser table of the human test (one human, five agents, replies and promises included) currently runs with templates and a rule-based reader.

## Repository Structure

```
.
├── src/                     The code, imported as `src` (scripts and tests run from the repository root)
│   ├── env/                 Rules, game state machine, action space, information-state encoding
│   ├── agents/              Actor–critic policy, random and rule-based players, rule teacher (speech), logic partner
│   ├── comm/                Message protocol: slots, consistency masks, words that bind deeds, observation features
│   ├── training/            Trajectory recording and GAE, PPO update, opponent pool, trainer, best response
│   ├── evaluation/          Paired frozen evaluation, win-rate decomposition, message statistics
│   ├── runner.py            Plays many games in lockstep with batched decisions
│   └── config.py            YAML-backed training configuration
├── baseline/                Frozen silent baseline (Silent): seed0-2 checkpoints (10,000,384 games) with config, metrics and run info
├── comm/                    Comm networks, seed0-2, played by the human test's talk table: one clean pass of the pipeline, 3,000 iterations of talk (1,536,000 talk games); Section 8's results are from the earlier networks
├── configs/                 baseline_silent.yaml (Silent), comm.yaml (Comm), pilot_silent.yaml (tests)
├── scripts/                 train.py, run_comm_pipeline.sh, distill_speaker.py, evaluate.py, crossplay.py, comm_diagnostics.py, dialect_probe.py, trace_game.py, free_seat_check.py, analyze_messages.py, best_response.py, exploitability.py, plot_training.py, benchmark.py, play_web.py
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
bash scripts/run_comm_pipeline.sh                          # Comm from the frozen baseline: distill, listen, anchored talk
python scripts/evaluate.py --checkpoint runs/comm/talk/seed0/checkpoint_01000.pt --games 2000 --partners
python scripts/evaluate.py --checkpoint runs/comm/talk/seed0/checkpoint_01000.pt --silence all   # every slot silent
python scripts/crossplay.py runs/comm/talk/seed{0,1,2}/checkpoint_01000.pt       # 3 x 3 cross-seed matrix
python scripts/trace_game.py --checkpoint runs/comm/talk/seed0/checkpoint_01000.pt --games 300 --out trace.md
python scripts/best_response.py --checkpoint baseline/seed0/checkpoint_19532.pt --side evil   # optional diagnostic
```

- **Outputs:** each run writes `config.yaml`, `metrics.jsonl` (one JSON object per iteration), `run_info.jsonl` and checkpoints (every 20 iterations) to `runs/<name>/seed<k>/`.
- **Changing settings:** override any config key with `--set`, for example `--set ppo.lr=1e-4`.
- **Several seeds:** `--seed 0 1 2` trains the seeds in parallel processes; each writes its console output to `runs/<name>/seed<k>/train.log`. `--threads` sets the torch CPU threads per run (default with several seeds: an even share of the cores).
- **Continuing a run:** `--resume` continues from the run's `latest.pt`, for example after a crash or to extend the budget.
- **Starting from saved weights:** `--init-from` loads network weights only, for warm starts and for branching Comm from the baseline.

**Reference lines.** These come from 3,000 paired games with the default scripted players; trained policies should beat them.

| Matchup (good vs evil) | Good win rate |
|---|---|
| random vs random | ≈ 0.26 |
| rule vs rule | ≈ 0.14 |
| rule vs random | ≈ 0.65 |
| random vs rule | ≈ 0.01 |

### Human test site

`scripts/human_test.py` serves the human test at one address (live at https://ronova.uk). The home page asks
whether the visitor agrees to research use of their data, then offers two modes; pages are bilingual (Chinese
by default, English on request) and live in `web/human_test/`.

- **Human vs. AI** (`/match/`, `scripts/human_test_match.py`): one human against five seats of the frozen silent
  baseline (seed chosen per game; sampled as in evaluation; no communication, no learning).
- **Assassination analysis** (`/analysis/`, `scripts/human_test_analysis.py`): a visitor records an offline human
  game (leaders, teams, votes, Fail counts). After three successes and the Assassin's and Morgana's seats, the
  record is replayed into the exact Assassin observation and each frozen seed reports P(Merlin) per good seat.

```bash
python scripts/human_test.py                               # http://127.0.0.1:8770
python scripts/human_test.py --no-browser                  # behind a tunnel or reverse proxy to port 8770
python scripts/human_test.py --export-analysis out.jsonl   # shared analysis records, then exit
```

Data (git-ignored) goes to `logs/human_test/`. Only data the visitor agreed to share is used for research:
- `match_games.jsonl`: one line per finished match of a consenting visitor; abandoned matches and matches of
  visitors who declined are never written. Fields: `consent_version`, `finished_utc`, `player` (random id per
  browser session), `player_game`, `baseline_seed`, `checkpoint_sha256`, `seed`, `human_seat` (0–5),
  `requested_role`, `human_role`, `human_won`, `seconds`, `human_decisions` and `game` (full record, all roles).
  Matches served by this site carry `"entry": "site"`; earlier lines come from the previous standalone table.
- `analysis/<id>.json`: one file per analysis record, kept so the visitor can continue it. A record code (cookie,
  enterable on another device) limits each record to its owner; files store only a hash of the code. Only records
  with `"share": true` enter the pooled statistics and `--export-analysis`.

No IP address, name or browser data is stored. Keep these games apart from the automated evaluation: they
measure play against humans, not self-play win rates. Each browser gets its own match table; idle tables expire
after two hours, and at 128 open tables a new visitor replaces one idle for at least ten minutes.

## Compute

The networks are small, so rollouts run on the CPU in worker processes (`workers` in the config; `src/training/workers.py`) and the PPO update runs on the GPU when there is one. Measured on a laptop with a Ryzen 9 8945HX (16 cores) and an RTX 5070 Ti, including evaluation and checkpoints:
- **Silent baseline:** three seeds with 8 workers each take about 2 s per iteration, so the 19,532-iteration baseline takes about 11 h.
- **Comm:** it adds up to 32 speaking decisions per game and a message block of several thousand dimensions to every observation. One pass of the pipeline for three seeds with 5 workers each takes about 7 h (distillation 8 min, listening 80 min, 512,000 games of anchored talk 5.5 h, at 17–28 s per iteration of 512 games) and writes about 8 GB of checkpoints.
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
