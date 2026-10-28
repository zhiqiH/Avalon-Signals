# Structured Communication in Six-Player Avalon

Graduate term project for **CS 7329 Reinforcement Learning** at Southern Methodist University (Fall 2026).

What do agents gain from talking in a hidden-role game? This project trains multi-agent reinforcement learning agents to play six-player *The Resistance: Avalon*, first **in silence** — using only public actions such as team proposals, votes and mission results — and then with a **structured five-slot message channel**. Comparing the two isolates what explicit communication adds to win rates, to strategic behavior, and to the information carried by public actions.

**Status:** research design complete. The environment, scripted players and evaluation are implemented and tested. The silent baseline S (self-play PPO) is fixed: 3 seeds × 1.5M games, results in [results/tables/baseline_silent_final.json](results/tables/baseline_silent_final.json). Communication configurations come next.

## Research Questions

| | Question |
|---|---|
| **RQ1** Game balance | Relative to the silent baseline, which team benefits more when communication is added? |
| **RQ2** Strategic behavior | How do team building, voting, sabotage and identity concealment change? |
| **RQ3** Implicit signaling | In silent play, do public actions carry information? Once explicit messages exist, is that implicit signal complemented or crowded out? |
| **RQ4** Message components | Which of the five message types account for the effect of communication? |

## Setup

**Roles.** Good: Merlin, Percival and two Loyal Servants. Evil: Morgana and the Assassin. Standard six-player rules apply:
- mission sizes are 2-3-4-3-4;
- a team needs at least four approvals, and five consecutive rejections give the win to evil;
- a single Fail card fails a mission;
- after three successful missions, the Assassin gets one guess at Merlin.

**Five-slot message protocol.** After each team proposal and before the vote, every player sends one structured message, starting with the leader. Each slot can be switched on or off independently, and every active slot includes a "say nothing" option. Messages may be false; the referee checks only legality.

| Slot | Content | Values |
|---|---|---|
| ① Role claim | Who I claim to be | none, Servant, Percival, Merlin |
| ② Alignment judgment | Who I think is good or evil | none / good / evil for each other player |
| ③ Rationale | Why I judge so | a mission, vote or proposal record (must point to a real event), a role-knowledge claim, or none |
| ④ Team suggestion | Who should go on the mission | none, or any legal team |
| ⑤ Action intent | Approve or reject the current team; who should respond | none / approve / reject, plus an optional player |

**Configuration ladder.**

| Config | ① | ② | ③ | ④ | ⑤ | Purpose |
|---|:-:|:-:|:-:|:-:|:-:|---|
| S | | | | | | Silent baseline (control): self-play PPO |
| M | | ✓ | ✓ | | | Minimal communication: who is good or evil, and why |
| M+Claim | ✓ | ✓ | ✓ | | | Adds role claims |
| M+Team | | ✓ | ✓ | ✓ | | Adds team suggestions |
| M+Intent | | ✓ | ✓ | | ✓ | Adds action intent |
| F | ✓ | ✓ | ✓ | ✓ | ✓ | Full structured communication |

**Ablations.** The primary effect is ΔGood = P(good wins | F) − P(good wins | S). The M+k configurations measure what each slot adds on top of minimal communication. Switching individual slots off in a frozen F policy measures how much the trained policy depends on them.

**Cross-seed evaluation.** Self-play learns conventions that only work inside one training run. In the silent baseline, good wins 0.39 against its own seed's evil but 0.54 against other seeds' evil. Every configuration therefore reports a 3 × 3 matrix of good_i vs evil_j, and ΔGood is given on both same-seed and cross-seed games. Seeds are only crossed at evaluation, never in training. The project measures a learning system and does not target a Nash equilibrium: an equilibrium-regularized variant (PPO + magnetic mirror descent, `configs/baseline_eq.yaml`, archived) reached approximate exploitability 0.60 vs 0.65 for plain self-play at 1.5M games, which is no meaningful move toward equilibrium.

## Method

- **Agents:** independent PPO with one actor–critic per team, shared by all seats of that team.
  - **Input:** a fixed-length information-state vector. It holds the player's role, its role-specific private knowledge, the complete public history (at most 5 quests × 5 proposals) and its own quest cards, all in seat-relative coordinates. Because this history is bounded, the vector is a complete information state, so the actor and critic can be plain MLPs with no recurrence.
  - **Actions:** a flat 30-way action space covering team proposals, votes, quest cards and the assassination; illegal actions are masked before the softmax.
- **Messages and reward:** message heads are trained RIAL-style with the same terminal ±1 reward (γ = 1, GAE).
- **Self-play:** agents train against an opponent pool that mixes the current, historical and rule-based policies.
- **Silent-first training:**
  - The silent baseline is trained first.
  - Each communication configuration branches from the same seed's silent checkpoint at half the training budget.
  - Message inputs are zero-initialized, so comparisons are paired by seed.
  - A from-scratch control checks whether warm-starting suppresses communication.
- **Diagnostics before full runs:** a toy check with rule-based evil players, sender-side mutual information, and listener-side causal influence.
- **Evaluation:**
  - Frozen policies play 1,000 games per condition and seed with paired role assignments.
  - Every configuration also reports a 3 × 3 cross-seed matrix; approximate exploitability (learned best responses) is an optional diagnostic.
  - Results are reported with seed-level bootstrap confidence intervals, and the win rate is decomposed into mission and assassination outcomes.

## Repository Structure

```
.
├── docs/                    Design documents (written in Chinese)
│   ├── Avalon_Research_Design.docx
│   ├── Avalon_Implementation_Spec.docx
│   └── src/                 python-docx scripts that generate both documents
├── references/              Reading notes on the core papers (PDFs are not tracked)
├── src/avalon_rl/
│   ├── env/                 Rules, game state machine, action space, information-state encoding
│   ├── agents/              Actor–critic policy, random and rule-based players
│   ├── training/            Trajectory recording and GAE, PPO update, opponent pool, trainer, best response (diagnostic)
│   ├── evaluation/          Paired frozen evaluation and win-rate decomposition
│   ├── comm/                Five-slot message protocol (next step)
│   ├── runner.py            Plays many games in lockstep with batched decisions
│   └── config.py            YAML-backed training configuration
├── configs/                 baseline_silent.yaml (S, final), pilot_silent.yaml (short pilot), baseline_eq.yaml (archived experiment)
├── scripts/                 train.py, evaluate.py, crossplay.py, best_response.py, exploitability.py, plot_training.py, benchmark.py, launch_windows.ps1
├── tests/                   Rules, information leakage, observation encoding, GAE, a short training run
├── notebooks/               Analysis notebooks
├── results/                 Curated figures and tables
└── runs/                    Training outputs (not tracked)
```

## Quick Start

Python 3.10 or newer; the environment is named after the project on every machine.

```bash
conda create -y -n avalon-signals --override-channels -c conda-forge python=3.12
conda activate avalon-signals
pip install -e ".[dev]"
pytest                                                     # rules, leakage, GAE and a short training run
python scripts/benchmark.py --games 2048                   # simulation and update throughput on this machine
python scripts/train.py --config configs/pilot_silent.yaml # 2,048-game pilot
python scripts/train.py --config configs/baseline_silent.yaml --seed 0
python scripts/evaluate.py --checkpoint runs/baseline_silent/seed0/latest.pt --games 1000
python scripts/crossplay.py runs/baseline_silent/seed{0,1,2}/latest.pt     # 3 x 3 cross-seed matrix
```

- **Outputs:** each run writes `config.yaml`, `metrics.jsonl` (one JSON object per iteration) and checkpoints to `runs/<name>/seed<k>/`.
- **Changing settings:** override any config key with `--set`, for example `--set ppo.lr=1e-4`.
- **Continuing a run:** `--resume` continues an interrupted run.
- **Starting from saved weights:** `--init-from` loads network weights only, for warm starts and the branch training of the communication configurations.

**Reference lines.** These come from 3,000 paired games with the default scripted players; trained policies should beat them.

| Matchup (good vs evil) | Good win rate |
|---|---|
| random vs random | ≈ 0.26 |
| rule vs rule | ≈ 0.14 |
| rule vs random | ≈ 0.65 |
| random vs rule | ≈ 0.01 |

## Running on a Remote GPU Machine

Code is edited on a Mac. Experiments run over SSH on a Windows laptop with an RTX 5070 Ti; code reaches the laptop through `git pull`. The full setup and daily loop are in [docs/remote_windows.md](docs/remote_windows.md).

**Where the time goes.** The networks are small and rollouts run on the CPU, so throughput comes mostly from running several seeds in parallel. The GPU only speeds up the PPO update. On an Apple M5 CPU, self-play collects about 500 games/s, so a 100,000-game run takes minutes rather than hours. Run `scripts/benchmark.py` on the remote machine before fixing the budget.

**Starting runs.** Windows OpenSSH kills a session's processes when the connection drops. Start runs with the launcher, which keeps them alive after a disconnect:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\launch_windows.ps1 -Config configs\baseline_silent.yaml -Seeds 0,1,2 -Device cpu -Threads 4
```

On the laptop's CPU (Ryzen 9 8945HX), training is faster than on its GPU for these small networks. The launcher can also start any Python job detached with `-Raw` and `-Name`, for example a best-response run.

## Design Documents

| Document | Contents |
|---|---|
| Research Design ([PDF](docs/Avalon_Research_Design.pdf), [DOCX](docs/Avalon_Research_Design.docx)) | Research questions, game setup, message protocol, experimental design, metrics and evidence standards, scope of conclusions |
| Implementation Spec ([DOCX](docs/Avalon_Implementation_Spec.docx)) | Must-have vs. optional components, environment, message encoding, networks, training pipeline, diagnostics, evaluation, logging, compute, optional natural-language extension |

To rebuild the documents:

```bash
pip install -r docs/src/requirements.txt
python docs/src/build_research.py
python docs/src/build_implementation.py
```

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

Paper PDFs are kept locally and are not redistributed here; [references/README.md](references/README.md) has per-paper reading notes (in Chinese).
