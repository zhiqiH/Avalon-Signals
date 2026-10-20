# Structured Communication in Six-Player Avalon

Graduate term project for **CS 7329 Reinforcement Learning** at Southern Methodist University (Fall 2026).

What do agents gain from talking in a hidden-role game? This project trains multi-agent reinforcement learning agents to play six-player *The Resistance: Avalon*, first **in silence** — using only public actions such as team proposals, votes and mission results — and then with a **structured five-slot message channel**. Comparing the two isolates what explicit communication adds to win rates, to strategic behavior, and to the information carried by public actions.

**Status:** research design complete; implementation in progress.

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
| S | | | | | | Silent baseline (control) |
| M | | ✓ | ✓ | | | Minimal communication: who is good or evil, and why |
| M+Claim | ✓ | ✓ | ✓ | | | Adds role claims |
| M+Team | | ✓ | ✓ | ✓ | | Adds team suggestions |
| M+Intent | | ✓ | ✓ | | ✓ | Adds action intent |
| F | ✓ | ✓ | ✓ | ✓ | ✓ | Full structured communication |

**Ablations.** The primary effect is ΔGood = P(good wins | F) − P(good wins | S). The M+k configurations measure what each slot adds on top of minimal communication. Switching individual slots off in a frozen F policy measures how much the trained policy depends on them.

## Method

- **Agents:** recurrent independent PPO with an MLP + GRU actor–critic for each team. Parameters are shared within a team, actions are legal-action masked, and each game phase has its own action head.
- **Messages and reward:** message heads are trained RIAL-style with the same terminal ±1 reward (γ = 1, GAE).
- **Self-play:** agents train against an opponent pool that mixes the current, historical and rule-based policies.
- **Silent-first training:**
  - The silent baseline is trained first.
  - Each communication configuration branches from the same seed's silent checkpoint at half the training budget.
  - Message inputs are zero-initialized, so comparisons are paired by seed.
  - A from-scratch control checks whether warm-starting suppresses communication.
- **Diagnostics before full runs:** a toy check with rule-based evil players, sender-side mutual information, and listener-side causal influence.
- **Evaluation:** frozen policies play 1,000 games per condition and seed with paired role assignments. Results are reported with seed-level bootstrap confidence intervals, and the win rate is decomposed into mission and assassination outcomes.

## Repository Structure

```
.
├── docs/                    Design documents (written in Chinese)
│   ├── Avalon_Research_Design.docx
│   ├── Avalon_Implementation_Spec.docx
│   └── src/                 python-docx scripts that generate both documents
├── references/              Reading notes on the core papers (PDFs are not tracked)
├── src/avalon_rl/
│   ├── env/                 Six-player game engine and observation filtering
│   ├── comm/                Five-slot message protocol and masks
│   ├── agents/              Recurrent actor–critic, random and rule-based opponents
│   ├── training/            PPO, self-play opponent pool, silent-first branch training
│   └── evaluation/          Communication diagnostics, frozen evaluation, statistics
├── configs/                 Experiment configurations (S, M, M+k, F)
├── scripts/                 Training and evaluation entry points
├── tests/                   Rule, information-leakage and message-mask tests
├── notebooks/               Analysis notebooks
└── results/                 Curated figures and tables
```

## Design Documents

| Document | Contents |
|---|---|
| [Research Design](docs/Avalon_Research_Design.docx) | Research questions, game setup, message protocol, experimental design, metrics and evidence standards, scope of conclusions |
| [Implementation Spec](docs/Avalon_Implementation_Spec.docx) | Must-have vs. optional components, environment, message encoding, networks, training pipeline, diagnostics, evaluation, logging, compute, optional natural-language extension |

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
