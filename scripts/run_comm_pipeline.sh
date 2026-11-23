#!/usr/bin/env bash
# Staged communication training (configuration F: all five message slots), all seeds in parallel at every stage:
#   1. distill    the rule teacher's messages into the message heads of each seed's frozen silent baseline
#   2. stage2_F   listeners learn (LoRA) while every learned player speaks with the frozen distilled speaker
#   3. stage3b_F  anchored talk: speakers learn too, without a message entropy bonus. Two anchors keep the
#                 talk's meaning the teacher's: a KL term to the distilled speaker that falls to KL_FLOOR and
#                 stays there, and the rule teacher itself speaking for TEACHER_SHARE of the seats.
#   4. silent_lora  the silent control (CONTROL=1): the same adaptation and budget without a message channel
# In stages 2 and 3 a share PARTNER of the games puts logic partners (src/agents/logic_agent.py) in one or
# two Servant/Percival seats, the rule teacher speaking for them; the rest splits evenly between self-play,
# past snapshots and the rule players.
# Three seeds with five rollout workers each use the machine best.
# Running the script again continues where it stopped: a finished distillation is kept, and every training
# stage resumes from its latest.pt.
#
#   bash scripts/run_comm_pipeline.sh                      # runs/comm_F, seeds 0 1 2
#   SEEDS="0" OUT=/tmp/smoke LISTEN_ITERS=2 TALK_ITERS=2 DISTILL_GAMES=200 bash scripts/run_comm_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-runs/comm_F}
SEEDS=${SEEDS:-"0 1 2"}
WORKERS=${WORKERS:-5}
LISTEN_ITERS=${LISTEN_ITERS:-300}
TALK_ITERS=${TALK_ITERS:-1000}
DISTILL_GAMES=${DISTILL_GAMES:-4000}
KL_FLOOR=${KL_FLOOR:-0.03}
TEACHER_SHARE=${TEACHER_SHARE:-0.4}
PARTNER=${PARTNER:-0.25}
LORA_RANK=${LORA_RANK:-8}
CONTROL=${CONTROL:-0}
GAMES_PER_ITER=512
DISTILLED="$OUT/distill_F/seed{seed}/checkpoint_00000.pt"
LISTENED="$OUT/stage2_F/seed{seed}/checkpoint_$(printf %05d "$LISTEN_ITERS").pt"

stage() { echo; echo "=== $(date '+%F %T') $*"; }
# shellcheck disable=SC2086  # SEEDS is a list
train() { "$PY" scripts/train.py --seed $SEEDS --device cuda --workers "$WORKERS" --out-dir "$OUT" --resume --set "adapt.lora_rank=$LORA_RANK" "$@"; }
OTHER=$("$PY" -c "print((1 - $PARTNER) / 3)")
MIX=(--set "opponents.p_partner=$PARTNER" --set "opponents.p_self=$OTHER" --set "opponents.p_pool=$OTHER" --set "opponents.p_rule=$OTHER")

stage "1/4 distill ($DISTILL_GAMES games per seed)"
pids=()
for s in $SEEDS; do
    [ -f "$OUT/distill_F/seed$s/checkpoint_00000.pt" ] && continue
    mkdir -p "$OUT/distill_F/seed$s"
    "$PY" scripts/distill_speaker.py --baseline "baseline/seed$s/checkpoint_19532.pt" --config configs/comm_F.yaml \
        --seed "$s" --games "$DISTILL_GAMES" --out "$OUT/distill_F/seed$s" > "$OUT/distill_F/seed$s/distill.log" 2>&1 &
    pids+=($!)
done
for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid"; done

stage "2/4 listener stage ($LISTEN_ITERS iterations)"
train --config configs/comm_F.yaml --name stage2_F --total-games $((LISTEN_ITERS * GAMES_PER_ITER)) \
    --init-from "$DISTILLED" --set "adapt.speaker_from=$DISTILLED" "${MIX[@]}"

stage "3/4 anchored talk ($TALK_ITERS iterations, KL floor $KL_FLOOR, teacher share $TEACHER_SHARE)"
train --config configs/comm_F.yaml --name stage3b_F --total-games $((TALK_ITERS * GAMES_PER_ITER)) \
    --init-from "$LISTENED" --set ppo.msg_ent_coef=0.0 --set adapt.msg_kl_coef=0.1 --set adapt.msg_kl_anneal=330 \
    --set "adapt.msg_kl_floor=$KL_FLOOR" --set "adapt.teacher_share=$TEACHER_SHARE" \
    --set "adapt.anchor_from=$DISTILLED" "${MIX[@]}"

if [ "$CONTROL" = 1 ]; then
    stage "4/4 silent control ($((LISTEN_ITERS + TALK_ITERS)) iterations)"
    train --config configs/baseline_silent.yaml --name silent_lora \
        --total-games $(((LISTEN_ITERS + TALK_ITERS) * GAMES_PER_ITER)) --init-from "baseline/seed{seed}/checkpoint_19532.pt" \
        --set checkpoint_every=20 "${MIX[@]}"
fi

stage "done"
