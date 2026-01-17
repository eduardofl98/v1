import streamlit as st

import time
import uuid
import random
from dataclasses import dataclass, asdict

import pandas as pd
import streamlit as st


# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "AI Training & Loss Aversion (Lottery Experiment)"
N_PRE = 40
N_TRAIN = 25
N_POST = 40

# If you later connect an LLM, you can switch this on
USE_LLM = False


# -----------------------------
# LOTTERY MODEL
# -----------------------------
@dataclass
class MixedGamble:
    p_win: float
    win: float
    p_lose: float
    lose: float
    gamble_id: str

    @property
    def ev(self) -> float:
        return self.p_win * self.win - self.p_lose * self.lose


def sample_mixed_gamble(difficulty: int, rng: random.Random) -> MixedGamble:
    """
    Generates a simple 50/50 mixed gamble.
    difficulty controls typical magnitudes (higher = larger stakes / more challenging).
    """
    p = 0.5

    # difficulty bucket affects range
    # keep ranges modest and consistent with thesis experiments
    if difficulty <= 0:
        win = rng.choice([6, 8, 10, 12, 14, 16])
        lose = rng.choice([4, 6, 8, 10, 12])
    elif difficulty == 1:
        win = rng.choice([10, 12, 14, 16, 18, 20, 22])
        lose = rng.choice([8, 10, 12, 14, 16, 18])
    else:
        win = rng.choice([16, 18, 20, 22, 24, 26, 28, 30])
        lose = rng.choice([12, 14, 16, 18, 20, 22, 24])

    # Ensure it's a true "mixed" gamble and not degenerate
    win = float(win)
    lose = float(lose)

    return MixedGamble(
        p_win=p,
        win=win,
        p_lose=p,
        lose=lose,
        gamble_id=str(uuid.uuid4())[:8],
    )


# -----------------------------
# RULES / TRAINING LOGIC
# -----------------------------
def simple_loss_aversion_flag(gamble: MixedGamble, decision: str) -> str:
    """
    A very simple flagger:
    - If EV is clearly positive and user rejects, tag as potential loss aversion.
    - If EV is clearly negative and user accepts, tag as risk-seeking or noise.
    """
    ev = gamble.ev
    if ev >= 2.0 and decision == "reject":
        return "loss_aversion_possible"
    if ev <= -2.0 and decision == "accept":
        return "risk_seeking_or_noise"
    return "none"


def adapt_difficulty(current: int, recent_flags: list[str]) -> int:
    """
    Simple adaptivity:
    - If many 'loss_aversion_possible' flags recently, reduce difficulty (make it easier / clearer).
    - If few flags and stable, increase slightly.
    """
    if not recent_flags:
        return current

    la = sum(1 for f in recent_flags if f == "loss_aversion_possible")
    n = len(recent_flags)

    if la / n >= 0.6:
        return max(0, current - 1)
    if la / n <= 0.2:
        return min(2, current + 1)
    return current


def coach_feedback_template(gamble: MixedGamble, decision: str, flag: str) -> str:
    """
    Placeholder coaching text (no LLM). Keep short and neutral.
    """
    ev = gamble.ev
    ev_text = f"EV ≈ {ev:+.1f}€ (expected value)."
    frame = (
        f"This is a 50/50 gamble: win {gamble.win:.0f}€ or lose {gamble.lose:.0f}€."
    )

    if flag == "loss_aversion_possible":
        return (
            f"{frame} {ev_text} If the possible loss felt disproportionately salient, "
            "try focusing on the probability structure and the long-run average outcome."
        )
    elif flag == "risk_seeking_or_noise":
        return (
            f"{frame} {ev_text} Consider whether you would make the same choice repeatedly—"
            "what would happen on average over many trials?"
        )
    else:
        # generic micro-coaching
        return (
            f"{frame} {ev_text} A quick check: what mattered more—"
            "the potential loss magnitude or the expected value?"
        )


# -----------------------------
# SESSION STATE INIT
# -----------------------------
def init_state():
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = str(uuid.uuid4())[:8]

    if "phase" not in st.session_state:
        st.session_state.phase = "consent"  # consent -> pre -> train -> post -> done

    if "trial_index" not in st.session_state:
        st.session_state.trial_index = 0

    if "difficulty" not in st.session_state:
        st.session_state.difficulty = 0

    if "current_gamble" not in st.session_state:
        rng = random.Random()
        st.session_state.current_gamble = sample_mixed_gamble(st.session_state.difficulty, rng)

    if "rng_seed" not in st.session_state:
        st.session_state.rng_seed = int(time.time())

    if "logs" not in st.session_state:
        st.session_state.logs = []  # list of dict rows

    if "last_flags" not in st.session_state:
        st.session_state.last_flags = []  # store recent flags for adaptivity

    if "trial_started_at" not in st.session_state:
        st.session_state.trial_started_at = time.time()

    if "reflection" not in st.session_state:
        st.session_state.reflection = ""


def rng():
    # deterministic-ish per session if needed
    return random.Random(st.session_state.rng_seed + st.session_state.trial_index)


def phase_total_trials(phase: str) -> int:
    return {"pre": N_PRE, "train": N_TRAIN, "post": N_POST}.get(phase, 0)


def advance_trial():
    st.session_state.trial_index += 1
    st.session_state.reflection = ""
    st.session_state.trial_started_at = time.time()

    # End-of-phase transition
    total = phase_total_trials(st.session_state.phase)
    if st.session_state.trial_index >= total:
        if st.session_state.phase == "pre":
            st.session_state.phase = "train"
        elif st.session_state.phase == "train":
            st.session_state.phase = "post"
        elif st.session_state.phase == "post":
            st.session_state.phase = "done"

        st.session_state.trial_index = 0
        st.session_state.difficulty = 0
        st.session_state.last_flags = []

    # Sample next gamble for the new phase/trial (if not done)
    if st.session_state.phase in ("pre", "train", "post"):
        st.session_state.current_gamble = sample_mixed_gamble(st.session_state.difficulty, rng())


def log_decision(decision: str, reflection: str, feedback: str, flag: str):
    g: MixedGamble = st.session_state.current_gamble
    rt = time.time() - st.session_state.trial_started_at

    row = {
        "participant_id": st.session_state.participant_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": st.session_state.phase,
        "trial_in_phase": st.session_state.trial_index,
        "gamble_id": g.gamble_id,
        "p_win": g.p_win,
        "win": g.win,
        "p_lose": g.p_lose,
        "lose": g.lose,
        "ev": g.ev,
        "decision": decision,              # accept / reject
        "flag": flag,                      # loss_aversion_possible / ...
        "feedback": feedback,              # coach text (template or LLM)
        "reflection": reflection.strip(),  # user free-text
        "rt_seconds": round(rt, 3),
        "difficulty": st.session_state.difficulty,
    }
    st.session_state.logs.append(row)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="centered")
init_state()

st.title("🎯 " + APP_TITLE)
st.caption(f"Participant ID: `{st.session_state.participant_id}`")

phase = st.session_state.phase

if phase == "consent":
    st.subheader("Consent & instructions")
    st.write(
        "You will make a series of choices between **accepting** or **rejecting** a 50/50 lottery.\n\n"
        "- If you **accept**, you face the possible gain or loss.\n"
        "- If you **reject**, you get 0€.\n\n"
        "During the training phase, you will receive brief coaching feedback and a short reflection prompt."
    )

    agree = st.checkbox("I understand and agree to participate.", value=False)
    if st.button("Start", disabled=not agree):
        st.session_state.phase = "pre"
        st.session_state.trial_index = 0
        st.session_state.difficulty = 0
        st.session_state.last_flags = []
        st.session_state.trial_started_at = time.time()
        st.session_state.current_gamble = sample_mixed_gamble(st.session_state.difficulty, rng())
        st.rerun()

elif phase in ("pre", "train", "post"):
    total = phase_total_trials(phase)
    idx = st.session_state.trial_index + 1

    st.subheader(f"Phase: {phase.upper()}  —  Trial {idx}/{total}")

    g: MixedGamble = st.session_state.current_gamble

    st.markdown(
        f"""
**Decision:** Accept or Reject the following 50/50 gamble:

- 50% chance to **win**: **{g.win:.0f}€**
- 50% chance to **lose**: **{g.lose:.0f}€**
- If you **reject**: **0€**
        """.strip()
    )

    col1, col2 = st.columns(2)

    # Reflection only during training
    reflection = ""
    if phase == "train":
        st.markdown("**Quick reflection (1 sentence):**")
        reflection = st.text_input(
            "What influenced your choice most?",
            value=st.session_state.reflection,
            placeholder="e.g., the possible loss felt too big, or the expected value mattered more...",
            label_visibility="collapsed",
        )
        st.session_state.reflection = reflection

    def handle(decision: str):
        flag = simple_loss_aversion_flag(g, decision)

        # Update adaptivity only during training
        if phase == "train":
            st.session_state.last_flags = (st.session_state.last_flags + [flag])[-10:]
            st.session_state.difficulty = adapt_difficulty(st.session_state.difficulty, st.session_state.last_flags)

        # Feedback (template now; replace with LLM later)
        feedback = ""
        if phase == "train":
            feedback = coach_feedback_template(g, decision, flag)

        log_decision(
            decision=decision,
            reflection=reflection if phase == "train" else "",
            feedback=feedback,
            flag=flag,
        )

        advance_trial()
        st.rerun()

    with col1:
        if st.button("✅ Accept", use_container_width=True):
            handle("accept")
    with col2:
        if st.button("❌ Reject", use_container_width=True):
            handle("reject")

    if phase == "train":
        st.info("After each choice, you’ll receive short coaching feedback (currently a template).")

    # Show last feedback (optional)
    if st.session_state.logs:
        last = st.session_state.logs[-1]
        if last["phase"] == "train" and last["feedback"]:
            st.markdown("### Coach feedback")
            st.write(last["feedback"])

elif phase == "done":
    st.subheader("✅ Finished")
    st.write("Thanks! You can download your recorded decisions below.")

    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"loss_aversion_experiment_{st.session_state.participant_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button("Restart session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
