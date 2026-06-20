# ClaudeDJ — Product Requirements Document

**Reaction-aware autonomous DJ agent**
AI Hackathon Berkeley 2026 · 3-person team · 24-hour build

> One-sentence pitch: *A Claude-driven DJ agent that uses Redis memory and vector search to adapt music from a listener's live reactions.*

---

## 1. Summary

ClaudeDJ keeps music playing, reads the listener's live reactions (webcam + CLI), and continuously re-ranks and re-queues songs using Redis memory and vector search. A Claude Code SDK agent is the queue manager; an MCP server is the tool boundary; Redis is the memory and retrieval layer. The product surface is a tiny, ambient mini-player — no skip button, no queue editor. The DJ narrates only when it starts or changes direction.

This PRD is grounded in **Pulse** (Bao et al., *"Your reactions suggest you liked the movie: Automatic content rating via reaction sensing,"* UbiComp 2013), which first validated the thesis we depend on, and adapts it for a real-time, single-listener, agent-driven product.

---

## 2. Research grounding: what we build on (Pulse, 2013)

Pulse proved the core idea a decade ago: you can infer how much someone likes content from their **passive reactions** instead of asking them. Pulse instrumented a movie player with the device's front camera, microphone, accelerometer/gyroscope, and touch controls; learned the mapping from sensed reactions to ratings; and produced ratings at **multiple granularities** — per-minute segment ratings rolled up into an overall star rating, plus a reaction tag-cloud and reaction-indexed highlight clips. Across 11 users watching 4–6 movies, its automatically generated star ratings landed within a **mean gap of 0.46 on a 5-point scale** of users' real ratings. That is the evidence base under ClaudeDJ.

ClaudeDJ keeps Pulse's thesis but diverges on the axes that define our product:

| Axis | Pulse (2013) | ClaudeDJ (2026) |
|---|---|---|
| Timing | Post-hoc — rate *after* watching | Real-time — adapt *mid-song* |
| Output | Passive annotation (rating, tags) | **Action** — choose & queue the next song (closed loop) |
| Intelligence | Offline ML (collaborative filtering + GPR + SVM) trained across many users | In-context reasoning by a Claude agent over per-session Redis memory; **no training** |
| Sensing | Sensors on the playback device | External webcam + CLI |
| Scope | Multi-user collaborative filtering | Single listener, personalized live |

**The pitch in one breath:** Pulse showed reaction → preference works, but it produced an *offline, multi-user, passive label*. ClaudeDJ makes it *real-time and single-user*, replaces the offline ML stack with a *Claude agent reasoning over Redis*, and **closes the loop** — it acts on the signal by picking the next song rather than just annotating content.

---

## 3. Design principles derived from Pulse

Each principle below carries its Pulse rationale and drives the requirements in §6.

**P1 — Implicit over explicit.** Pulse's premise was that explicit ratings are costly, low-incentive, and a "highly-lossy compression" of experience. → ClaudeDJ has no rating UI. The listener just listens; the mini-player deliberately has no skip button.

**P2 — Reaction meaning is content-dependent; never read a signal in isolation.** Pulse's sharpest caution: the same reaction means opposite things across content — laughter is positive in a comedy, negative in horror; some users fidget when engaged, others go still. → ClaudeDJ must interpret a reaction *relative to the current track context*. Movement during an up-tempo track signals engagement; stillness during a ballad does **not** signal dislike. This contextual judgment is exactly what an LLM agent does well, and it differentiates us from a naive emotion classifier. Concretely, `get_session_context` must include the current track's energy/cluster so Claude reads the score in context, not as an absolute.

**P3 — Calibrate per person; distrust any universal model.** Pulse found a one-size-fits-all model collapses to the mean because of three heterogeneities — user *behavior*, *environment*, and *taste* — and recovered by bootstrapping from high-confidence signals to learn each user's idiosyncrasies. → ClaudeDJ should (a) capture a short neutral baseline of the listener at session start and score reactions as *deltas from baseline*, and (b) anchor decisions on clear, strong reactions and treat middling signals cautiously. Personalization lives in **memory**, not in a pretrained classifier.

**P4 — Fuse multiple modalities; no single channel is reliable.** Pulse combined visual (face/eye/lip), acoustic (laughter vs. speech via MFCC + SVM), motion, and control signals. → ClaudeDJ fuses presence + head movement + facial expression + playback behavior (+ optional singing/humming) into one aggregate score so no noisy channel dominates.

**P5 — Bridge the time-scale gap by windowing.** Pulse reconciled second-scale reactions (a laugh lasts seconds) with minute-scale ratings by labeling short windows and aggregating. → ClaudeDJ aggregates sub-second reaction traces over a **mid-song window** into a single engagement score before deciding; the agent decides at the track/cluster scale, not the frame scale.

**P6 — Sensing is a background cost; never block the experience.** Pulse kept its sensing overhead marginal (≈16% extra energy on tablets) and processed locally. → ClaudeDJ's reaction worker and narration must never block playback; the queue is always populated before the current song ends; a deterministic fallback covers a slow agent.

**P7 — Privacy by default: derive and discard.** Pulse kept raw sensor data on-device and uploaded only ratings/labels with consent, explicitly flagging face-detection privacy concerns. → ClaudeDJ processes webcam frames **locally** and stores only **derived reaction scores, never raw video**. (Pairs with our lyrics policy: store derived vectors + source references, never retained raw lyrics.) This is our responsible-AI story for the Claude track.

---

## 4. Goals & non-goals

**Goals**
- G1 — Start from a natural-language request ("play reggaeton").
- G2 — Keep 3–6 tracks queued at all times; never dead air.
- G3 — Read live reactions and adapt mid-session: positive → similar; negative → shift away; neutral → hold then drift.
- G4 — Respect the cluster run rule (stay 3–6 songs in a working cluster).
- G5 — Maintain compact session memory in Redis so the DJ reasons over recent context, not raw logs.
- G6 — Ambient UX: tiny mini-player, short narration, no manual controls.
- G7 — Demonstrable end-to-end loop with Redis usage clearly visible (judging).

**Non-goals**
- No chat interface, manual queue editor, or skip button.
- No raw lyrics storage unless the provider license explicitly allows it.
- No claim that Spotify provides native embeddings or official full lyrics.
- No multi-user collaborative filtering — the Pulse mechanism we deliberately drop.
- No pretrained preference model — the agent + memory does the interpretive work.

---

## 5. User & primary use case

A single listener at a desk or lounge, music as a secondary activity, webcam in view. They state a vibe and then just listen or work. Success = they keep listening without touching any controls, and the music keeps fitting. (Mirrors Pulse's "minimal user participation" listener.)

---

## 6. Functional requirements

### Request & playback
- **FR-1** Accept a natural-language vibe request via CLI (MVP surface) and start playback.
- **FR-2** Maintain a populated queue of 3–6 upcoming tracks at all times. *(P6, G2)*
- **FR-3** Begin playback through Spotify (Premium device playback) from a selected track.

### Reaction sensing
- **FR-4** Capture a per-listener neutral baseline in the first ~3 s of a session; score reactions as deltas from baseline. *(P3)*
- **FR-5** Produce an aggregate reaction score by fusing presence, head movement, facial expression, playback behavior, and optional vocal (singing/humming) cues. *(P4)*
- **FR-6** Aggregate raw reaction traces over a mid-song window into a single engagement score before any decision. *(P5)*
- **FR-7** Reaction interpretation is conditioned on the current track context (energy/cluster), never treated as absolute valence. *(P2)*
- **FR-8** CLI feedback (`like` / `dislike` / `meh`) is a first-class signal and the MVP demo input; webcam blends in when available.

### Decision loop
- **FR-9** On request: search embeddings → select 3–6 tracks → narrate the opening choice → start playback.
- **FR-10** Mid-song: read compact session context; classify the windowed signal as positive / neutral / negative.
- **FR-11** Positive → mark liked, search similar (seed = current track), refresh queue.
- **FR-12** Negative → mark current cluster disliked, search shifted candidates, replace queue, narrate the change.
- **FR-13** Neutral → hold queue if cluster run < min (3); allow a slight shift once min is satisfied.
- **FR-14** Enforce the cluster run rule: stay 3–6 songs; a *strongly* negative reaction may break the minimum early. *(P2/P3 — strong signals trusted, weak ones not)*
- **FR-15** Decide before the current song ends; never wait for track end. *(P6)*
- **FR-16** Anchor on high-confidence reactions; do not over-react to middling scores. *(P3)*

### Memory (Redis)
- **FR-17** Persist a reaction trace per track: timestamp, presence/movement/face/playback component scores, aggregate score.
- **FR-18** Maintain session state: current track, queue, recent tracks, current cluster, cluster streak, min/max run (3/6), DJ status.
- **FR-19** Maintain memory: liked clusters, disliked clusters, recent skips, yesterday's genres, tracks to avoid replaying.
- **FR-20** `get_session_context` returns a **compact decision bundle** (current track + context, seconds remaining, queue, current score, recent trend, recent tracks, streak, liked/disliked clusters, yesterday's genres, recommended action) — not raw logs. *(P5/P6)*
- **FR-21** At session end, write a compact session summary + summary embedding for later structured/semantic history search. *(post-MVP)*

### UX
- **FR-22** Draggable mini-player: album art, title, artist, one status line, optional progress bar. No controls.
- **FR-23** Narration is short and occurs only on start and on direction changes.
- **FR-24** Status line reflects state: `listening` / `staying close · n/6` / `shifting after this` / `reading the room`.

### Reliability
- **FR-25** Never block playback on embedding generation or narration. *(P6)*
- **FR-26** Deterministic fallback: if Claude is slow, play next from pre-ranked candidates.
- **FR-27** Keep a verified fallback playlist (≥20 tracks) with confirmed metadata, lyrics coverage, and embeddings.
- **FR-28** Store derived vectors + source metadata by default; never retain raw lyrics or raw webcam frames. *(P7)*

---

## 7. Reaction model (Pulse-informed core)

The reaction model converts raw webcam/CLI signals into a contextual positive/neutral/negative decision. It follows Pulse's fuse → window → interpret pattern, with the final judgment made by the agent rather than a regression.

**Components (FR-5).** Per ~1 s frame, extract:
- `presence` — is the listener there at all? (absent ⇒ no decision; hold queue)
- `movement` — head/body motion magnitude (head-nod / bob is the strongest "into it" cue)
- `face` — expression deltas from baseline (smile, brow, engagement), as deltas not absolutes (FR-4)
- `playback` —
