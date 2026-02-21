# Copilot Multi-Agent Architecture Explanation

## Shared Infrastructure

Before the architectures, every call goes through the same two layers:

```
Agent.act()
    └─► call_copilot_with_retry()
            ├─ _build_prompt(system_prompt, messages)  ← flattens history to one string
            ├─ client.create_session(SessionConfig(model=...))
            ├─ session.send_and_wait(MessageOptions(prompt=...))
            └─ session.destroy()
                [retries up to 5x with exponential backoff on 429/timeout]
```

`_build_prompt` in `plancraft/agents/copilot_llm.py` serializes the entire conversation like:
```
[System Instructions] ...
[Observation] <env state>
[Your Previous Action] <last action>
[Your Action] Respond with exactly one action:
```

Every architecture shares the same `self.conversation` list (from `CopilotBaseAgent`) which accumulates the episode's history. The **Copilot SDK does not have a persistent session** — a fresh session is created and destroyed for every single LLM call.

---

## Architecture 1 — Independent MAS (`CopilotIndependentAgent`)

```
                 ┌─────────────────────────────────┐
Observation ───► │         self.conversation        │
                 └────────────┬────────────────────┘
                              │ asyncio.gather (parallel)
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         Copilot Call    Copilot Call    Copilot Call   (N agents, temp=0.7)
         Agent 0         Agent 1         Agent N-1
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                     Counter(actions).most_common(1)
                              │
                       MAJORITY VOTE
                              │
                    ┌─────────▼──────────┐
                    │  search?  ──► _oracle_search()  ──► recurse
                    │  else  ──► append to conversation
                    └────────────────────┘
                              │
                           action
```

- **No communication** between agents — each sees identical history
- **Temperature 0.7** introduces diversity; ties broken by `Counter.most_common`
- `search:` triggers an **oracle lookup** and a **recursive `act(None)` call** before returning

---

## Architecture 2 — Centralized MAS (`CopilotCentralizedAgent`)

```
                 ┌──────────────────────────────┐
Observation ───► │        self.conversation      │
                 └─────────────┬────────────────┘
                               │
                               ▼  [sequential — must finish before step 2]
                    ┌──────────────────────┐
                    │  Copilot Call        │
                    │  ORCHESTRATOR        │  ← planning_prompt (SYSTEM_PROMPT + "output a 1-sentence plan")
                    │  temp=0.0 (default)  │
                    └──────────┬───────────┘
                               │  plan = "Place logs in A1, A2..."
                               ▼
               conversation + [model: plan] + [user: "what is the exact action?"]
                               │
                               ▼  [sequential]
                    ┌──────────────────────┐
                    │  Copilot Call        │
                    │  WORKER              │  ← SYSTEM_PROMPT only
                    │  temp=0.0 (default)  │
                    └──────────┬───────────┘
                               │  action = "move: from [I1] to [A1] with quantity 1"
                               ▼
                    search? ──► oracle ──► recurse
                    else ──► append + return action
```

- **Strictly sequential** — 2 Copilot calls per step
- Both calls use the **same model** — the roles are differentiated purely by the prompt content
- The plan is **not** appended to `self.conversation`; it's injected as ephemeral context only for the worker call

---

## Architecture 3 — Decentralized MAS (`CopilotDecentralizedAgent`)

```
                 ┌──────────────────────────────┐
Observation ───► │        self.conversation      │
                 └─────────────┬────────────────┘
                               │
                    ── INITIAL PROPOSALS (parallel) ──
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         Copilot Call     Copilot Call     Copilot Call     (N agents, temp=0.7)
         proposal[0]      proposal[1]      proposal[N-1]
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   for r in range(rounds):    │ ◄──────────┐
                    │                              │            │
                    │  debate_context =            │            │
                    │  "Other agents proposed:\n  │            │
                    │   - Agent 0: proposal[0]\n  │            │
                    │   - Agent 1: proposal[1]\n  │            │
                    │   Given these, update action"│            │
                    │                              │            │
                    │  ── DEBATE ROUND (parallel) ──           │
                    │  ┌──────────┬──────────┐    │            │
                    │  ▼          ▼          ▼    │            │
                    │  Copilot   Copilot   Copilot │ (N, 0.7)  │
                    │  updated   updated   updated │            │
                    │  prop[0]   prop[1]   prop[N] │            │
                    │  └──────────┴──────────┘    │            │
                    │        current_proposals     │────────────┘
                    └──────────────────────────────┘
                               │  (after all rounds)
                               ▼
                     Counter(current_proposals).most_common(1)
                               │
                         MAJORITY VOTE
                               │
                    search? ──► oracle ──► recurse
                    else ──► append + return action
```

- **Each debate round**: every agent sees **all other agents' proposals** from the previous round
- Agents **within a round** run in **parallel**; rounds are **sequential**
- Total Copilot calls per step: `N + (rounds × N)` = `N × (1 + rounds)`
- The debate context is injected into a **temporary message list** — not persisted in `self.conversation`

---

## Architecture 4 — Hybrid MAS (`CopilotHybridAgent`)

```
                 ┌──────────────────────────────┐
Observation ───► │        self.conversation      │
                 └─────────────┬────────────────┘
                               │
                               ▼  [sequential]
                    ┌──────────────────────┐
                    │  Copilot Call        │
                    │  ORCHESTRATOR        │  ← SYSTEM_PROMPT + "[Orchestrator] give a directive"
                    │  temp=0.0 (default)  │
                    └──────────┬───────────┘
                               │  directive = "Search for recipe first"
                               ▼
         conversation + [model: directive] + [user: "Workers, propose an action"]
                               │
                    ── WORKER PROPOSALS (parallel) ──
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         Copilot Call     Copilot Call     Copilot Call     (N workers, temp=0.7)
         action[0]        action[1]        action[N-1]
              └────────────────┼────────────────┘
                               │
            worker_msgs + [model: "Worker proposals: [a0, a1, ...]"]
                          + [user: "Manager, decide the final action"]
                               │
                               ▼  [sequential]
                    ┌──────────────────────┐
                    │  Copilot Call        │
                    │  MANAGER / DECIDER   │  ← SYSTEM_PROMPT
                    │  temp=0.0 (default)  │
                    └──────────┬───────────┘
                               │  final_action
                               ▼
                    search? ──► oracle ──► recurse
                    else ──► append + return final_action
```

- **3 phases, 3 Copilot call types**: orchestrator (sequential) → workers (parallel) → manager (sequential)
- Total calls per step: `1 + N + 1`
- **Key difference from Centralized**: workers have **creative freedom** (temp=0.7); the manager picks from real diversity rather than a single worker executing a plan
- **Key difference from Decentralized**: the orchestrator gives a **directive that shapes all worker proposals upfront**, rather than proposals evolving through debate

---

## Side-by-Side Comparison

| | **Independent** | **Centralized** | **Decentralized** | **Hybrid** |
|---|---|---|---|---|
| **Calls/step** | N | 2 | N×(1+rounds) | N+2 |
| **Parallelism** | Full | None | Within-round | Workers only |
| **Diversity source** | temp=0.7 sampling | None | temp=0.7 + debate | temp=0.7 sampling |
| **Consensus mechanism** | Majority vote | None (single worker) | Majority vote after debate | Manager LLM pick |
| **Cross-agent communication** | ❌ | ❌ | ✅ proposals shared | ✅ directive + proposals |
| **search: handling** | Recursive act | Recursive act | Recursive act | Recursive act |
