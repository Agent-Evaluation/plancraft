# MassGen Integration Plan — Concerns & Discussion

## 🔴 Major: Architectural Mismatch

The plan treats MassGen as a simple LLM replacement, but it's fundamentally different. Looking at `massgen_config.yaml`, MassGen is a **multi-agent orchestration framework** — it runs 3 agents (agent_a, agent_b, agent_c), has a voting system, task planning, round-based orchestration, and its own internal conversation management.

The plan proposes using `massgen.run()` as a drop-in replacement for `call_copilot_with_retry()`, sending the full prompt as a `query`. This creates a **nested multi-agent problem**:

1. The **Plancraft agent** maintains a conversation loop, calling the LLM once per step
2. **MassGen** internally runs 3 agents that discuss, plan, and vote on the answer

**Key questions:**
- Is this intentional? Do we *want* MassGen's full multi-agent orchestration for each Plancraft step? That means 3 agents internally deliberating on each `move: from [I1] to [A1] with quantity 1` action.
- The config timeouts (`initial_round_timeout_seconds: 1200` = 20 minutes) suggest each `act()` call could take **minutes**, far longer than a single Copilot SDK call.
- Would it be simpler to use MassGen in **single-agent mode** (`model="gpt-5-mini"` without a config file)?

## 🟡 Prompt Format Concern

`_build_prompt()` formats the conversation into a single flat string (`[System Instructions]`, `[Observation]`, `[Your Previous Action]`). This was necessary because the Copilot SDK's `session.send_and_wait()` takes a single prompt string.

`massgen.run()` also takes a single `query` string, but it's designed for **task-level queries**, not turn-by-turn conversation. Passing a growing conversation history as a giant flat string may degrade MassGen's performance as episodes grow longer.

## 🟡 No Multi-Architecture Support

The existing `eval_copilot.py` supports 5 architectures (single, independent, centralized, decentralized, hybrid) via `copilot_multi.py`. The plan only replaces the **single** agent. The multi-agent architectures are not addressed.

## 🟢 Good Decisions

- **`enable_filesystem=False`** — correct for a sandboxed Plancraft agent
- **4 new files, no modifications** — clean separation, easy to revert
- **Reusing `_build_prompt()`** — good code reuse

## Open Questions

1. **Do we want MassGen's full 3-agent orchestration for every Plancraft step?** Or would a single-model `massgen.run(query=..., model="gpt-5-mini")` call be more appropriate?
2. **Are we only targeting the single-agent architecture**, or do we eventually want MassGen multi-agent variants too?
3. **Has the `massgen.run()` API signature been verified?** The plan includes it as a reference, but it should be confirmed against the installed version.
