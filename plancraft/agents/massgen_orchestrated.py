"""
MassGen orchestrated Plancraft agent.

This agent keeps Plancraft's step-wise `act()` API while delegating each action
selection to a **persistent** MassGen Orchestrator session that lives for the
duration of one evaluation example.

Architectural change (memory-leak fix):
  Previously each `act()` call went through the stateless `litellm.completion`
  layer which cold-booted the entire multi-agent orchestration infrastructure
  on every single turn. Now a single `Orchestrator` is created in `reset()`
  and reused across all turns of the same example via `chat_simple()`.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Optional

from plancraft.environment.search import gold_search_recipe

SYSTEM_PROMPT = """\
You are a Minecraft crafting agent. Your goal is to craft a target item by \
manipulating items in your inventory using a 3×3 crafting grid.

## Actions
Respond with EXACTLY ONE action per turn (no extra text):

1. **move** – move items between slots
   `move: from [Source] to [Target] with quantity N`

2. **smelt** – smelt an item (e.g. ores → ingots)
   `smelt: from [Source] to [Target] with quantity N`

3. **impossible** – declare the task impossible with current inventory
   `impossible: <reason>`

4. **search** – look up how to craft an item
   `search: <item_name>`

## Slot names
- Crafting grid (3×3):
    [A1] [A2] [A3]
    [B1] [B2] [B3]
    [C1] [C2] [C3]
- Crafting output: [0]  (crafted items appear here)
- Inventory: [I1] through [I36]

## Crafting rules
1. Place raw materials from your inventory INTO the crafting grid slots \
([A1]–[C3]) in the correct pattern.
2. When the pattern is correct the result appears in slot [0].
3. Move the result from [0] to any inventory slot [I1]–[I36] to collect it.
4. Shaped recipes require items in specific grid positions.
5. Shapeless recipes can be placed in any grid slots.
6. Smelting uses the `smelt` action directly — no grid needed.

## Strategy
- First, if you don't know the recipe, use `search: <target_item>` to look it up.
- Then place the required items in the crafting grid.
- Finally, move the crafted item from [0] to an inventory slot.
- If the inventory lacks the required materials, declare `impossible: <reason>`.
- You may need multi-step crafting (e.g. logs → planks → sticks).

## Important
- Respond with ONLY the action, nothing else.
- Use exact slot names like [I1], [A1], [0], etc.
- Quantities must be between 1 and 64.
"""


def _create_orchestrator(
    config_path: str,
    system_prompt: str,
    enable_filesystem: bool = False,
) -> Any:
    """Create a persistent MassGen Orchestrator from a YAML config file.

    This is called once per evaluation example in ``reset()`` so the heavy
    agent-creation work happens only once instead of on every turn.
    """
    from massgen import Orchestrator
    from massgen.agent_config import AgentConfig
    from massgen.cli import create_agents_from_config, load_config_file

    config_dict, _raw = load_config_file(config_path)
    orchestrator_cfg = config_dict.get("orchestrator", {})

    # Disable filesystem/MCP tools for lightweight benchmark agents
    if not enable_filesystem:
        for agent_def in config_dict.get("agents", []):
            backend = agent_def.get("backend", {})
            backend["exclude_file_operation_mcps"] = True

    agents = create_agents_from_config(config_dict, orchestrator_cfg)
    if not agents:
        raise RuntimeError("No agents created from config")

    # Inject system prompt into every agent
    for agent in agents.values():
        if hasattr(agent, "custom_system_instruction"):
            agent.custom_system_instruction = system_prompt

    agent_config = AgentConfig.from_dict(orchestrator_cfg) if orchestrator_cfg else None

    orchestrator = Orchestrator(
        agents=agents,
        config=agent_config,
        snapshot_storage=orchestrator_cfg.get("snapshot_storage"),
        agent_temporary_workspace=orchestrator_cfg.get("agent_temporary_workspace"),
    )
    return orchestrator


class MassGenOrchestratedAgent:
    """Single Plancraft interface backed by a persistent MassGen orchestrator."""

    def __init__(
        self,
        config_path: str,
        heartbeat_seconds: int = 30,
        event_logger: Optional[Callable[[dict], None]] = None,
    ):
        self.config_path = config_path
        self.heartbeat_seconds = heartbeat_seconds
        self.event_logger = event_logger

        self.step_count = 0
        self.example_id: Optional[str] = None
        self.target: Optional[str] = None

        self.last_massgen_metadata: dict = {}
        self.last_call_elapsed_seconds: float = 0.0

        self._orchestrator: Any = None

    def reset(self, example_id: str, target: str) -> None:
        # Tear down previous orchestrator (let GC reclaim)
        self._orchestrator = None

        self.step_count = 0
        self.example_id = example_id
        self.target = target
        self.last_massgen_metadata = {}
        self.last_call_elapsed_seconds = 0.0

        self._orchestrator = _create_orchestrator(
            config_path=self.config_path,
            system_prompt=SYSTEM_PROMPT,
            enable_filesystem=False,
        )

    # ------------------------------------------------------------------
    # Core turn logic
    # ------------------------------------------------------------------

    async def act(self, observation_text: Optional[str]) -> str:
        self.step_count += 1
        turn_context = {
            "example_id": self.example_id,
            "target": self.target,
            "agent_turn": self.step_count,
        }
        self._log_event({"event": "agent_turn_start", **turn_context})

        prompt = observation_text or ""
        started_at = time.time()

        # Stream response from the persistent orchestrator
        action_text = await self._chat_and_collect(prompt)

        elapsed = time.time() - started_at
        self.last_call_elapsed_seconds = elapsed

        # Retrieve coordination metadata from the live orchestrator
        metadata: dict[str, Any] = {}
        try:
            coord = self._orchestrator.get_coordination_result()
            metadata = {
                "massgen_selected_agent": coord.get("selected_agent"),
                "massgen_vote_results": coord.get("vote_results"),
                "massgen_session_id": self._orchestrator.get_session_id()
                if hasattr(self._orchestrator, "get_session_id")
                else None,
                "massgen_log_directory": coord.get("log_directory"),
            }
        except Exception:
            pass

        self.last_massgen_metadata = metadata

        print(f"[MassGenOrchestratedAgent] MODEL RESPONSE: {action_text}")
        if metadata:
            compact = {
                "selected_agent": metadata.get("massgen_selected_agent"),
                "vote_results": metadata.get("massgen_vote_results"),
                "session_id": metadata.get("massgen_session_id"),
                "log_directory": metadata.get("massgen_log_directory"),
            }
            print(
                "[MassGenOrchestratedAgent] METADATA: "
                f"{json.dumps(compact, ensure_ascii=True, default=str)}"
            )

        if "search:" in action_text.lower():
            search_result = self._oracle_search(action_text)
            if search_result:
                self.log(f"search: {action_text} -> Found recipe")
                self._log_event(
                    {
                        "event": "oracle_search_injected",
                        "query_action": action_text,
                        "recipe_chars": len(search_result),
                        **turn_context,
                    }
                )
                return await self.act(search_result)

        self._log_event(
            {
                "event": "agent_turn_complete",
                "elapsed_seconds": round(elapsed, 3),
                "action_text": action_text,
                **turn_context,
            }
        )
        return action_text

    async def _chat_and_collect(self, user_message: str) -> str:
        """Send *user_message* to the persistent orchestrator and collect the
        full response text from the stream."""
        parts: list[str] = []
        async for chunk in self._orchestrator.chat_simple(user_message):
            if chunk.content:
                parts.append(chunk.content)
        text = "".join(parts).strip()
        if not text:
            raise ValueError("Empty response from MassGen orchestrator")
        return text

    def _oracle_search(self, query: str) -> str:
        match = re.search(r"search:\s*(\S+)", query)
        if match:
            target = match.group(1).strip().lower()
            return gold_search_recipe(target)
        return "No results found."

    def _log_event(self, event: dict) -> None:
        if self.event_logger:
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **event,
            }
            self.event_logger(payload)

    def log(self, msg: str) -> None:
        print(f"[{self.__class__.__name__}] {msg}")
