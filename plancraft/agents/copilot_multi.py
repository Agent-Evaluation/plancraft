
import asyncio
import logging
from collections import Counter
from copilot import CopilotClient
from .copilot_base import CopilotBaseAgent
from .copilot_llm import call_copilot_with_retry

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


class CopilotIndependentAgent(CopilotBaseAgent):
    """
    Independent MAS: N agents run in parallel via Copilot SDK.
    Majority Voting at each step.
    """
    def __init__(self, model_name: str, client: CopilotClient, num_agents: int = 2):
        super().__init__(model_name, client)
        self.num_agents = num_agents

    async def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        # Run all agents in parallel using asyncio.gather (like MassGen)
        async def _call_agent(i: int) -> str | None:
            try:
                return await call_copilot_with_retry(
                    self.client,
                    self.model_name,
                    self.conversation,
                    SYSTEM_PROMPT,
                    temperature=0.7,
                )
            except Exception as e:
                self.log(f"Agent {i} failed: {e}")
                return None

        results = await asyncio.gather(
            *[_call_agent(i) for i in range(self.num_agents)]
        )
        actions = [r for r in results if r is not None]

        if not actions:
            return "impossible: All agents failed"

        counts = Counter(actions)
        best_action, count = counts.most_common(1)[0]
        self.log(f"Votes: {dict(counts)} -> Winner: {best_action}")

        if "search:" in best_action.lower():
            search_result = self._oracle_search(best_action)
            if search_result:
                self.conversation.append({"role": "model", "content": best_action})
                self.conversation.append({"role": "user", "content": search_result})
                return await self.act(None)

        self.conversation.append({"role": "model", "content": best_action})
        return best_action


class CopilotCentralizedAgent(CopilotBaseAgent):
    """
    Centralized MAS: Orchestrator + Workers via Copilot SDK.
    """
    def __init__(self, model_name: str, client: CopilotClient):
        super().__init__(model_name, client)

    async def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        # Step 1: Orchestrator plans
        planning_prompt = (
            SYSTEM_PROMPT +
            "\n\n[Internal Step] First, briefly analyze what is needed. "
            "Do NOT output the action yet. Just output a 1-sentence plan."
        )

        plan = await call_copilot_with_retry(
            self.client, self.model_name, self.conversation, planning_prompt
        )
        self.log(f"Orchestrator Plan: {plan}")

        # Step 2: Worker executes based on plan
        turn_messages = self.conversation + [
            {"role": "model", "content": plan},
            {"role": "user", "content": "Based on this plan, what is the exact action?"},
        ]

        action = await call_copilot_with_retry(
            self.client, self.model_name, turn_messages, SYSTEM_PROMPT
        )

        if "search:" in action.lower():
            search_result = self._oracle_search(action)
            if search_result:
                self.conversation.append({"role": "model", "content": action})
                self.conversation.append({"role": "user", "content": search_result})
                return await self.act(None)

        self.conversation.append({"role": "model", "content": action})
        return action


class CopilotDecentralizedAgent(CopilotBaseAgent):
    """
    Decentralized MAS: Agents debate via Copilot SDK.
    """
    def __init__(self, model_name: str, client: CopilotClient, num_agents: int = 2, rounds: int = 2):
        super().__init__(model_name, client)
        self.num_agents = num_agents
        self.rounds = rounds

    async def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        # Initial Proposals — all agents in parallel
        initial_tasks = [
            call_copilot_with_retry(
                self.client, self.model_name, self.conversation, SYSTEM_PROMPT, temperature=0.7
            )
            for _ in range(self.num_agents)
        ]
        proposals = list(await asyncio.gather(*initial_tasks))

        # Debate Rounds — agents within each round run in parallel
        current_proposals = proposals
        for r in range(self.rounds):
            self.log(f"Round {r} Proposals: {current_proposals}")

            debate_context = "Other agents proposed:\n" + "\n".join(
                [f"- Agent {j}: {p}" for j, p in enumerate(current_proposals)]
            )
            debate_context += "\nGiven these proposals, what is your updated action?"

            debate_tasks = []
            for i in range(self.num_agents):
                debate_msgs = self.conversation + [
                    {"role": "user", "content": debate_context}
                ]
                debate_tasks.append(
                    call_copilot_with_retry(
                        self.client, self.model_name, debate_msgs, SYSTEM_PROMPT, temperature=0.7
                    )
                )
            current_proposals = list(await asyncio.gather(*debate_tasks))

        # Final Vote
        counts = Counter(current_proposals)
        best_action, _ = counts.most_common(1)[0]

        if "search:" in best_action.lower():
            search_result = self._oracle_search(best_action)
            if search_result:
                self.conversation.append({"role": "model", "content": best_action})
                self.conversation.append({"role": "user", "content": search_result})
                return await self.act(None)

        self.conversation.append({"role": "model", "content": best_action})
        return best_action


class CopilotHybridAgent(CopilotBaseAgent):
    """
    Hybrid MAS: Orchestrator + Decentralized Workers via Copilot SDK.
    """
    def __init__(self, model_name: str, client: CopilotClient, num_agents: int = 2):
        super().__init__(model_name, client)
        self.num_agents = num_agents

    async def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        # Step 1: Orchestrator Directive (must be sequential — workers depend on it)
        directive_prompt = (
            SYSTEM_PROMPT +
            "\n\n[Orchestrator] Briefly analyze the situation and give a directive to your workers."
        )
        directive = await call_copilot_with_retry(
            self.client, self.model_name, self.conversation, directive_prompt
        )
        self.log(f"Manager Directive: {directive}")

        # Step 2: Workers propose in parallel
        worker_msgs = self.conversation + [
            {"role": "model", "content": directive},
            {"role": "user", "content": "Workers, propose an action based on directive."},
        ]

        worker_tasks = [
            call_copilot_with_retry(
                self.client, self.model_name, worker_msgs, SYSTEM_PROMPT, temperature=0.7
            )
            for _ in range(self.num_agents)
        ]
        actions = list(await asyncio.gather(*worker_tasks))

        # Step 3: Manager Decision
        final_msgs = worker_msgs + [
            {"role": "model", "content": f"Worker proposals: {actions}"},
            {"role": "user", "content": "Manager, decide the final action."},
        ]

        final_action = await call_copilot_with_retry(
            self.client, self.model_name, final_msgs, SYSTEM_PROMPT
        )

        if "search:" in final_action.lower():
            search_result = self._oracle_search(final_action)
            if search_result:
                self.conversation.append({"role": "model", "content": final_action})
                self.conversation.append({"role": "user", "content": search_result})
                return await self.act(None)

        self.conversation.append({"role": "model", "content": final_action})
        return final_action
