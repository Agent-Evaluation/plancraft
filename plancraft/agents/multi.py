
import logging
from collections import Counter
from typing import List
from google import genai
from .base import BaseAgent
from .llm import call_gemini_with_retry

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

class IndependentAgent(BaseAgent):
    """
    Independent MAS: N agents run in parallel.
    For sequential tasks, we implement this as Majority Voting at each step.
    Each agent proposes an action, and the most common action is taken.
    """
    def __init__(self, model_name: str, client: genai.Client, num_agents: int = 3):
        super().__init__(model_name, client)
        self.num_agents = num_agents
        # Each agent effectively sees the same shared history of the *consensus* actions,
        # but we query the model N times (potentially with temperature > 0 to induce diversity).

    def act(self, observation_text: str) -> str:
        # Update shared history
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        # Query N times
        # We need slightly higher temperature to get diversity
        messages = [
            genai.types.Content(
                role=msg["role"] if msg["role"] != "assistant" else "model",
                parts=[genai.types.Part(text=msg["content"])],
            )
            for msg in self.conversation
        ]

        actions = []
        for i in range(self.num_agents):
            try:
                # Use call_gemini_with_retry but ideally we want parallel calls.
                # For simplicity, sequential calls for now.
                # Use temp=0.7 for diversity
                action = call_gemini_with_retry(
                    self.client, 
                    self.model_name, 
                    messages, 
                    SYSTEM_PROMPT,
                    temperature=0.7 
                )
                actions.append(action)
            except Exception as e:
                self.log(f"Agent {i} failed: {e}")

        if not actions:
            return "impossible: All agents failed"

        # Majority Vote
        counts = Counter(actions)
        best_action, count = counts.most_common(1)[0]
        
        self.log(f"Votes: {dict(counts)} -> Winner: {best_action}")

        # Check for search
        if "search:" in best_action.lower():
            search_result = self._oracle_search(best_action)
            if search_result:
                self.conversation.append({"role": "model", "content": best_action})
                self.conversation.append({"role": "user", "content": search_result})
                return self.act(None) # Recurse

        self.conversation.append({"role": "model", "content": best_action})
        return best_action


class CentralizedAgent(BaseAgent):
    """
    Centralized MAS: Orchestrator + Workers.
    Pattern:
    1. Orchestrator analyzes situation and prompts workers.
    2. Workers provide input.
    3. Orchestrator decides final action.
    """
    def __init__(self, model_name: str, client: genai.Client):
        super().__init__(model_name, client)
        
    def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        # Step 1: Orchestrator Thought - "What do I need to know?"
        # We simulate this by asking the model to "Plan" first.
        messages = [
             genai.types.Content(
                role=msg["role"] if msg["role"] != "assistant" else "model",
                parts=[genai.types.Part(text=msg["content"])],
            )
            for msg in self.conversation
        ]
        
        # Inject an internal thought step
        planning_prompt = (
            SYSTEM_PROMPT + 
            "\n\n[Internal Step] First, briefly analyze what is needed. "
            "Do NOT output the action yet. Just output a 1-sentence plan."
        )
        
        plan = call_gemini_with_retry(self.client, self.model_name, messages, planning_prompt)
        self.log(f"Orchestrator Plan: {plan}")
        
        # Step 2: "Workers" - We can simulate workers by prompting the model to 
        # "Critique this plan" or "Verify inventory".
        # For simplicity, let's treat the plan as the 'Orchestrator's' guidance
        # and then ask the 'Executioner' (same model, different role) to output the action.
        
        # Add plan to context temporarily (we don't want to clutter the main conversation history 
        # passed to the environment with internal thoughts, but we do want the model to see it)
        # Actually, we should keep internal thoughts in self.conversation? 
        # No, the environment doesn't see them. The model does.
        
        # We will extend the current messages for this turn only.
        turn_messages = messages + [
             genai.types.Content(role="model", parts=[genai.types.Part(text=plan)]),
             genai.types.Content(role="user", parts=[genai.types.Part(text="Based on this plan, what is the exact action?")])
        ]
        
        action = call_gemini_with_retry(self.client, self.model_name, turn_messages, SYSTEM_PROMPT)
        
        # Handle search
        if "search:" in action.lower():
            search_result = self._oracle_search(action)
            if search_result:
                self.conversation.append({"role": "model", "content": action}) # Add search action
                self.conversation.append({"role": "user", "content": search_result}) # Add result
                return self.act(None)

        self.conversation.append({"role": "model", "content": action})
        return action

class DecentralizedAgent(BaseAgent):
    """
    Decentralized MAS: Agents debate.
    """
    def __init__(self, model_name: str, client: genai.Client, num_agents: int = 3, rounds: int = 2):
        super().__init__(model_name, client)
        self.num_agents = num_agents
        self.rounds = rounds

    def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})
            
        messages = [
            genai.types.Content(
                role=msg["role"] if msg["role"] != "assistant" else "model",
                parts=[genai.types.Part(text=msg["content"])],
            )
            for msg in self.conversation
        ]

        # Initial Proposals
        proposals = []
        for i in range(self.num_agents):
            prop = call_gemini_with_retry(self.client, self.model_name, messages, SYSTEM_PROMPT, temperature=0.7)
            proposals.append(prop)
            
        # Debate Rounds
        current_proposals = proposals
        for r in range(self.rounds):
            self.log(f"Round {r} Proposals: {current_proposals}")
            new_proposals = []
            
            # Construct a "Debate Context"
            debate_context = "Other agents proposed:\n" + "\n".join([f"- Agent {j}: {p}" for j, p in enumerate(current_proposals)])
            debate_context += "\nGiven these proposals, what is your updated action?"
            
            # Each agent sees the others' proposals
            for i in range(self.num_agents):
                # We append the debate context to the history
                debate_msgs = messages + [
                    genai.types.Content(role="user", parts=[genai.types.Part(text=debate_context)])
                ]
                new_prop = call_gemini_with_retry(self.client, self.model_name, debate_msgs, SYSTEM_PROMPT, temperature=0.7)
                new_proposals.append(new_prop)
            current_proposals = new_proposals

        # Final Vote
        counts = Counter(current_proposals)
        best_action, _ = counts.most_common(1)[0]
        
        if "search:" in best_action.lower():
            search_result = self._oracle_search(best_action)
            if search_result:
                self.conversation.append({"role": "model", "content": best_action})
                self.conversation.append({"role": "user", "content": search_result})
                return self.act(None)

        self.conversation.append({"role": "model", "content": best_action})
        return best_action

# HybridAgent omitted for brevity, similar to Centralized but with peer steps.

class HybridAgent(BaseAgent):
    """
    Hybrid MAS: Orchestrator + Decentralized Workers.
    1. Orchestrator delegates task.
    2. Workers debate.
    3. Orchestrator aggregates final decision.
    """
    def __init__(self, model_name: str, client: genai.Client, num_agents: int = 3):
        super().__init__(model_name, client)
        self.num_agents = num_agents

    def act(self, observation_text: str) -> str:
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})

        messages = [
            genai.types.Content(
                role=msg["role"] if msg["role"] != "assistant" else "model",
                parts=[genai.types.Part(text=msg["content"])],
            )
            for msg in self.conversation
        ]

        # Step 1: Orchestrator Directive
        directive_prompt = (
            SYSTEM_PROMPT + 
            "\n\n[Orchestrator] Briefly analyze the situation and give a directive to your workers."
        )
        directive = call_gemini_with_retry(self.client, self.model_name, messages, directive_prompt)
        self.log(f"Manager Directive: {directive}")

        # Step 2: Worker Debate (1 Round)
        worker_prompts = messages + [
             genai.types.Content(role="model", parts=[genai.types.Part(text=directive)]),
             genai.types.Content(role="user", parts=[genai.types.Part(text="Workers, propose an action based on directive.")])
        ]
        
        actions = []
        for i in range(self.num_agents):
             act = call_gemini_with_retry(self.client, self.model_name, worker_prompts, SYSTEM_PROMPT, temperature=0.7)
             actions.append(act)
        
        # Step 3: Manager Decision
        final_prompt = worker_prompts + [
             genai.types.Content(role="model", parts=[genai.types.Part(text=f"Worker proposals: {actions}")]),
             genai.types.Content(role="user", parts=[genai.types.Part(text="Manager, decide the final action.")])
        ]
        
        final_action = call_gemini_with_retry(self.client, self.model_name, final_prompt, SYSTEM_PROMPT)
        
        if "search:" in final_action.lower():
            search_result = self._oracle_search(final_action)
            if search_result:
                self.conversation.append({"role": "model", "content": final_action})
                self.conversation.append({"role": "user", "content": search_result})
                return self.act(None)

        self.conversation.append({"role": "model", "content": final_action})
        return final_action
