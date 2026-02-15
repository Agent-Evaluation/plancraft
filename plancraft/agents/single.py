
from .base import BaseAgent
from .llm import call_gemini_with_retry, get_gemini_client
from google import genai
import time
import re

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

class SingleAgent(BaseAgent):
    """
    Standard SAS (Single-Agent System) implementation.
    Maintains a conversation history and calls Gemini for each step.
    """
    def __init__(self, model_name: str, client: genai.Client):
        super().__init__(model_name, client)

    def act(self, observation_text: str) -> str:
        # 1. Update history with latest environment observation
        if observation_text:
            self.conversation.append({"role": "user", "content": observation_text})
            
        # 2. Call Gemini
        messages = [
            genai.types.Content(
                role=msg["role"] if msg["role"] != "assistant" else "model",
                parts=[genai.types.Part(text=msg["content"])],
            )
            for msg in self.conversation
        ]

        action_text = call_gemini_with_retry(
            self.client, 
            self.model_name, 
            messages, 
            SYSTEM_PROMPT
        )
        
        # 3. Add model response to history
        self.conversation.append({"role": "model", "content": action_text})

        # 4. Check for oracle search action
        if "search:" in action_text.lower():
            search_result = self._oracle_search(action_text)
            if search_result:
                self.log(f"🔍 {action_text} -> Found recipe")
                # Feed result back as user message immediately
                self.conversation.append({"role": "user", "content": search_result})
                # Re-prompt model now that it has search info
                # To modify the action, we recurse or loop. 
                # Since the environment expects ONE action, and search is 'free' (doesn't advance env time step usually, 
                # but eval_gemini.py treated search as a step that consumed a turn but didn't call env.step).
                # Actually, eval_gemini.py continued the loop: "continue" -> call Gemini again.
                # So here, we should recurse to get the next action.
                return self.act(None) # Pass None to avoid re-adding observation
        
        return action_text
