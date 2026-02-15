
from abc import ABC, abstractmethod
from typing import Optional
from plancraft.environment.env import PlancraftEnvironment
from plancraft.config import PlancraftExample
from .llm import call_gemini_with_retry
from google import genai
import time
import re
from plancraft.environment.search import gold_search_recipe

class BaseAgent(ABC):
    """
    Abstract Base Class for all Plancraft agents.
    """
    def __init__(self, model_name: str, client: genai.Client):
        self.model_name = model_name
        self.client = client
        self.conversation = []  # Stores history
        self.step_count = 0
        self.example_id = None
        self.target = None
        
    def reset(self, example_id: str, target: str):
        """Resets the agent state for a new episode."""
        self.conversation = []
        self.step_count = 0
        self.example_id = example_id
        self.target = target
        
    @abstractmethod
    def act(self, observation_text: str) -> str:
        """
        Receives an observation string and returns an action string.
        """
        pass

    def _oracle_search(self, query: str) -> str:
        """Performs search using the oracle recipe tool."""
        # Simple regex to extract query from "search: <item>"
        match = re.search(r"search:\s*(\S+)", query)
        if match:
             from plancraft.environment.search import gold_search_recipe
             target = match.group(1).strip().lower()
             return gold_search_recipe(target)
        return "No results found."

    def log(self, msg: str):
        print(f"[{self.__class__.__name__}] {msg}")
