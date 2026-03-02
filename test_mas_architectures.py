"""
Mock-based unit tests for the MAS architecture implementations in copilot_multi.py.

These tests verify STRUCTURAL correctness (call counts, flow, parameters) without
requiring a live Copilot CLI connection, by patching call_copilot_with_retry.

Each test reflects the formal definitions from:
  "Towards a Science of Scaling Agent Systems" (Kim et al., 2025, §3.1 / Table 2)

Runs with plain pytest (no pytest-asyncio required) via asyncio.run() wrappers.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client():
    """Return a dummy object that satisfies CopilotBaseAgent.__init__."""
    return object()


MOCK_MODULE = "plancraft.agents.copilot_multi.call_copilot_with_retry"

ACTION = "move: from [I1] to [A1] with quantity 1"
OBS = "Inventory: [I1] oak_planks x4"


# ---------------------------------------------------------------------------
# 1. CopilotIndependentAgent — synthesis_only
# ---------------------------------------------------------------------------

class TestCopilotIndependentAgent:
    """
    C = {(ai, aagg)}, Ω = synthesis_only
    n agents in parallel → 1 synthesis aggregation call.
    Expected calls: n + 1.  NO majority voting.
    """

    NUM_AGENTS = 3

    def test_call_count(self):
        from plancraft.agents.copilot_multi import CopilotIndependentAgent

        side_effects = [f"move: from [I{i+1}] to [A1] with quantity 1" for i in range(self.NUM_AGENTS)]
        side_effects.append(ACTION)  # synthesis call

        async def _run():
            with patch(MOCK_MODULE, new_callable=AsyncMock, side_effect=side_effects) as mock_call:
                agent = CopilotIndependentAgent("m", _make_client(), num_agents=self.NUM_AGENTS)
                agent.reset("T", "stick")
                await agent.act(OBS)
                return mock_call.call_count

        count = asyncio.run(_run())
        assert count == self.NUM_AGENTS + 1, (
            f"Expected {self.NUM_AGENTS + 1} calls (n parallel + 1 synthesis), got {count}"
        )

    def test_no_majority_voting_even_when_all_agree(self):
        """Even when all proposals are identical, synthesis call still runs (n+1 total)."""
        from plancraft.agents.copilot_multi import CopilotIndependentAgent

        async def _run():
            with patch(MOCK_MODULE, new_callable=AsyncMock, return_value=ACTION) as mock_call:
                agent = CopilotIndependentAgent("m", _make_client(), num_agents=self.NUM_AGENTS)
                agent.reset("T", "stick")
                await agent.act(OBS)
                return mock_call.call_count

        count = asyncio.run(_run())
        assert count == self.NUM_AGENTS + 1, (
            f"synthesis_only must still run the aggregation call. Got {count}"
        )

    def test_returns_string(self):
        from plancraft.agents.copilot_multi import CopilotIndependentAgent

        async def _run():
            with patch(MOCK_MODULE, new_callable=AsyncMock, return_value=ACTION):
                agent = CopilotIndependentAgent("m", _make_client(), num_agents=2)
                agent.reset("T", "stick")
                return await agent.act(OBS)

        action = asyncio.run(_run())
        assert isinstance(action, str) and len(action) > 0


# ---------------------------------------------------------------------------
# 2. CopilotCentralizedAgent — hierarchical, r rounds, n workers
# ---------------------------------------------------------------------------

class TestCopilotCentralizedAgent:
    """
    C = {(aorch, ai) : ∀i}, Ω = hierarchical
    Per round: 1 directive + n workers + 1 synthesis = n+2.
    Total: r * (n+2).
    """

    NUM_AGENTS = 3
    ROUNDS = 2

    def test_call_count(self):
        from plancraft.agents.copilot_multi import CopilotCentralizedAgent

        expected = self.ROUNDS * (self.NUM_AGENTS + 2)

        async def _run():
            with patch(MOCK_MODULE, new_callable=AsyncMock, return_value=ACTION) as mock_call:
                agent = CopilotCentralizedAgent(
                    "m", _make_client(), num_agents=self.NUM_AGENTS, rounds=self.ROUNDS
                )
                agent.reset("T", "stick")
                await agent.act(OBS)
                return mock_call.call_count

        count = asyncio.run(_run())
        assert count == expected, (
            f"Expected {expected} calls for r={self.ROUNDS}, n={self.NUM_AGENTS}. Got {count}"
        )

    def test_single_round_default(self):
        """Default rounds=1 → n+2 calls (1 directive + n workers + 1 synthesis)."""
        from plancraft.agents.copilot_multi import CopilotCentralizedAgent

        async def _run():
            with patch(MOCK_MODULE, new_callable=AsyncMock, return_value=ACTION) as mock_call:
                agent = CopilotCentralizedAgent("m", _make_client(), num_agents=self.NUM_AGENTS)
                agent.reset("T", "stick")
                await agent.act(OBS)
                return mock_call.call_count

        count = asyncio.run(_run())
        assert count == self.NUM_AGENTS + 2


# ---------------------------------------------------------------------------
# 3. CopilotDecentralizedAgent — consensus, d debate rounds
# ---------------------------------------------------------------------------

class TestCopilotDecentralizedAgent:
    """
    C = {(ai, aj) : ∀i,j, i≠j}, Ω = consensus
    n initial proposals + d*n debate calls.
    Total: n * (d+1).
    """

    NUM_AGENTS = 3
    ROUNDS = 2

    def test_call_count(self):
        from plancraft.agents.copilot_multi import CopilotDecentralizedAgent

        expected = self.NUM_AGENTS * (1 + self.ROUNDS)

        async def _run():
            with patch(MOCK_MODULE, new_callable=AsyncMock, return_value=ACTION) as mock_call:
                agent = CopilotDecentralizedAgent(
                    "m", _make_client(), num_agents=self.NUM_AGENTS, rounds=self.ROUNDS
                )
                agent.reset("T", "stick")
                await agent.act(OBS)
                return mock_call.call_count

        count = asyncio.run(_run())
        assert count == expected, (
            f"Expected {expected} calls n*(1+d). Got {count}"
        )

    def test_consensus_majority_vote(self):
        """After debate rounds, majority vote determines the final action."""
        from plancraft.agents.copilot_multi import CopilotDecentralizedAgent

        action_a = "move: from [I1] to [A1] with quantity 1"
        action_b = "move: from [I2] to [B1] with quantity 1"
        NUM, D = 3, 1
        # Initial: 3 calls; Debate round 1: 3 calls — 2 return A, 1 returns B
        returns = [action_a, action_a, action_b, action_a, action_a, action_b]

        async def _run():
            with patch(MOCK_MODULE, new_callable=AsyncMock, side_effect=returns):
                agent = CopilotDecentralizedAgent("m", _make_client(), num_agents=NUM, rounds=D)
                agent.reset("T", "stick")
                return await agent.act(OBS)

        action = asyncio.run(_run())
        assert action == action_a


# ---------------------------------------------------------------------------
# 4. CopilotHybridAgent — hierarchical + lateral
# ---------------------------------------------------------------------------

class TestCopilotHybridAgent:
    """
    C = star + peer edges, Ω = hierarchical + lateral
    Per round: 1 directive + n workers + p*n peer + 1 synthesis = n*(p+1)+2.
    Total: r * (n*(p+1)+2).
    """

    NUM_AGENTS = 3
    ROUNDS = 1
    PEER_ROUNDS = 1

    def test_call_count_with_peer_rounds(self):
        from plancraft.agents.copilot_multi import CopilotHybridAgent

        expected = self.ROUNDS * (self.NUM_AGENTS * (self.PEER_ROUNDS + 1) + 2)

        async def _run():
            with patch(MOCK_MODULE, new_callable=AsyncMock, return_value=ACTION) as mock_call:
                agent = CopilotHybridAgent(
                    "m", _make_client(),
                    num_agents=self.NUM_AGENTS,
                    rounds=self.ROUNDS,
                    peer_rounds=self.PEER_ROUNDS,
                )
                agent.reset("T", "stick")
                await agent.act(OBS)
                return mock_call.call_count

        count = asyncio.run(_run())
        assert count == expected, (
            f"Expected {expected} calls "
            f"(r={self.ROUNDS}, n={self.NUM_AGENTS}, p={self.PEER_ROUNDS}). Got {count}"
        )

    def test_zero_peer_rounds_degenerates_to_centralized(self):
        """peer_rounds=0 → no lateral exchange → 1 directive + n workers + 1 synthesis = n+2."""
        from plancraft.agents.copilot_multi import CopilotHybridAgent

        async def _run():
            with patch(MOCK_MODULE, new_callable=AsyncMock, return_value=ACTION) as mock_call:
                agent = CopilotHybridAgent(
                    "m", _make_client(),
                    num_agents=self.NUM_AGENTS, rounds=1, peer_rounds=0,
                )
                agent.reset("T", "stick")
                await agent.act(OBS)
                return mock_call.call_count

        count = asyncio.run(_run())
        assert count == self.NUM_AGENTS + 2
