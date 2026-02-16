"""Quick smoke test for the Copilot agent implementation."""
import asyncio
import sys


async def test_client_connection():
    """Test 1: Can we connect to the Copilot CLI?"""
    from copilot import CopilotClient
    print("TEST 1: Client connection")
    client = CopilotClient()
    try:
        await client.start()
        state = client.get_state()
        print(f"  State: {state}")
        status = await client.get_status()
        print(f"  Version: {status.version}")
        proto = getattr(status, 'protocol_version', getattr(status, 'protocolVersion', 'unknown'))
        print(f"  Protocol: {proto}")
        ping = await client.ping("test")
        print(f"  Ping: OK")
        print("  RESULT: PASS")
        return client
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        print("  RESULT: FAIL")
        return None


async def test_single_agent(client):
    """Test 2: Can the single agent produce an action?"""
    from plancraft.agents.copilot_single import CopilotSingleAgent
    from plancraft.agents.copilot_llm import DEFAULT_MODEL
    print(f"\nTEST 2: Single agent (model={DEFAULT_MODEL})")
    agent = CopilotSingleAgent(DEFAULT_MODEL, client)
    agent.reset("TEST001", "stick")

    observation = (
        "Craft: stick\n"
        "Inventory:\n"
        "  [I1] oak_planks quantity 4\n"
        "Crafting grid: empty\n"
        "Crafting output [0]: empty"
    )

    try:
        action = await agent.act(observation)
        print(f"  Agent action: {action}")
        print("  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        print("  RESULT: FAIL")
        return False


async def test_full_episode(client):
    """Test 3: Run a single episode end-to-end."""
    from plancraft.simple import PlancraftGymWrapper, get_plancraft_examples
    from plancraft.environment.actions import (
        MoveActionHandler,
        SmeltActionHandler,
        ImpossibleActionHandler,
    )
    from plancraft.agents.copilot_single import CopilotSingleAgent
    from plancraft.agents.copilot_llm import DEFAULT_MODEL

    print(f"\nTEST 3: Full episode (1 example from val.small)")
    examples = get_plancraft_examples(split="val.small")
    example = examples[0]
    print(f"  Example: {example.id}")
    print(f"  Target: {example.target}")
    print(f"  Impossible: {example.impossible}")

    env = PlancraftGymWrapper(
        example=example,
        actions=[MoveActionHandler(), SmeltActionHandler(), ImpossibleActionHandler()],
        max_steps=10,
        use_text_inventory=True,
    )
    agent = CopilotSingleAgent(DEFAULT_MODEL, client)
    agent.reset(example.id, example.target)

    observation, reward, terminated, truncated, info = env.step("")
    step = 0
    try:
        while not (terminated or truncated) and step < 10:
            step += 1
            action = await agent.act(observation["text"])
            print(f"  Step {step}: {action}")
            observation, reward, terminated, truncated, info = env.step(action)

        success = env.success
        reason = info.get("reason", "unknown")
        print(f"  Success: {success} | Reason: {reason} | Steps: {step}")
        print(f"  RESULT: {'PASS' if success else 'COMPLETED (not solved)'}")
        return True
    except Exception as e:
        print(f"  ERROR at step {step}: {type(e).__name__}: {e}")
        print("  RESULT: FAIL")
        return False


async def main():
    print("=" * 60)
    print("COPILOT AGENT SMOKE TEST")
    print("=" * 60)

    # Test 1: Connection
    client = await test_client_connection()
    if not client:
        print("\nClient connection failed. Cannot proceed.")
        sys.exit(1)

    try:
        # Test 2: Single agent action
        ok = await test_single_agent(client)
        if not ok:
            print("\nSingle agent test failed. Skipping full episode.")
            sys.exit(1)

        # Test 3: Full episode
        await test_full_episode(client)

    finally:
        print("\nCleaning up...")
        errors = await client.stop()
        if errors:
            print(f"  Stop errors: {errors}")
        else:
            print("  Client stopped cleanly.")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
