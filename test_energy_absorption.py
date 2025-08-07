"""
Test script to verify that agents absorb energy when killing other agents.
"""

from environment import Environment
from agent import Agent
from config import ENERGY_FROM_FOOD, INITIAL_ENERGY

def test_energy_absorption():
    """Test that agents absorb energy when killing other agents."""
    print("Testing energy absorption feature...")
    
    # Create a simple environment
    env = Environment()
    
    # Create two test agents with different food counts
    agent1 = Agent(x=100, y=100, orientation=0)
    agent2 = Agent(x=105, y=100, orientation=0)  # Close enough to collide
    
    # Set up the scenario: agent1 has more food, so it should win
    agent1.food_count = 5
    agent1.energy = 500
    agent2.food_count = 3
    agent2.energy = 300
    
    print(f"Before collision:")
    print(f"Agent1 - Food: {agent1.food_count}, Energy: {agent1.energy}, Alive: {agent1.alive}")
    print(f"Agent2 - Food: {agent2.food_count}, Energy: {agent2.energy}, Alive: {agent2.alive}")
    
    # Expected energy gain for agent1
    expected_energy_gain = agent2.food_count * ENERGY_FROM_FOOD
    expected_final_energy = agent1.energy + expected_energy_gain
    expected_final_food = agent1.food_count + agent2.food_count
    
    print(f"\nExpected results:")
    print(f"Agent1 should gain {expected_energy_gain} energy (3 food × {ENERGY_FROM_FOOD} energy/food)")
    print(f"Agent1 final energy should be: {expected_final_energy}")
    print(f"Agent1 final food count should be: {expected_final_food}")
    
    # Simulate collision
    env._resolve_agent_collision(agent1, agent2)
    
    print(f"\nAfter collision:")
    print(f"Agent1 - Food: {agent1.food_count}, Energy: {agent1.energy}, Alive: {agent1.alive}")
    print(f"Agent2 - Food: {agent2.food_count}, Energy: {agent2.energy}, Alive: {agent2.alive}")
    
    # Verify results
    success = True
    if agent1.energy != expected_final_energy:
        print(f"❌ ERROR: Agent1 energy is {agent1.energy}, expected {expected_final_energy}")
        success = False
    
    if agent1.food_count != expected_final_food:
        print(f"❌ ERROR: Agent1 food count is {agent1.food_count}, expected {expected_final_food}")
        success = False
    
    if agent2.alive:
        print(f"❌ ERROR: Agent2 should be dead but is still alive")
        success = False
    
    if success:
        print("✅ Energy absorption test PASSED!")
    else:
        print("❌ Energy absorption test FAILED!")
    
    return success

def test_tie_scenario():
    """Test that both agents die in a tie scenario."""
    print("\n" + "="*50)
    print("Testing tie scenario...")
    
    # Create two agents with equal food counts
    agent1 = Agent(x=200, y=200, orientation=0)
    agent2 = Agent(x=205, y=200, orientation=0)
    
    agent1.food_count = 4
    agent1.energy = 400
    agent2.food_count = 4
    agent2.energy = 350
    
    print(f"Before collision (tie scenario):")
    print(f"Agent1 - Food: {agent1.food_count}, Energy: {agent1.energy}, Alive: {agent1.alive}")
    print(f"Agent2 - Food: {agent2.food_count}, Energy: {agent2.energy}, Alive: {agent2.alive}")
    
    # Create environment and simulate collision
    env = Environment()
    env._resolve_agent_collision(agent1, agent2)
    
    print(f"\nAfter collision:")
    print(f"Agent1 - Food: {agent1.food_count}, Energy: {agent1.energy}, Alive: {agent1.alive}")
    print(f"Agent2 - Food: {agent2.food_count}, Energy: {agent2.energy}, Alive: {agent2.alive}")
    
    # Verify both agents are dead
    if not agent1.alive and not agent2.alive:
        print("✅ Tie scenario test PASSED! Both agents died as expected.")
        return True
    else:
        print("❌ Tie scenario test FAILED! Both agents should be dead.")
        return False

if __name__ == "__main__":
    test1_passed = test_energy_absorption()
    test2_passed = test_tie_scenario()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Energy absorption test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Tie scenario test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("🎉 All tests PASSED! Energy absorption feature is working correctly.")
    else:
        print("⚠️  Some tests FAILED. Please check the implementation.")
