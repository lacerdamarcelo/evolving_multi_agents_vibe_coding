"""
Demonstration of the relative food count system.
This script shows how agents now perceive other agents' food counts relative to their own.
"""

import torch
from agent import Agent
from environment import Environment

def demonstrate_relative_food_perception():
    """Demonstrate the relative food count perception system."""
    print("=== Relative Food Count Perception Demonstration ===")
    print()
    
    print("Enhanced Agent Perception:")
    print("• Agents now see other agents' food counts relative to their own")
    print("• Positive values = agent has MORE food than the other")
    print("• Negative values = agent has LESS food than the other")
    print("• Zero values = agents have EQUAL food counts")
    print()
    
    # Create test scenario with agents having different food counts
    agent_a = Agent(100, 100, 0)
    agent_a.food_count = 5
    
    agent_b = Agent(150, 100, 0)
    agent_b.food_count = 2
    
    agent_c = Agent(200, 100, 0)
    agent_c.food_count = 8
    
    agent_d = Agent(250, 100, 0)
    agent_d.food_count = 5  # Same as agent_a
    
    test_agents = [agent_a, agent_b, agent_c, agent_d]
    
    print("Test Scenario:")
    print(f"  Agent A: {agent_a.food_count} food at ({agent_a.x}, {agent_a.y})")
    print(f"  Agent B: {agent_b.food_count} food at ({agent_b.x}, {agent_b.y})")
    print(f"  Agent C: {agent_c.food_count} food at ({agent_c.x}, {agent_c.y})")
    print(f"  Agent D: {agent_d.food_count} food at ({agent_d.x}, {agent_d.y})")
    print()
    
    # Analyze perception from each agent's perspective
    env = Environment()
    
    for i, observer in enumerate(test_agents):
        print(f"=== Agent {chr(65+i)} Perspective (food_count={observer.food_count}) ===")
        
        input_tensor = observer.get_perception_input(test_agents, env.food_points)
        
        # Extract other agents data
        other_agents_start = 2
        other_agents_data = input_tensor[other_agents_start:other_agents_start + 147]
        
        print("Perceived other agents:")
        agent_count = 0
        for j in range(0, len(other_agents_data), 3):
            if j + 2 < len(other_agents_data):
                distance = other_agents_data[j].item()
                angle = other_agents_data[j+1].item()
                relative_food = other_agents_data[j+2].item()
                
                if distance > 0:  # Non-padded entry
                    agent_count += 1
                    status = "ADVANTAGE" if relative_food > 0 else "DISADVANTAGE" if relative_food < 0 else "EQUAL"
                    print(f"  Agent {agent_count}: Distance={distance:.0f}, Angle={angle:.0f}°, "
                          f"Relative_Food={relative_food:+.0f} ({status})")
        print()

def demonstrate_competitive_strategies():
    """Show how relative food count enables competitive strategies."""
    print("="*60)
    print("=== Competitive Strategy Analysis ===")
    print()
    
    # Create environment with varied food distribution
    env = Environment()
    
    # Run simulation to create food count variation
    print("Running simulation to create competitive scenarios...")
    for i in range(30):
        env.update()
    
    stats = env.get_statistics()
    print(f"Simulation results: {stats['living_count']} agents alive, {stats['total_food_consumed']} food consumed")
    print()
    
    # Analyze competitive landscape
    living_agents = [agent for agent in env.agents if agent.alive]
    food_counts = [agent.food_count for agent in living_agents]
    
    if food_counts:
        max_food = max(food_counts)
        min_food = min(food_counts)
        avg_food = sum(food_counts) / len(food_counts)
        
        print(f"Competitive Landscape:")
        print(f"  Food count range: {min_food} to {max_food}")
        print(f"  Average food count: {avg_food:.2f}")
        print()
        
        # Find agents with different competitive positions
        leader = max(living_agents, key=lambda a: a.food_count)
        follower = min(living_agents, key=lambda a: a.food_count)
        
        print("=== Leader's Perspective (Most Food) ===")
        analyze_competitive_position(leader, living_agents, env.food_points)
        
        print("=== Follower's Perspective (Least Food) ===")
        analyze_competitive_position(follower, living_agents, env.food_points)

def analyze_competitive_position(agent, all_agents, food_points):
    """Analyze an agent's competitive position."""
    input_tensor = agent.get_perception_input(all_agents, food_points)
    
    # Extract other agents data
    other_agents_start = 2
    other_agents_data = input_tensor[other_agents_start:other_agents_start + 147]
    
    advantages = 0
    disadvantages = 0
    equals = 0
    
    for i in range(2, len(other_agents_data), 3):
        if other_agents_data[i-2].item() > 0:  # Non-padded entry
            relative_food = other_agents_data[i].item()
            if relative_food > 0:
                advantages += 1
            elif relative_food < 0:
                disadvantages += 1
            else:
                equals += 1
    
    total_perceived = advantages + disadvantages + equals
    
    print(f"Agent food count: {agent.food_count}")
    print(f"Competitive analysis:")
    print(f"  Advantages (agent has more food): {advantages}/{total_perceived} ({advantages/total_perceived*100:.1f}%)")
    print(f"  Disadvantages (agent has less food): {disadvantages}/{total_perceived} ({disadvantages/total_perceived*100:.1f}%)")
    print(f"  Equal positions: {equals}/{total_perceived} ({equals/total_perceived*100:.1f}%)")
    
    if advantages > disadvantages:
        print("  → DOMINANT position: mostly advantageous matchups")
    elif disadvantages > advantages:
        print("  → VULNERABLE position: mostly disadvantageous matchups")
    else:
        print("  → BALANCED position: mixed competitive landscape")
    print()

def demonstrate_evolution_impact():
    """Show how relative food count impacts evolution."""
    print("="*60)
    print("=== Evolution with Competitive Awareness ===")
    print()
    
    from evolution import EvolutionManager
    
    env = Environment()
    evolution_manager = EvolutionManager()
    
    print("Testing evolution with competitive awareness...")
    
    # Run initial simulation
    for i in range(50):
        env.update()
    
    initial_stats = env.get_statistics()
    print(f"Initial generation: {initial_stats['total_food_consumed']} food consumed")
    
    # Evolve with competitive awareness
    new_agents, gen_stats, eval_summary = evolution_manager.evolve_generation_with_evaluation(
        env.agents, num_runs=3, num_iterations=40
    )
    
    print(f"Evolution results:")
    print(f"  Best agent avg food: {eval_summary['best_avg_food_count']:.2f}")
    print(f"  Population avg food: {eval_summary['population_avg_food_count']:.2f}")
    print(f"  Selected {gen_stats['selected_agents_count']} top performers")
    
    print("\n✅ Competitive awareness enables strategic evolution!")
    print("✅ Agents can now evolve threat assessment capabilities!")
    print("✅ Enhanced decision making based on competitive position!")

if __name__ == "__main__":
    demonstrate_relative_food_perception()
    demonstrate_competitive_strategies()
    demonstrate_evolution_impact()
    
    print("\n" + "="*60)
    print("🎉 RELATIVE FOOD COUNT SYSTEM COMPLETE!")
    print("\nKey Enhancements:")
    print("• Competitive awareness of other agents' success")
    print("• Strategic decision making based on relative position")
    print("• Threat assessment and opportunity identification")
    print("• Enhanced evolution of competitive behaviors")
    print("• More realistic agent interactions and strategies")
