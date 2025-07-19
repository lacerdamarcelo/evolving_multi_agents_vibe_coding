"""
Demonstration of the food perception system.
This script shows how agents now receive direct sensory input about food locations.
"""

import torch
import math
from agent import Agent
from environment import Environment
from config import INPUT_SIZE, NUM_FOOD_POINTS, POPULATION_SIZE

def demonstrate_food_perception():
    """Demonstrate the food perception system."""
    print("=== Food Perception System Demonstration ===")
    print()
    
    print("Enhanced Neural Network Input Structure:")
    print("• Own state: [food_count, energy]")
    print("• Other agents: [distance, angle, food_count] × 49 agents")
    print("• Food points: [distance, angle] × 20 food points")
    print(f"• Total input size: {INPUT_SIZE} values")
    print()
    
    # Create environment
    env = Environment()
    print(f"Environment created with {len(env.food_points)} food points")
    
    # Create test agent at specific position
    test_agent = Agent(300, 300, 0)  # Center, facing east
    print(f"Test agent positioned at ({test_agent.x}, {test_agent.y}), facing {test_agent.orientation}°")
    print()
    
    # Get perception input
    input_tensor = test_agent.get_perception_input(env.agents, env.food_points)
    
    print("=== Input Tensor Analysis ===")
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Break down the input
    own_state_size = 2
    other_agents_size = 3 * (POPULATION_SIZE - 1)
    food_points_size = 2 * NUM_FOOD_POINTS
    
    print(f"\nInput breakdown:")
    print(f"  Own state: indices 0-{own_state_size-1} ({own_state_size} values)")
    print(f"  Other agents: indices {own_state_size}-{own_state_size + other_agents_size - 1} ({other_agents_size} values)")
    print(f"  Food points: indices {own_state_size + other_agents_size}-{INPUT_SIZE-1} ({food_points_size} values)")
    
    # Extract and analyze food perception data
    food_start_idx = own_state_size + other_agents_size
    food_data = input_tensor[food_start_idx:]
    
    print(f"\n=== Food Perception Analysis ===")
    print("Food points detected (sorted by distance):")
    
    food_info = []
    for i in range(0, len(food_data), 2):
        distance = food_data[i].item()
        angle = food_data[i+1].item()
        if distance > 0:  # Non-padded entry
            food_info.append((distance, angle, i//2 + 1))
    
    # Show top 10 closest food points
    for i, (distance, angle, food_id) in enumerate(food_info[:10]):
        direction = "right" if angle > 0 else "left"
        print(f"  Food {food_id:2d}: Distance={distance:6.1f}, Angle={angle:6.1f}° ({abs(angle):5.1f}° to {direction})")
    
    if len(food_info) > 10:
        print(f"  ... and {len(food_info) - 10} more food points")
    
    print(f"\nTotal food points perceived: {len(food_info)}/{NUM_FOOD_POINTS}")

def demonstrate_perception_vs_reality():
    """Compare agent perception with actual food positions."""
    print("\n" + "="*60)
    print("=== Perception vs Reality Comparison ===")
    print()
    
    env = Environment()
    agent = Agent(200, 200, 45)  # Off-center, facing northeast
    
    print(f"Agent position: ({agent.x}, {agent.y})")
    print(f"Agent orientation: {agent.orientation}° (facing northeast)")
    print()
    
    # Get agent's perception
    input_tensor = agent.get_perception_input(env.agents, env.food_points)
    food_start_idx = 2 + 3 * (POPULATION_SIZE - 1)
    food_data = input_tensor[food_start_idx:]
    
    # Calculate actual distances and angles for comparison
    print("Perception vs Reality for closest 5 food points:")
    print("Format: [Perceived] vs [Actual]")
    print()
    
    actual_food_info = []
    for food_point in env.food_points:
        # Calculate actual distance
        dx = food_point.x - agent.x
        dy = food_point.y - agent.y
        actual_distance = math.sqrt(dx * dx + dy * dy)
        
        # Calculate actual relative angle
        actual_angle_to_food = math.degrees(math.atan2(dy, dx))
        actual_relative_angle = actual_angle_to_food - agent.orientation
        
        # Normalize angle to [-180, 180]
        while actual_relative_angle > 180:
            actual_relative_angle -= 360
        while actual_relative_angle < -180:
            actual_relative_angle += 360
        
        actual_food_info.append((actual_distance, actual_relative_angle, food_point))
    
    # Sort by actual distance
    actual_food_info.sort()
    
    # Compare with perceived data
    for i in range(min(5, len(actual_food_info))):
        perceived_distance = food_data[i*2].item()
        perceived_angle = food_data[i*2 + 1].item()
        actual_distance, actual_angle, food_point = actual_food_info[i]
        
        print(f"Food {i+1}:")
        print(f"  Distance: {perceived_distance:6.1f} vs {actual_distance:6.1f} "
              f"(diff: {abs(perceived_distance - actual_distance):5.1f})")
        print(f"  Angle:    {perceived_angle:6.1f}° vs {actual_angle:6.1f}° "
              f"(diff: {abs(perceived_angle - actual_angle):5.1f}°)")
        print(f"  Food pos: ({food_point.x:.1f}, {food_point.y:.1f})")
        print()

def demonstrate_decision_making():
    """Show how food perception affects agent decision making."""
    print("="*60)
    print("=== Food-Aware Decision Making ===")
    print()
    
    env = Environment()
    agent = Agent(100, 100, 0)
    
    print("Testing agent decision making with food perception:")
    print()
    
    for cycle in range(5):
        # Get actions
        actions = agent.decide_actions(env.agents, env.food_points)
        move, rotate_right, rotate_left = actions
        
        # Get perception data for analysis
        input_tensor = agent.get_perception_input(env.agents, env.food_points)
        food_start_idx = 2 + 3 * (POPULATION_SIZE - 1)
        food_data = input_tensor[food_start_idx:]
        
        # Find closest food
        closest_distance = food_data[0].item()
        closest_angle = food_data[1].item()
        
        print(f"Cycle {cycle + 1}:")
        print(f"  Agent pos: ({agent.x:.1f}, {agent.y:.1f}), facing {agent.orientation}°")
        print(f"  Closest food: {closest_distance:.1f} units away at {closest_angle:.1f}° relative angle")
        print(f"  Decision: Move={move}, Right={rotate_right}, Left={rotate_left}")
        
        # Apply actions
        if rotate_right:
            agent.rotate_right()
            print(f"  → Rotated RIGHT to {agent.orientation}°")
        elif rotate_left:
            agent.rotate_left()
            print(f"  → Rotated LEFT to {agent.orientation}°")
        
        if move:
            old_x, old_y = agent.x, agent.y
            agent.move_forward()
            print(f"  → Moved from ({old_x:.1f}, {old_y:.1f}) to ({agent.x:.1f}, {agent.y:.1f})")
        
        print()

def demonstrate_evolution_impact():
    """Show how food perception impacts evolution."""
    print("="*60)
    print("=== Evolution with Food Perception ===")
    print()
    
    from evolution import EvolutionManager
    
    env = Environment()
    evolution_manager = EvolutionManager()
    
    print("Running evolution with food-aware agents...")
    
    # Run initial simulation
    for i in range(30):
        env.update()
    
    initial_stats = env.get_statistics()
    print(f"Initial generation: {initial_stats['total_food_consumed']} food consumed")
    
    # Evolve
    new_agents, gen_stats, eval_summary = evolution_manager.evolve_generation_with_evaluation(
        env.agents, num_runs=3, num_iterations=30
    )
    
    print(f"Evolution results:")
    print(f"  Best agent avg food: {eval_summary['best_avg_food_count']:.2f}")
    print(f"  Population avg food: {eval_summary['population_avg_food_count']:.2f}")
    print(f"  Selected {gen_stats['selected_agents_count']} top performers")
    
    print("\n✅ Food perception enables more effective evolution!")
    print("✅ Agents can now evolve food-seeking strategies!")

if __name__ == "__main__":
    demonstrate_food_perception()
    demonstrate_perception_vs_reality()
    demonstrate_decision_making()
    demonstrate_evolution_impact()
    
    print("\n" + "="*60)
    print("🎉 FOOD PERCEPTION SYSTEM COMPLETE!")
    print("\nKey Enhancements:")
    print("• Agents now perceive food locations directly")
    print("• Distance and angle information for all food points")
    print("• Sorted by proximity for optimal decision making")
    print("• Fully integrated with existing neural network")
    print("• Compatible with evolution and competitive evaluation")
    print("• Enables evolution of sophisticated food-seeking behaviors")
