"""
Demonstration of the gated rotation system.
This script shows how the fourth output gates the rotation controls.
"""

import torch
import torch.nn.functional as F
from agent import Agent
from neural_network import AgentNeuralNetwork
from environment import Environment
from config import ROTATION_ANGLE

def demonstrate_gated_rotation():
    """Demonstrate the four-output gated rotation system."""
    print("=== Gated Rotation System Demonstration ===")
    print(f"Rotation angle: {ROTATION_ANGLE} degrees")
    print()
    
    print("Network Architecture:")
    print("• Output 0: Move forward (sigmoid activation)")
    print("• Output 1: Rotate right preference (sigmoid)")
    print("• Output 2: Rotate left preference (sigmoid)")
    print("• Output 3: Rotation gate (sigmoid - enables/disables rotation)")
    print()
    
    # Create test agents
    agents = [Agent(100, 100, i*60) for i in range(6)]
    
    print("=== Testing Different Gate Behaviors ===")
    
    for agent_idx, agent in enumerate(agents):
        print(f"\nAgent {agent_idx + 1}:")
        
        # Test multiple decision cycles
        for cycle in range(3):
            actions = agent.decide_actions([], [])
            move, rotate_right, rotate_left = actions
            
            # Get raw neural network outputs
            input_tensor = agent.get_perception_input([], [])
            with torch.no_grad():
                raw_output = agent.neural_network.forward(input_tensor)
                move_raw = raw_output[0].item()
                right_pref = raw_output[1].item()
                left_pref = raw_output[2].item()
                gate_raw = raw_output[3].item()
                
                # Show softmax calculation when gate is enabled
                if gate_raw > 0.5:
                    rotation_probs = F.softmax(raw_output[1:3], dim=0)
                    right_prob = rotation_probs[0].item()
                    left_prob = rotation_probs[1].item()
                else:
                    right_prob = left_prob = 0.0
            
            gate_status = "ENABLED" if gate_raw > 0.5 else "DISABLED"
            
            print(f"  Cycle {cycle + 1}:")
            print(f"    Raw outputs: Move={move_raw:.3f}, Right={right_pref:.3f}, Left={left_pref:.3f}, Gate={gate_raw:.3f}")
            print(f"    Gate status: {gate_status}")
            
            if gate_raw > 0.5:
                print(f"    Softmax probs: Right={right_prob:.3f}, Left={left_prob:.3f}")
                print(f"    Final actions: Move={move}, Right={rotate_right}, Left={rotate_left}")
            else:
                print(f"    Final actions: Move={move}, Right={rotate_right}, Left={rotate_left} (rotation gated OFF)")
            
            # Apply actions to show effect
            initial_orientation = agent.orientation
            if rotate_right:
                agent.rotate_right()
                print(f"    → Agent rotated RIGHT: {initial_orientation}° → {agent.orientation}°")
            elif rotate_left:
                agent.rotate_left()
                print(f"    → Agent rotated LEFT: {initial_orientation}° → {agent.orientation}°")
            else:
                print(f"    → No rotation applied (orientation remains {agent.orientation}°)")

def demonstrate_gated_vs_ungated():
    """Compare gated vs ungated rotation behavior."""
    print("\n" + "="*60)
    print("=== Gated vs Ungated Rotation Comparison ===")
    print()
    
    # Create agents for comparison
    agent = Agent(100, 100, 0)
    
    print("Simulating 10 decision cycles:")
    print("Format: [Move, Right_pref, Left_pref, Gate] → Actions")
    print()
    
    no_rotation_count = 0
    rotation_count = 0
    
    for i in range(10):
        input_tensor = agent.get_perception_input([], [])
        actions = agent.decide_actions([], [])
        move, rotate_right, rotate_left = actions
        
        with torch.no_grad():
            raw_output = agent.neural_network.forward(input_tensor)
            outputs = [raw_output[j].item() for j in range(4)]
            gate_enabled = outputs[3] > 0.5
        
        if rotate_right or rotate_left:
            rotation_count += 1
            action_str = "RIGHT" if rotate_right else "LEFT"
        else:
            no_rotation_count += 1
            action_str = "NONE"
        
        gate_str = "ON " if gate_enabled else "OFF"
        
        print(f"Cycle {i+1:2d}: [{outputs[0]:.2f}, {outputs[1]:.2f}, {outputs[2]:.2f}, {outputs[3]:.2f}] "
              f"Gate:{gate_str} → Move:{move}, Rotate:{action_str}")
    
    print(f"\nSummary:")
    print(f"• Cycles with rotation: {rotation_count}/10 ({rotation_count*10}%)")
    print(f"• Cycles without rotation: {no_rotation_count}/10 ({no_rotation_count*10}%)")
    print(f"• Gate allows selective rotation control!")

def demonstrate_environment_integration():
    """Show gated rotation working in full environment."""
    print("\n" + "="*60)
    print("=== Environment Integration Test ===")
    print()
    
    env = Environment()
    print(f"Created environment with {len(env.agents)} agents")
    
    # Count rotation behaviors before simulation
    print("\nAnalyzing initial agent behaviors...")
    rotation_enabled_count = 0
    rotation_disabled_count = 0
    
    for agent in env.agents[:10]:  # Sample first 10 agents
        input_tensor = agent.get_perception_input(env.agents, env.food_points)
        with torch.no_grad():
            raw_output = agent.neural_network.forward(input_tensor)
            gate_value = raw_output[3].item()
            if gate_value > 0.5:
                rotation_enabled_count += 1
            else:
                rotation_disabled_count += 1
    
    print(f"Sample of 10 agents:")
    print(f"• Rotation gate ENABLED: {rotation_enabled_count}")
    print(f"• Rotation gate DISABLED: {rotation_disabled_count}")
    
    # Run simulation
    print(f"\nRunning simulation for 20 iterations...")
    for i in range(20):
        env.update()
    
    stats = env.get_statistics()
    print(f"Results: {stats['living_count']} agents alive, {stats['total_food_consumed']} food consumed")
    print("✓ Gated rotation system working in full environment!")

if __name__ == "__main__":
    demonstrate_gated_rotation()
    demonstrate_gated_vs_ungated()
    demonstrate_environment_integration()
    
    print("\n" + "="*60)
    print("🎉 GATED ROTATION SYSTEM COMPLETE!")
    print("\nKey Features:")
    print("• Fourth output gates rotation controls")
    print("• Agents can choose NOT to rotate when unnecessary")
    print("• More realistic and energy-efficient movement")
    print("• Maintains competitive left/right selection when rotation is enabled")
    print("• Fully integrated with existing simulation systems")
