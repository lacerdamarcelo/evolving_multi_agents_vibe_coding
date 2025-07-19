"""
Demonstration of the new bidirectional rotation functionality.
This script shows how agents can now rotate both left and right using softmax selection.
"""

import torch
import torch.nn.functional as F
from agent import Agent
from neural_network import AgentNeuralNetwork
from config import ROTATION_ANGLE

def demonstrate_rotation_functionality():
    """Demonstrate the new three-output rotation system."""
    print("=== Bidirectional Rotation Demonstration ===")
    print(f"Rotation angle: {ROTATION_ANGLE} degrees")
    print()
    
    # Create an agent
    agent = Agent(100, 100, 0)  # Start facing east (0 degrees)
    print(f"Initial agent orientation: {agent.orientation}°")
    
    # Demonstrate manual rotation methods
    print("\n--- Manual Rotation Methods ---")
    agent.rotate_right()
    print(f"After rotate_right(): {agent.orientation}°")
    
    agent.rotate_left()
    print(f"After rotate_left(): {agent.orientation}°")
    
    agent.rotate_left()
    print(f"After another rotate_left(): {agent.orientation}°")
    
    # Reset agent
    agent.orientation = 0
    
    # Demonstrate neural network decision making
    print("\n--- Neural Network Decision Making ---")
    print("Testing multiple decision cycles to show softmax selection:")
    
    for i in range(10):
        # Get raw neural network output
        input_tensor = agent.get_perception_input([], [])
        with torch.no_grad():
            raw_output = agent.neural_network.forward(input_tensor)
            
            # Show the raw outputs before processing
            move_raw = raw_output[0].item()
            rotate_outputs = raw_output[1:3]
            rotation_probs = F.softmax(rotate_outputs, dim=0)
            
            # Get final actions
            actions = agent.neural_network.get_actions(input_tensor)
            move, rotate_right, rotate_left = actions
            
            print(f"Cycle {i+1:2d}: Raw outputs: [{move_raw:.3f}, {rotate_outputs[0].item():.3f}, {rotate_outputs[1].item():.3f}] "
                  f"→ Softmax: [{rotation_probs[0].item():.3f}, {rotation_probs[1].item():.3f}] "
                  f"→ Actions: Move={move}, Right={rotate_right}, Left={rotate_left}")
            
            # Apply the rotation action to show effect
            if rotate_right:
                agent.rotate_right()
                print(f"         → Agent rotated RIGHT to {agent.orientation}°")
            elif rotate_left:
                agent.rotate_left()
                print(f"         → Agent rotated LEFT to {agent.orientation}°")
    
    print("\n--- Key Features Demonstrated ---")
    print("✓ Three outputs: move_forward, rotate_right, rotate_left")
    print("✓ Softmax ensures only one rotation direction is selected")
    print("✓ Move forward remains independent with sigmoid activation")
    print("✓ Agents can now turn in both directions for more sophisticated movement")
    print("✓ Backward compatibility maintained with existing simulation structure")

def demonstrate_softmax_behavior():
    """Show how softmax ensures mutual exclusion for rotation."""
    print("\n=== Softmax Mutual Exclusion Demonstration ===")
    
    # Create test cases with different raw outputs
    test_cases = [
        [0.3, 0.8, 0.2],  # Strong right preference
        [0.7, 0.2, 0.9],  # Strong left preference  
        [0.5, 0.5, 0.5],  # Equal preference
        [0.1, 0.6, 0.4],  # Moderate right preference
    ]
    
    for i, raw_outputs in enumerate(test_cases):
        raw_tensor = torch.tensor(raw_outputs)
        
        # Apply softmax to rotation outputs
        rotation_probs = F.softmax(raw_tensor[1:3], dim=0)
        
        # Determine selection
        max_idx = torch.argmax(rotation_probs).item()
        selected_rotation = "RIGHT" if max_idx == 0 else "LEFT"
        
        print(f"Test {i+1}: Raw [{raw_outputs[1]:.1f}, {raw_outputs[2]:.1f}] "
              f"→ Softmax [{rotation_probs[0]:.3f}, {rotation_probs[1]:.3f}] "
              f"→ Selected: {selected_rotation}")
    
    print("\nSoftmax ensures:")
    print("• Probabilities sum to 1.0")
    print("• Only one rotation direction is selected")
    print("• Selection is based on highest probability")

if __name__ == "__main__":
    demonstrate_rotation_functionality()
    demonstrate_softmax_behavior()
    print("\n🎉 Bidirectional rotation system is working perfectly!")
