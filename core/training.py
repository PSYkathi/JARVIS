import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from network import JarvisNetwork

# --- GENERATE SIMPLE TRAINING DATA ---
# Task: learn the pattern input -> input squared (normalized)
# Simple but real enough to demonstrate learning

def generate_data(samples=1000):
    X = torch.randn(samples, 10)
    # Target: sum of squares of first 5 inputs
    y = (X[:, :5] ** 2).sum(dim=1, keepdim=True)
    # Normalize target
    y = (y - y.mean()) / y.std()
    return X, y

# --- TRAINING LOOP ---
def train():
    print("=== Training JARVIS Network ===\n")

    # Build network
    network = JarvisNetwork(input_size=10, hidden_size=64, output_size=1)

    # Loss function - measures HOW WRONG the network is
    loss_fn = nn.MSELoss()

    # Optimizer - fixes the weights based on the mistake
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    # Generate data
    X_train, y_train = generate_data(1000)
    X_val, y_val = generate_data(200)

    loss_history = []

    # Training loop
    epochs = 200
    for epoch in range(epochs):

        # 1. FORWARD PASS - make a prediction
        predictions = network(X_train)

        # 2. CALCULATE LOSS - how wrong are we?
        loss = loss_fn(predictions, y_train)

        # 3. ZERO GRADIENTS - clear previous gradients
        optimizer.zero_grad()

        # 4. BACKWARD PASS - calculate how much each weight caused the error
        loss.backward()

        # 5. UPDATE WEIGHTS - fix the weights
        optimizer.step()

        loss_history.append(loss.item())

        # Print progress
        if (epoch + 1) % 20 == 0:
            val_pred = network(X_val)
            val_loss = loss_fn(val_pred, y_val)
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss.item():.4f}")

    print("\nTraining complete!")
    print(f"Starting loss: {loss_history[0]:.4f}")
    print(f"Final loss:    {loss_history[-1]:.4f}")
    print(f"Improvement:   {((loss_history[0]-loss_history[-1])/loss_history[0]*100):.1f}%")

    # Plot the learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='blue', linewidth=2)
    plt.title('JARVIS Learning Curve - Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (how wrong the network is)')
    plt.grid(True)
    plt.savefig('core/learning_curve.png')
    print("\nLearning curve saved to core/learning_curve.png")
    print("Open it and see JARVIS learning in real time!")

if __name__ == "__main__":
    train()