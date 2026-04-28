import torch
import torch.nn as nn
import torch.optim as optim

# We use PyTorch as our math tool - just matrix operations
# No pre-built AI, just math accelerated by PyTorch

class NeuronLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Weights matrix + bias vector - initialized randomly
        self.weights = nn.Parameter(torch.randn(input_size, output_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        # Matrix multiplication = all neurons computed at once
        raw = torch.matmul(x, self.weights) + self.bias
        output = torch.relu(raw)
        return output

class JarvisNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = NeuronLayer(input_size, hidden_size)
        self.layer2 = NeuronLayer(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x

# --- TEST ---
if __name__ == "__main__":
    # Build the network
    network = JarvisNetwork(
        input_size=10,
        hidden_size=64,
        output_size=5
    )

    # Count total parameters
    total_params = sum(p.numel() for p in network.parameters())
    print("=== JARVIS Neural Network ===")
    print(f"Input size:   10 neurons")
    print(f"Hidden size:  64 neurons x 2 layers")
    print(f"Output size:  5 neurons")
    print(f"Total parameters: {total_params}")
    print()

    # Run a forward pass with random input
    test_input = torch.randn(1, 10)
    output = network(test_input)

    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.detach().numpy()}")
    print()
    print("Forward pass successful!")
    print("These outputs are random garbage - the network hasnt learned anything yet.")
    print("Next step: teach it by calculating its mistakes and fixing them.")