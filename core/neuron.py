import math

# A single neuron - the most basic unit of all AI
# output = activation(weights · inputs + bias)

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        # Step 1: weighted sum
        raw = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        # Step 2: activation
        output = relu(raw)
        return raw, output

# --- TEST ---
if __name__ == "__main__":
    # A neuron with 3 inputs
    neuron = Neuron(
        weights=[0.5, -0.3, 0.8],
        bias=0.1
    )

    inputs = [1.0, 2.0, 3.0]

    raw, output = neuron.forward(inputs)

    print("=== Single Neuron ===")
    print(f"Inputs:     {inputs}")
    print(f"Weights:    {neuron.weights}")
    print(f"Bias:       {neuron.bias}")
    print(f"Raw sum:    {raw:.4f}")
    print(f"After ReLU: {output:.4f}")
    print()
    print("Manual check:")
    print(f"(0.5×1.0) + (-0.3×2.0) + (0.8×3.0) + 0.1")
    print(f"= {0.5*1.0} + {-0.3*2.0} + {0.8*3.0} + 0.1")
    print(f"= {0.5*1.0 + -0.3*2.0 + 0.8*3.0 + 0.1:.4f}")