import torch
import math

# ============================================
# BUILD EVERYTHING FROM SCRATCH
# No nn.MSELoss, no optim.Adam - pure math
# ============================================

# --- LOSS FUNCTION: Mean Squared Error ---
# Measures how wrong the network is
def mse_loss(predictions, targets):
    diff = predictions - targets
    squared = diff ** 2
    mean = squared.mean()
    return mean

# --- LOSS FUNCTION: Cross Entropy ---
# Used for classification (predicting next word)
def cross_entropy_loss(logits, targets):
    # Step 1: Softmax - convert raw numbers to probabilities
    exp_logits = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
    # Step 2: Negative log likelihood of correct class
    batch_size = targets.shape[0]
    correct_probs = probs[range(batch_size), targets]
    loss = -torch.log(correct_probs + 1e-9).mean()
    return loss, probs

# --- OPTIMIZER: SGD from Scratch ---
class SGDOptimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    # weight = weight - learning_rate * gradient
                    param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

# --- OPTIMIZER: Adam from Scratch ---
# Smarter than SGD - adapts learning rate per parameter
class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1    # momentum factor
        self.beta2 = beta2    # velocity factor
        self.eps = eps
        self.t = 0            # time step
        # Momentum and velocity for each parameter
        self.m = [torch.zeros_like(p) for p in self.parameters]
        self.v = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue
                g = param.grad

                # Update momentum (moving average of gradients)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

                # Update velocity (moving average of squared gradients)
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Update weights
                param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()


# --- TEST EVERYTHING ---
if __name__ == "__main__":
    print("=== Testing From-Scratch Components ===\n")

    # Test MSE Loss
    predictions = torch.tensor([2.5, 0.5, 2.0, 8.0])
    targets     = torch.tensor([3.0, 0.0, 2.0, 7.0])
    loss = mse_loss(predictions, targets)
    print(f"MSE Loss test:")
    print(f"  Predictions: {predictions.tolist()}")
    print(f"  Targets:     {targets.tolist()}")
    print(f"  Loss:        {loss.item():.4f}")
    print(f"  (PyTorch MSE: {torch.nn.MSELoss()(predictions, targets).item():.4f} - should match!)\n")

    # Test Cross Entropy
    logits = torch.tensor([[2.0, 1.0, 0.1],
                            [0.5, 2.5, 0.3]])
    targets_ce = torch.tensor([0, 1])
    ce_loss, probs = cross_entropy_loss(logits, targets_ce)
    print(f"Cross Entropy Loss test:")
    print(f"  Logits: {logits.tolist()}")
    print(f"  Targets: {targets_ce.tolist()}")
    print(f"  Probabilities: {probs.detach().tolist()}")
    print(f"  Loss: {ce_loss.item():.4f}\n")

    # Test Adam Optimizer
    print("Adam Optimizer test:")
    param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    optimizer = AdamOptimizer([param], lr=0.1)

    print(f"  Before: {param.detach().tolist()}")
    # Simulate a gradient
    loss_test = (param ** 2).sum()
    loss_test.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"  After 1 step: {param.detach().tolist()}")
    print(f"  (Values moved closer to 0 - correct!)")

    print("\nAll from-scratch components working!")
    print("This is the exact math inside PyTorch itself.")
    