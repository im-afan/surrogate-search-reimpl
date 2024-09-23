import torch

# Example model, losses, and optimizer setup
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Random input and targets
inputs = torch.randn(10)
target1 = torch.randn(1)
target2 = torch.randn(1)

# Forward pass for loss1 first
output1 = model(inputs)
loss1 = torch.nn.functional.mse_loss(output1, target1)

# Now calculate loss2 after loss1
output2 = model(inputs)
loss2 = torch.nn.functional.mse_loss(output2, target2)

# Backward pass on loss1 first (retain the graph for further backpropagation)
loss1.backward(retain_graph=True)
optimizer.step()

# Now backward through loss2
optimizer.zero_grad()
loss2.backward()

# Update the model with the accumulated gradients
optimizer.step()

# Clear gradients for the next step
optimizer.zero_grad()