import torch
import torch.nn as nn
import torch.optim as optim

# Dummy dataset generation
def generate_data(num_samples):
    # Generate random features and binary labels
    X = torch.rand(num_samples, 2)  # 2D feature space
    y = (X[:, 0] + X[:, 1] > 1).float()  # Simple decision boundary
    return X, y

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(2, 1)  # Single layer for binary classification

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Sigmoid activation for binary output

# Training parameters
num_epochs = 100
batch_size = 16
learning_rate = 0.01
num_samples = 1000

# Generate dummy data
X, y = generate_data(num_samples)
dataset = torch.utils.data.TensorDataset(X, y)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn_a = nn.BCELoss()  # Binary Cross Entropy Loss
loss_fn_b = nn.MSELoss()  # Mean Squared Error Loss

# Initialize loss_a_prev
loss_a_prev = torch.tensor(0.0, requires_grad=True)

# Training loop
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X).squeeze()  # Squeeze to match shape
        # Compute loss_a and loss_b
        loss_a = loss_fn_a(outputs, batch_y)  # Binary Cross Entropy
        loss_b = loss_fn_b(outputs, batch_y)  # Mean Squared Error
        
        # Compute loss_total using the previous loss_a
        loss_total = loss_a_prev + loss_b
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        # Update loss_a_prev
        loss_a_prev = loss_a  # Keep the graph for gradient propagation

    # Print the loss for every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss Total: {loss_total.item():.4f}')

# Final evaluation
with torch.no_grad():
    test_outputs = model(X).squeeze()
    test_loss = loss_fn_a(test_outputs, y)
    print(f'Final Test Loss: {test_loss.item():.4f}')
