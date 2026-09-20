import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X = torch.randn(1000, 3)

true_w = torch.tensor([0.8, 0.3, -0.5])
true_b = 0.02

y = X @ true_w + true_b + 0.1 * torch.randn(1000)
y = y.unsqueeze(1)

X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()

    prediction = model(X_train)
    loss = loss_fn(prediction, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

with torch.no_grad():
    test_preditcion = model(X_test)
    test_loss = loss_fn(test_preditcion, y_test)

print(f"\nTest MSE: {test_loss.item():.4f}")

for name, parameter in model.named_parameters():
    print(f"{name}:")
    print(parameter.detach())

initial_capital = 10_000

with torch.no_grad():
    predicted_return = model(X_test[0:1]).item()

hypothetical_value = initial_capital * (1 + predicted_return)

print(f"\nPredicted return: {predicted_return:.2%}")
print(f"Hypothetical outcome: ${hypothetical_value:,.2f}")