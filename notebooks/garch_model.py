import torch
import torch.nn as nn
import torch.optim as optim

class GARCH11(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.8))

    def forward(self, returns):
        T = len(returns)
        sigma2 = torch.zeros_like(returns)
        sigma2[0] = returns.var()

        for t in range(1, T):
            sigma2[t] = (
                self.omega +
                self.alpha * returns[t-1]**2 +
                self.beta * sigma2[t-1]
            )
        return sigma2

def negative_log_likelihood(model, returns):
    sigma2 = model(returns)
    loglik = 0.5 * (torch.log(sigma2) + (returns**2) / sigma2)
    return loglik.sum()

# Example: Replace this with your own return series
# returns = torch.tensor([...], dtype=torch.float32)

# model = GARCH11()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(1000):
#     optimizer.zero_grad()
#     loss = negative_log_likelihood(model, returns)
#     loss.backward()
#     optimizer.step()
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}: NLL = {loss.item():.4f}")
