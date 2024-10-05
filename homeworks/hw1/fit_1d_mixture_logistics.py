import numpy as np
import torch

from torch.nn import functional as F


class MixtureOfLogistics(torch.nn.Module):
  def __init__(self, d: int, n: int, device: torch.device) -> None:
    super().__init__()

    self.d = d
    self.n = n
    self.device = device

    self.logits = torch.nn.Parameter(torch.ones(n, device=device))
    self.means = torch.nn.Parameter(torch.zeros(n, device=device))
    self.log_std = torch.nn.Parameter(torch.zeros(n, device=device))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-6
    x = x.unsqueeze(1).expand(-1, self.n)
    x_plus_norm = (x + 0.5 - self.means) / torch.exp(self.log_std)
    x_minus_norm = (x - 0.5 - self.means) / torch.exp(self.log_std)

    cdf_plus = torch.sigmoid(x_plus_norm)
    cdf_minus = torch.sigmoid(x_minus_norm)
    cdf_delta = cdf_plus - cdf_minus

    cdf_x = torch.where(
      x < epsilon,
      cdf_plus,
      torch.where(
        x > self.d - 1 - epsilon,
        1 - cdf_minus,
        cdf_delta
      )
    )
    x_log_prob = torch.log(torch.clamp(cdf_x, min=epsilon))

    log_pi = F.log_softmax(self.logits, dim=0)
    return torch.logsumexp(log_pi + x_log_prob, dim=1)

  def loss(self, x: torch.Tensor) -> torch.Tensor:
    return -self(x)

  def compute_distribution(self) -> torch.Tensor:
    x = torch.arange(self.d, device=self.device).float()
    return torch.exp(self(x))


d = 50
device = "cpu"
train_data = (torch.normal(0.5, 0.25, (10000,), device=device).clip(0, 1) * (d-1)).int()
test_data = (torch.normal(0.5, 0.25, (10000,), device=device).clip(0, 1) * (d-1)).int()



train_losses, test_losses = [], []

opt_hparams = {"lr": 1e-3, "weight_decay": 1e-4}
train_hparams = {"num_epochs": 300, "batch_size": 64}
batch_size = train_hparams["batch_size"]

model = MixtureOfLogistics(d=100, n=10, device=device)
optimizer = torch.optim.Adam(model.parameters(), **opt_hparams)



for epoch in range(train_hparams["num_epochs"]):

    indices = torch.randperm(len(train_data))
    for start_id in range(0, len(train_data), batch_size):
      batch_indices = indices[start_id:start_id+batch_size]
      train_batch = train_data[batch_indices]

      optimizer.zero_grad()
      train_loss = (model.loss(train_batch)).mean()
      train_loss.backward()
      optimizer.step()

      train_losses.append(train_loss.detach().cpu().numpy())

    with torch.no_grad():
      test_loss = torch.mean(model.loss(test_data))
      test_losses.append(test_loss.cpu().numpy())

      if epoch % 10 == 0:
        print("epoch: ", epoch, ", test loss: ", test_loss)

train_losses = np.stack(train_losses)
test_losses = np.stack(test_losses)
distribution = model.compute_distribution().detach().cpu().numpy()

import matplotlib.pyplot as plt
plt.hist(train_data.cpu().numpy(), bins=d, density=True)
plt.plot(distribution)
plt.show()

