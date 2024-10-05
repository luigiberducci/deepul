import torch
import numpy as np
import matplotlib.pyplot as plt


class DiscreteDistributionModel(torch.nn.Module):

    def __init__(self, d: int, device: torch.device) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(d, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distribution = self.compute_distribution()
        return distribution[x]

    def compute_distribution(self) -> torch.Tensor:
        exp_theta = torch.exp(self.theta)
        return exp_theta / torch.sum(exp_theta)

d = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
train_data = (torch.normal(0.5, 0.25, (10000,), device=device).clip(0, 1) * (d-1)).int()
test_data = (torch.normal(0.5, 0.25, (10000,), device=device).clip(0, 1) * (d-1)).int()


train_losses, test_losses = [], []

opt_hparams = {"lr": 1e-3, "weight_decay": 1e-4}
train_hparams = {"num_epochs": 100, "batch_size": 128}
batch_size = train_hparams["batch_size"]

model = DiscreteDistributionModel(d, device=device)
optimizer = torch.optim.Adam(model.parameters(), **opt_hparams)


def nll(px: torch.Tensor) -> torch.Tensor:
    return -torch.log(px)


for epoch in range(train_hparams["num_epochs"]):
    indices = torch.randperm(len(train_data))
    for start_id in range(0, len(train_data), batch_size):
        batch_indices = indices[start_id:start_id + batch_size]
        train_batch = train_data[batch_indices]

        optimizer.zero_grad()
        px = model(train_batch)
        train_loss = torch.mean(nll(px))
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.detach().cpu().numpy())

    with torch.no_grad():
        px = model(test_data)
        test_loss = torch.mean(nll(px))
        test_losses.append(test_loss.cpu().numpy())
        print(test_losses[-1])

        y = model.compute_distribution().cpu().numpy()
        plt.clf()
        plt.hist(train_data.cpu().numpy(), bins=np.arange(51), density=True, alpha=0.5, label="train data")
        plt.step(np.arange(d), y, where="post")
        plt.xlabel("x")
        plt.ylabel("p(x)")
        plt.title(f"Epoch {epoch}, test loss: {test_losses[-1]:.4f}")
        plt.legend()
        plt.pause(0.1)

train_losses = np.stack(train_losses)
test_losses = np.stack(test_losses)
distribution = model.compute_distribution()

print(train_losses.shape)
print(test_losses.shape)
print(distribution.shape)

