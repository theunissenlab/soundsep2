import torch
from torch import nn


def partial_fit(epochs: int, model: nn.Module, loss_fn, opt, dl):
    total_loss = 0
    total_data = 0

    for epoch in range(epochs):
        model.train()

        for xb, yb in dl:
            loss = loss_fn(model(xb), yb.squeeze())
            total_loss += loss.item() * len(xb)
            total_data += len(xb)
            loss.backward()
            opt.step()
            opt.zero_grad()

    return total_loss / total_data


def partial_predict(model: nn.Module, dl):
    model.eval()

    pred_y = torch.concat([
        torch.sigmoid(model(xb)) for xb, _ in dl
    ])

    true_y = torch.concat([
        yb for _, yb in dl
    ])

    return pred_y, true_y.squeeze()


def partial_test(model: nn.Module, loss_fn, dl):
    model.eval()
    total_loss = 0
    total_data = 0
    for xb, yb in dl:
        loss = loss_fn(model(xb), yb.squeeze())
        total_loss += loss.item() * len(xb)
        total_data += len(xb)

    return total_loss / total_data



