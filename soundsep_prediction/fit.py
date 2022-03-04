from typing import Callable, Optional

import torch
import tqdm.auto
from torch import nn


def partial_fit(epochs: int, model: nn.Module, loss_fn, opt, dl, device="cpu", on_epoch_complete: Optional[Callable] = None):
    """
    on_epoch_complete : function
        is a function that takes the epoch number, model and epoch loss
    """
    total_loss = 0
    total_data = 0

    for epoch in range(epochs):
        model.train()

        for xb, yb in tqdm.auto.tqdm(dl):
            loss = loss_fn(model(xb.to(device)), yb.to(device).squeeze())
            total_loss += loss.item() * len(xb)
            total_data += len(xb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        if on_epoch_complete is not None:
            on_epoch_complete(epoch, model, total_loss / total_data)

    return total_loss / total_data


def partial_predict(model: nn.Module, dl, device="cpu"):
    model.eval()

    pred_y = torch.concat([
        torch.sigmoid(model(xb.to(device))).cpu().detach() for xb, _ in tqdm.auto.tqdm(dl)
    ])

    true_y = torch.concat([
        yb for _, yb in dl
    ])

    return pred_y, true_y.squeeze()


def partial_test(model: nn.Module, loss_fn, dl, device="cpu"):
    model.eval()
    total_loss = 0
    total_data = 0
    for xb, yb in tqdm.auto.tqdm(dl):
        loss = loss_fn(model(xb.to(device)), yb.to(device).squeeze())
        total_loss += loss.item() * len(xb)
        total_data += len(xb)

    return total_loss / total_data



