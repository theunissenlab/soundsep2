from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd
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


def partial_predict(model: nn.Module, dl, return_labels=False, device="cpu"):
    model.eval()

    pred_y = torch.concat([
        torch.sigmoid(model(xb.to(device))).cpu().detach() for xb, _ in tqdm.auto.tqdm(dl)
    ])

    if return_labels:
        true_y = torch.concat([
            yb for _, yb in dl
        ])
        return pred_y, true_y.squeeze()
    else:
        return pred_y


def partial_test(model: nn.Module, loss_fn, dl, device="cpu"):
    model.eval()
    total_loss = 0
    total_data = 0
    for xb, yb in tqdm.auto.tqdm(dl):
        loss = loss_fn(model(xb.to(device)), yb.to(device).squeeze())
        total_loss += loss.item() * len(xb)
        total_data += len(xb)

    return total_loss / total_data


def to_segments_table(
        p: Iterable[float],
        threshold: float,
        source_names: List[str],
        source_channels: List[int],
        stft_hop: int,
        min_gap_size: int = 4,
        min_segment_size: int = 2,
        min_p_max: float = 0.0,
        ):
    """Convert an array of probabilities into a table of start and stop times
    """
    """Given an array of probabilities, return the start and end times of every segment where the probability of a syllable exceeds 30%
    """
    all_segments = []
    for ch in range(p.shape[1]):
        mask = p[:, ch] > threshold
        above_thresh = np.where(mask)[0]  # 6, 7, 8, 9, 12, 13, ...
        if not len(above_thresh):
            continue
        diffs = np.diff(above_thresh)  # 1, 1, 1, 3, ...

        # First, lets flip the points where the p dips below threshold for `min_gap_size`
        # or fewer timepoints
        to_flip = (diffs <= min_gap_size) & (diffs > 1)
        amounts_to_flip = diffs[to_flip]
        places_to_flip = above_thresh[:-1][to_flip]
        for amt, place in zip(amounts_to_flip, places_to_flip):
            mask[place:place + amt] = True

        # This "above_thresh" smoothes out places were p spuriously dipped below threshold
        above_thresh = np.where(mask)[0]

        if not len(above_thresh):
            continue

        triggered_selector = np.diff(above_thresh) > min_gap_size
        endpoints = above_thresh[:-1][triggered_selector]
        endpoints = endpoints + 1
        startpoints = above_thresh[1:][triggered_selector]

        if len(startpoints) == 0 and len(endpoints) == 0:
            continue

        if endpoints[0] < startpoints[0]:
            startpoints = np.concatenate([above_thresh[:1], startpoints])

        if startpoints[-1] > endpoints[-1]:
            endpoints = np.concatenate([endpoints, above_thresh[-1:] + 1])

        segments = np.array(list(zip(startpoints, endpoints)))
        segments = segments[(segments[:, 1] - segments[:, 0]) > min_segment_size]

        # Finally, we reject segments where the largest p in the segment does not exceed min_p_max
        peak_filter = np.ones(len(segments)).astype(bool)
        for i, (i0, i1) in enumerate(segments):
            peak_filter[i] = np.max(p[i0:i1, ch]) >= min_p_max
        segments = segments[peak_filter]

        segments = segments * stft_hop

        for i0, i1 in segments:
            all_segments.append({
                "SourceName": source_names[ch],
                "SourceChannel": source_channels[ch],
                "StartIndex": i0,
                "StopIndex": i1,
                "Tags": []
            })

    if not len(all_segments):
        return pd.DataFrame([], columns=[
            "SourceName", "SourceChannel", "StartIndex", "StopIndex", "Tags"
        ])


    return pd.DataFrame(all_segments).sort_values("StartIndex")
