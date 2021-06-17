default_config = {
    "filename_pattern": "{filename}.wav",
    "channel_keys": None,
    "block_keys": ["filename"],

    "workspace.default_size": 2000,

    "stft.window": 302,
    "stft.step": 44,
    "stft.cache.size": 4 * 2000,

    # Detection plugin
    "detection.ampenv_rectify": 200.0,
    "detection.ignore_width": 0.002,
    "detection.min_size": 0.005,
    "detection.fuse_duration": 0.01,
}
