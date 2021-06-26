DEFAULTS = {
    "audio_directory": None,
    "recursive_search": False,
    "filename_pattern": "{filename}.wav",
    "channel_keys": None,
    "block_keys": None,

    # App/gui config
    "source_view.minimum_height": 300,
    "workspace.default_size": 2000,

    # Ampenv defaults
    "filter.low": 250,
    "filter.high": 10000,
    "filter.ampenv_rectify": 200.0,

    # Stft defaults
    "stft.window": 302,
    "stft.step": 44,
    "stft.cache.size": 4 * 2000,

    # Detection plugin
    "detection.ampenv_rectify": 200.0,
    "detection.ignore_width": 0.002,
    "detection.min_size": 0.002,
    "detection.fuse_duration": 0.005,
}
