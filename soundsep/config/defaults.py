DEFAULTS = {
    "audio_directory": None,
    "recursive_search": False,
    "filename_pattern": "{filename}.wav",
    "channel_keys": None,
    "block_keys": None,

    # App/gui config
    "source_view.minimum_height": 150,
    "workspace.default_size": 2000,
    "workspace.constant_refresh": True,

    # Ampenv defaults
    "filter.low": 250,
    "filter.high": 10000,
    "filter.ampenv_rectify": 200.0,

    # Stft defaults
    "stft.window": 302,
    "stft.step": 22,
    "stft.cache.fraction_cached": 0.75,
    "stft.cache.n_scales": 8,
    "stft.cache.size": 2000,

    # Detection plugin
    "detection.ampenv_rectify": 200.0,
    "detection.ignore_width": 0.002,
    "detection.min_size": 0.002,
    "detection.fuse_duration": 0.005,

    # Advanced detection defaults
    "detection.advanced.software_gain": 1.0,
    "detection.advanced.signal_low": 2000,
    "detection.advanced.signal_high": 10000,
    "detection.advanced.noise_low": 500,
    "detection.advanced.noise_high": 1500,
    "detection.advanced.signal_gain": 1.0,
    "detection.advanced.noise_gain": 1.0,
    "detection.advanced.smooth_ms": 2.0,
    "detection.advanced.threshold": 0.1,
    "detection.advanced.min_gap_sec": 0.01,
    "detection.advanced.min_dur_sec": 0.01,
    "detection.advanced.max_dur_sec": 10.0,
}
