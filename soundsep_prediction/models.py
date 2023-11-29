from torch import nn, load
from torchaudio.transforms import MelSpectrogram, Spectrogram

from soundsep_prediction.stft import StftParameters


class PredictionNetwork(nn.Module):
    """A 2-layer convolutional network for classifying a potential syllable point

    The dimensions of the output layer depends on the input data size,
    so not sure of a great way to decouple this from the specific
    dataloader used.
    """
    def __init__(
            self,
            channels: int,
            output_channels: int,
            stft_params: StftParameters = StftParameters(),
            ):
        super().__init__()
        self.spec = Spectrogram(
            n_fft=stft_params.nfft,
            hop_length=stft_params.hop,
        )
        self.layers = nn.Sequential(
            self.spec,
            nn.Conv2d(channels, 64, (4, 4)),
            nn.MaxPool2d(4),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4)),
            nn.MaxPool2d(4),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.out_layer = nn.Linear(1536, output_channels)

    def forward(self, x):
        return self.out_layer(self.layers(x)).float()

    @staticmethod
    def from_file(path, channels: int, output_channels: int) -> 'PredictionNetwork':
        """Load a pretrained network"""
        model = PredictionNetwork(channels, output_channels=output_channels)
        model.load_state_dict(load(path))
        return model


class MelPredictionNetwork(nn.Module):
    """A 2-layer convolutional network for classifying a potential syllable point

    The dimensions of the output layer depends on the input data size,
    so not sure of a great way to decouple this from the specific
    dataloader used.
    """
    def __init__(
            self,
            channels: int,
            output_channels: int,
            stft_params: StftParameters = StftParameters(),
            ):
        super().__init__()
        self.spec = MelSpectrogram(
            n_mels=32,
            n_fft=stft_params.nfft,
            hop_length=stft_params.hop,
            sample_rate=stft_params.sample_rate
        )
        self.layers = nn.Sequential(
            self.spec,
            nn.Conv2d(channels, 64, (4, 4)),
            nn.MaxPool2d(4),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (2, 2)),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.out_layer = nn.Linear(1536, output_channels)

    def forward(self, x):
        return self.out_layer(self.layers(x)).float()

    @staticmethod
    def from_file(path, channels: int, output_channels: int) -> 'MelPredictionNetwork':
        """Load a pretrained network"""
        model = MelPredictionNetwork(channels, output_channels=output_channels)
        model.load_state_dict(load(path))
        return model
