from dataclasses import dataclass

@dataclass
class StftParameters:
    hop: int
    half_window: int

    @property
    def n_fft(self):
        return 2 * self.half_window + 1
