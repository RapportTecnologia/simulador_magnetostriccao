import numpy as np

class FFTAnalyzer:
    """
    Analyzer for FFT method.
    """
    def __init__(self, sr, cutoff_freq):
        self.sr = sr
        self.cutoff = cutoff_freq

    def analyze(self, signal):
        """
        Compute FFT and return frequencies and magnitudes up to cutoff.
        """
        # Compute FFT magnitude
        F = np.abs(np.fft.rfft(signal))
        # Compute frequency bins
        freqs = np.fft.rfftfreq(len(signal), 1 / self.sr)
        # Apply cutoff
        mask = freqs <= self.cutoff
        return freqs[mask], F[mask]
