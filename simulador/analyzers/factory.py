import numpy as np
from .fft_analyzer import FFTAnalyzer
from .prony_analyzer import PronyAnalyzer

class AnalyzerFactory:
    """
    Factory to create analyzer instances based on method name.
    """
    @staticmethod
    def create(method, sr, cutoff_freq, order=None):
        method = method.lower()
        if method == 'fft':
            return FFTAnalyzer(sr, cutoff_freq)
        elif method == 'prony':
            k = order if order is not None else 10
            return PronyAnalyzer(k, sr, cutoff_freq)
        elif method == 'music':
            class MusicAnalyzer:
                def __init__(self, sr, cutoff):
                    pass
                def analyze(self, signal):
                    return np.array([]), np.array([])
            return MusicAnalyzer(sr, cutoff_freq)
        elif method == 'esprit':
            class EspritAnalyzer:
                def __init__(self, sr, cutoff):
                    pass
                def analyze(self, signal):
                    return np.array([]), np.array([])
            return EspritAnalyzer(sr, cutoff_freq)
        else:
            raise ValueError(f"Unknown analysis method: {method}")
