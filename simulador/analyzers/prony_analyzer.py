import numpy as np
from numpy.linalg import lstsq

class PronyAnalyzer:
    """
    Analyzer for Prony method.
    """
    def __init__(self, order, sr, cutoff_freq):
        self.order = order
        self.sr = sr
        self.cutoff = cutoff_freq

    def analyze(self, signal):
        """
        Estimate frequencies and amplitudes using Prony's method.
        Returns freqs (length order) and magnitudes (length order).
        """
        N = len(signal)
        K = self.order
        # Build Hankel matrix H (N-K x K)
        H = np.column_stack([signal[i:N-K+i] for i in range(K)])
        # Target vector y
        y = -signal[K:N]
        # Solve for linear predictors a
        a, *_ = lstsq(H, y, rcond=None)
        # Characteristic polynomial coefficients [1, a1,...,aK]
        coeffs = np.concatenate(([1], a))
        # Poles
        poles = np.roots(coeffs)
        # Sampling period
        Ts = 1 / self.sr
        # Frequencies (Hz)
        freqs = np.angle(poles) / (2 * np.pi * Ts)
        # Amplitudes
        V = np.vander(poles, N=K, increasing=True).T
        A, *_ = lstsq(V, signal[:K], rcond=None)
        return freqs, np.abs(A)
