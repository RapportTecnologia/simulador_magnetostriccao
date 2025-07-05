import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import eigh

class MusicAnalyzer:
    """
    Analyzer for MUSIC algorithm.
    """
    def __init__(self, order, sr, cutoff_freq):
        self.order = order
        self.sr = sr
        self.cutoff = cutoff_freq

    def analyze(self, signal):
        """
        Estimate frequencies and magnitudes using the MUSIC algorithm.
        Returns freqs (length order) and magnitudes (length order).
        """
        N = len(signal)
        K = self.order
        M = 2 * K
        if N < M:
            raise ValueError(f"Signal length {N} too short for MUSIC model dimension {M}")

        # Estimate autocorrelation
        r = np.correlate(signal, signal, mode='full') / N
        mid = len(r) // 2
        rxx = r[mid:mid + M]

        # Build autocorrelation matrix
        R = toeplitz(rxx)

        # Eigendecomposition
        e_vals, e_vecs = eigh(R)
        idx = np.argsort(e_vals)

        # Noise subspace eigenvectors (smallest M-K eigenvalues)
        E_n = e_vecs[:, idx[:M - K]]

        # Frequency grid (FFT bins up to cutoff)
        freqs = np.fft.rfftfreq(N, 1 / self.sr)
        mask = freqs <= self.cutoff
        freqs = freqs[mask]

        # Steering vectors (M x n_freqs)
        m = np.arange(M)[:, None]
        A = np.exp(-1j * 2 * np.pi * freqs[None, :] * m / self.sr)

        # Noise subspace projection
        E_n_h = np.conj(E_n.T)

        # Pseudospectrum
        ps = 1.0 / np.sum(np.abs(E_n_h @ A) ** 2, axis=0)

        # Peak picking: top K peaks
        if len(ps) < K:
            raise ValueError(f"Not enough frequency points to identify {K} peaks")
        peaks = np.argsort(ps)[-K:]
        est_freqs = freqs[peaks]
        est_mags = ps[peaks]

        # Sort by estimated frequency
        order_idx = np.argsort(est_freqs)
        return est_freqs[order_idx], est_mags[order_idx]
