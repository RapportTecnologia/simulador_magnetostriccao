import numpy as np
from numpy.linalg import eigh
from scipy.signal import hilbert

class EspritAnalyzer:
    """
    Analyzer for ESPRIT method.
    """
    def __init__(self, order, sr, cutoff_freq):
        self.order = order
        self.sr = sr
        self.cutoff = cutoff_freq

    def analyze(self, signal):
        # Convert real signal to analytic signal for ESPRIT
        if not np.iscomplexobj(signal):
            signal = hilbert(signal)

        """
        Estimate frequencies and magnitudes using the ESPRIT algorithm.
        Returns freqs (length order) and magnitudes (length order).
        """
        N = len(signal)
        K = self.order
        M = 2 * K
        if N < M:
            raise ValueError(f"Signal length {N} too short for ESPRIT model dimension {M}")
        # Build data matrix: M rows, L columns for ESPRIT
        L = N - M + 1
        X = np.vstack([signal[i:i+L] for i in range(M)])
        # Covariance matrix
        R = X @ X.conj().T / L
        # Eigendecomposition
        e_vals, e_vecs = eigh(R)
        idx = np.argsort(e_vals)[::-1]
        # Signal subspace
        Es = e_vecs[:, idx[:K]]
        # Subspace shifts
        Es1 = Es[:-1, :]
        Es2 = Es[1:, :]
        # Solve for rotational operator
        psi = np.linalg.pinv(Es1) @ Es2
        # Eigenvalues of psi
        lambdas, _ = np.linalg.eig(psi)
        Ts = 1 / self.sr
        # Estimate frequencies
        freqs = np.abs(np.angle(lambdas) / (2 * np.pi * Ts))
        # Amplitude estimation via least squares
        S = np.vander(lambdas, N, increasing=True).T
        a, *_ = np.linalg.lstsq(S, signal, rcond=None)
        mags = np.abs(a)
        # Apply cutoff filter
        mask = freqs <= self.cutoff
        if np.sum(mask) < K:
            raise ValueError(f"Not enough frequencies <= cutoff {self.cutoff}")
        freqs = freqs[mask]
        mags = mags[mask]
        # Peak picking: top K by magnitude
        peaks = np.argsort(mags)[-K:]
        est_freqs = freqs[peaks]
        est_mags = mags[peaks]
        # Sort by estimated frequency
        order_idx = np.argsort(est_freqs)
        return est_freqs[order_idx], est_mags[order_idx]
