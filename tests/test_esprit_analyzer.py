import unittest
import numpy as np
import sys
import os
# Add project root to PYTHONPATH to import simulador package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulador.analyzers.esprit_analyzer import EspritAnalyzer

class TestEspritAnalyzer(unittest.TestCase):
    def test_single_sinusoid(self):
        sr = 1000
        freq = 50
        t = np.arange(0, 1, 1/sr)
        signal = np.sin(2 * np.pi * freq * t)
        analyzer = EspritAnalyzer(order=1, sr=sr, cutoff_freq=100)
        est_freqs, est_mags = analyzer.analyze(signal)
        # Should detect one frequency close to the true value
        self.assertEqual(len(est_freqs), 1)
        self.assertAlmostEqual(est_freqs[0], freq, delta=1.0)
        self.assertGreater(est_mags[0], 0)

    def test_two_sinusoids(self):
        sr = 1000
        freqs_true = [50, 120]
        t = np.arange(0, 1, 1/sr)
        signal = np.sin(2 * np.pi * freqs_true[0] * t) + 0.5 * np.sin(2 * np.pi * freqs_true[1] * t)
        analyzer = EspritAnalyzer(order=2, sr=sr, cutoff_freq=200)
        est_freqs, est_mags = analyzer.analyze(signal)
        # Should detect two frequencies close to the true values
        self.assertEqual(len(est_freqs), 2)
        for f_true in freqs_true:
            # Check at least one estimated freq matches each true freq
            self.assertTrue(any(abs(est_freqs - f_true) < 2.0))

if __name__ == '__main__':
    unittest.main()
