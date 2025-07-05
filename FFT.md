# Análise de FFT

## Sumário

O método FFT (Fast Fourier Transform) calcula a transformada discreta de Fourier de um sinal, permitindo obter amplitudes e fases dos componentes de frequência.

## Princípios Fundamentais

1. **Transformada Discreta de Fourier**
   Converte um sinal de tempo em uma representação de frequência usando FFT, que computa eficientemente a DFT:
   ```
   F[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}
   ```
2. **Bins de Frequência**
   As frequências correspondentes são dadas por `frequencies = np.fft.rfftfreq(N, 1/fs)`.
3. **Aplicação de Corte**
   Descarte componentes acima de uma frequência de corte para focar na banda de interesse.

## Vantagens e Desvantagens

- **Vantagens**:
  - Extremamente rápido com complexidade O(N log N).
  - Implementações otimizadas amplamente disponíveis.
- **Desvantagens**:
  - Resolução limitada por tamanho da janela.
  - Efeito de vazamento espectral sem janela de suavização.

## Fluxo de Implementação em Python

```python
import numpy as np

def fft(signal, fs, cutoff):
    """
    Compute FFT and return frequencies and magnitudes up to cutoff.
    :param signal: array de sinal (1D)
    :param fs: taxa de amostragem em Hz
    :param cutoff: frequência de corte em Hz
    :return: (freqs, mags)
    """
    # Computa espectro
    F = np.abs(np.fft.rfft(signal))
    # Gera bins de frequência
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    # Aplica corte
    mask = freqs <= cutoff
    return freqs[mask], F[mask]
```

## Como Funciona no Simulador

1. Instancia `FFTAnalyzer(sr, cutoff)`.
2. Chama `analyze(signal)` para obter frequências e magnitudes até a frequência de corte.
3. Exibe espectro FFT no gráfico do simulador.

---

Este documento serve de base para implementação didática do método de FFT no simulador. A seguir, crie a classe `FFTAnalyzer` que encapsula essa lógica.
