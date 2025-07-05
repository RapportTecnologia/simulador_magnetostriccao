# Análise de MUSIC

## Sumário

O método MUSIC (Multiple Signal Classification) estima frequências de componentes em sinais compostos separando subespaço de sinal e ruído. Baseia-se na projeção de vetores de direção em um pseudospectro.

## Princípios Fundamentais

1. **Autocorrelação**  
   Estima autocorrelação do sinal:
   ```python
   r = np.correlate(signal, signal, mode='full') / N
   ```
2. **Matriz de Autocorrelação**  
   Constrói matriz Toeplitz R (M×M) a partir de rxx.
3. **Subespaço de Ruído**  
   Decompõe R em autovalores e autovetores, seleciona os M–K menores autovalores para formar Eₙ.
4. **Pseudospectro MUSIC**  
   Calcula pseudospectro:
   ```python
   P(f) = 1 / (a(f)ᴴ · Eₙ · Eₙᴴ · a(f))
   ```
5. **Detecção de Picos**  
   Identifica os K maiores picos em P(f) como frequências estimadas.

## Vantagens e Desvantagens

- **Vantagens**:  
  - Alta resolução para modos próximos em frequência.  
  - Sem busca exaustiva.
- **Desvantagens**:  
  - Sensível ao ruído e erro em Rxx.  
  - Depende da escolha de M (geralmente 2K).

## Fluxo de Implementação em Python

```python
import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import eigh

def music(signal, K, fs, cutoff):
    """
    Estima frequências e magnitudes usando o algoritmo MUSIC.
    :param signal: array 1D do sinal
    :param K: número de componentes
    :param fs: taxa de amostragem
    :param cutoff: frequência de corte (Hz)
    :return: (freqs, mags)
    """
    N = len(signal)
    M = 2 * K
    if N < M:
        raise ValueError(f"Comprimento {N} menor que dimensão {M} para MUSIC")

    # Autocorrelação
    r = np.correlate(signal, signal, mode='full') / N
    mid = len(r) // 2
    rxx = r[mid:mid + M]

    # Matriz de autocorrelação
    R = toeplitz(rxx)

    # Decomposição em autovalores
    e_vals, e_vecs = eigh(R)
    idx = np.argsort(e_vals)
    E_n = e_vecs[:, idx[:M - K]]

    # Grade de frequências e vetores de direção
    freqs = np.fft.rfftfreq(N, 1 / fs)
    mask = freqs <= cutoff
    freqs = freqs[mask]
    m = np.arange(M)[:, None]
    A = np.exp(-1j * 2 * np.pi * freqs[None, :] * m / fs)

    # Pseudospectro
    ps = 1.0 / np.sum(np.abs(np.conj(E_n).T @ A)**2, axis=0)

    # Pico
    peaks = np.argsort(ps)[-K:]
    est_freqs = freqs[peaks]
    est_mags = ps[peaks]
    order_idx = np.argsort(est_freqs)
    return est_freqs[order_idx], est_mags[order_idx]
```

## Como Funciona no Simulador

1. Calcula autocorrelação e constrói matriz R.  
2. Instancia `MusicAnalyzer(order, sr, cutoff)`.  
3. Chama `analyze(signal)` para obter frequências e magnitudes.  
4. Gera pseudospectro e detecta picos.  
5. Exibe espectro MUSIC em vez do FFT convencional.

---

Este documento serve de base para implementação didática do método MUSIC no simulador. A seguir, crie a classe `MusicAnalyzer` que encapsula essa lógica.
