# Análise de ESPRIT

## Sumário

O método ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques) estima parâmetros de sinais compósitos (frequências e magnitudes) a partir de um subespaço de sinal, sem necessidade de busca exaustiva. É amplamente usado em análise modal e processamento de sinais multicomponentes.

## Princípios Fundamentais

1. **Sinal Analítico**  
   Converte sinal real em sinal complexo analítico via transformada de Hilbert para preservar fase.

2. **Matriz de Dados**  
   Monta matriz de dados com M = 2K linhas e L colunas:
   ```
   X = [x[0:L]; x[1:L+1]; ...; x[M-1:L+M-1]]
   ```

3. **Subespaço de Sinal**  
   Calcula matriz de covariância R = X·Xᴴ / L e extrai subespaço de sinal pelos K maiores autovalores.

4. **Rotational Invariance**  
   Separa submatrizes Es1 e Es2 do subespaço de sinal e resolve o operador rotacional ψ = Es1ᵀ⁺ · Es2. Autovalores de ψ contêm informação de frequência.

5. **Estimativa de Amplidões**  
   Estima magnitudes via mínimos quadrados usando matriz de Vandermonde construída a partir dos autovalores.

## Vantagens e Desvantagens

- **Vantagens**:  
  - Alta resolução sem busca em grade.  
  - Pouco sensível à escolha de pontos de grade.
- **Desvantagens**:  
  - Requer inversão de matriz e cuidado numérico.  
  - Sensível a ruído e dimensão M mal dimensionada.

## Fluxo de Implementação em Python

```python
import numpy as np
from numpy.linalg import eig, pinv, lstsq
from scipy.signal import hilbert

def esprit(x, K, fs, cutoff):
    """
    Estimate frequencies and magnitudes using the ESPRIT algorithm.
    :param x: array de sinal (1D)
    :param K: número de componentes
    :param fs: taxa de amostragem
    :param cutoff: frequência de corte (Hz)
    :return: (freqs, mags)
    """
    # Converte sinal real em analítico
    if not np.iscomplexobj(x):
        x = hilbert(x)

    N = len(x)
    M = 2 * K
    if N < M:
        raise ValueError(f"Comprimento {N} menor que dimensão {M} para ESPRIT")
    L = N - M + 1

    # Matriz de dados
    X = np.vstack([x[i:i+L] for i in range(M)])

    # Covariância e subespaço de sinal
    R = X @ X.conj().T / L
    e_vals, e_vecs = eig(R)
    idx = np.argsort(e_vals)[::-1]
    Es = e_vecs[:, idx[:K]]

    # Rotational invariance
    Es1 = Es[:-1, :]
    Es2 = Es[1:, :]
    psi = pinv(Es1) @ Es2
    lambdas, _ = eig(psi)

    # Frequências
    Ts = 1 / fs
    freqs = np.angle(lambdas) / (2 * np.pi * Ts)

    # Amplidões via LS
    S = np.vander(lambdas, N, increasing=True).T
    a, *_ = lstsq(S, x, rcond=None)
    mags = np.abs(a)

    # Filtra e seleciona top K
    mask = freqs <= cutoff
    freqs = freqs[mask]
    mags = mags[mask]
    peaks = np.argsort(mags)[-K:]
    est_freqs = freqs[peaks]
    est_mags = mags[peaks]
    return np.sort(est_freqs), est_mags[np.argsort(est_freqs)]
```

## Como Funciona no Simulador

1. O sinal real é convertido em sinal analítico via `hilbert`.  
2. Instancia-se `EspritAnalyzer` para estimar frequências e magnitudes.  
3. Aplica-se filtro de `cutoff` para descartar componentes indesejados.  
4. Os picos de magnitude identificam os modos dominantes.  
5. Substitui-se o gráfico de FFT por espectro estimado pelo ESPRIT.

---

Este documento serve de base para implementação didática do método ESPRIT no simulador. A seguir, crie a classe `EspritAnalyzer` que encapsula essa lógica.
