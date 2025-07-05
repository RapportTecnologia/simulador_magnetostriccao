# Análise de Prony

## Sumário

A análise de Prony serve para estimar os parâmetros de um modelo de soma de exponenciais (amortecidas ou não) a partir de sinais discretos. É especialmente útil em análise modal, caracterização de sistemas dinâmicos e identificação de frequências e taxas de amortecimento.

## Princípios Fundamentais

1. **Modelo Exponencial**  
   Um sinal `x[n]` é modelado como soma de `K` exponenciais:
   ```
   x[n] = \sum_{k=1}^{K} A_k z_k^n,  \quad n = 0,1,\dots,N-1  
   ```
   - `A_k` são amplitudes complexas.  
   - `z_k = e^{(-\alpha_k + j\omega_k) T_s}` contém taxa de amortecimento `α_k` e frequência angular `ω_k` (com período de amostragem `T_s`).

2. **Equação de Previsão Linear**  
   Do modelo, obtém-se que cada ponto é combinação linear dos anteriores:
   ```
   x[n] + a_1 x[n-1] + \dots + a_K x[n-K] = 0.  
   ```
   Os coeficientes `a_i` relacionam-se às raízes `z_k` do polinômio característico.

3. **Estimativa dos Coeficientes**  
   - Monta-se um sistema de equações lineares usando janelas de tamanho `N`:
     ```
     \begin{bmatrix}
x[K]   & x[K-1] & \dots & x[0]   \\
x[K+1] & x[K]   & \dots & x[1]   \\
\vdots &        & \ddots& \vdots \\
x[N-1] & x[N-2] & \dots & x[N-K-1]
\end{bmatrix}
     \begin{bmatrix} a_1\\a_2\\\vdots\\a_K \end{bmatrix} = -
     \begin{bmatrix} x[K+1]\\x[K+2]\\\vdots\\x[N] \end{bmatrix}
     ```
   - Resolve-se `(Hankel) · a = -v` por mínimos quadrados.

4. **Extração de Polos**  
   - As raízes `z_k` são obtidas a partir do polinômio característico:
     ```
     z^K + a_1 z^{K-1} + \dots + a_K = 0.
     ```
   - A partir de `z_k`, calcula-se frequência `f_k = \arg(z_k)/(2\pi T_s)` e amortecimento `α_k = -\ln|z_k|/T_s`.

5. **Cálculo das Amplitudes**  
   - Resolvem-se amplitudes `A_k` em um sistema linear usando os polos e amostras iniciais.

## Vantagens e Desvantagens

- **Vantagens**:
  - Alta resolução em sinais curtos.  
  - Separa modos próximos em frequência.  
- **Desvantagens**:
  - Sensível a ruído e ordem `K` mal escolhida.  
  - Requer solução de sistemas potencialmente mal condicionados.

## Fluxo de Implementação em Python

```python
import numpy as np
from numpy.linalg import lstsq

def prony(x, K, fs):
    """
    Estima polos e amplitudes do sinal x usando método de Prony.
    :param x: array de sinal (1D)
    :param K: ordem do modelo (número de exponenciais)
    :param fs: taxa de amostragem
    :return: (freqs, amps, damp)
    """
    N = len(x)
    # Monta matriz Hankel (N-K x K)
    H = np.column_stack([x[i:N-K+i] for i in range(K)])
    # Vetor alvo
    y = -x[K:N]
    # Resolve equação H · a = y
    a, *_ = lstsq(H, y, rcond=None)
    # Polos como raízes do polinômio
    coeffs = np.concatenate(([1], a))
    poles = np.roots(coeffs)
    # Extração de frequência e amortecimento
    Ts = 1/fs
    freqs = np.angle(poles)/(2*np.pi*Ts)
    damp = -np.log(np.abs(poles))/Ts
    # Cálculo de amplitudes
    # Monta Vandermonde para K amostras iniciais
    V = np.vander(poles, N=K, increasing=True).T
    A, *_ = lstsq(V, x[:K], rcond=None)
    return freqs, A, damp
```

## Como Funciona no Simulador

1. O sinal é filtrado e pré-processado.  
2. Instancia-se `PronyAnalyzer` para gerar polos e amplitudes.  
3. Converte-se resultados em espectro com picos bem definidos.  
4. O gráfico de FFT original é substituído pelo gráfico de frequências estimadas pelo Prony.  

---
Este documento serve de base para implementação didática do método de Prony no simulador. A seguir, crie a classe `PronyAnalyzer` que encapsula essa lógica.
