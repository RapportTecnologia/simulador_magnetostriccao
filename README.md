# Simulador e Analisador de Magnetostricção

Este projeto implementa um sistema completo para simulação, análise e classificação de ruídos de magnetostricção em transformadores. O sistema consiste em dois componentes principais:

1. **Gerador de Amostras**: Gera sinais sintéticos que simulam diferentes estados de ruídos de magnetostricção.
2. **Analisador em Tempo Real**: Interface gráfica para captura, processamento, visualização e classificação de sinais de áudio.

## Índice

- [Requisitos](#requisitos)
- [Configuração do Ambiente](#configuração-do-ambiente)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Geração de Amostras](#geração-de-amostras)
- [Uso do Analisador](#uso-do-analisador)
- [Redes Neurais Utilizadas](#redes-neurais-utilizadas)
- [Fluxo de Trabalho Completo](#fluxo-de-trabalho-completo)

## Requisitos

- Python 3.8 ou superior
- Bibliotecas Python (ver seção de configuração)
- Dispositivos de áudio para captura/reprodução (para uso em tempo real)

## Configuração do Ambiente

### Criação do Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente no Linux/Mac
source venv/bin/activate

# Ativar ambiente no Windows
venv\Scripts\activate
```

### Instalação de Dependências

```bash
# Instalar todas as dependências necessárias
pip install -r requirements.txt
```

As dependências principais incluem:
- **numpy**: Processamento numérico
- **scipy**: Processamento de sinais (filtros, transformadas)
- **librosa**: Análise de áudio e extração de características
- **sounddevice**: Interface para dispositivos de áudio
- **PyQt5**: Interface gráfica
- **matplotlib**: Visualização de dados
- **tensorflow**: Aprendizado de máquina para classificação
- **tqdm**: Barra de progresso para geração de amostras

## Estrutura do Projeto

```
simulador/
├── simulador.py                        # Interface gráfica e classificador
├── gerador_amostras_magnetostriccao.py # Gerador de amostras
├── requirements.txt                    # Dependências do projeto
├── ruidos_externos/                    # Diretório para sons ambientes
├── impulse_responses/                  # Respostas impulsivas para efeitos de sala
├── eq_curvas_microfone/                # Perfis de equalização de microfones
└── samples_realistas/                  # Amostras geradas para treino/teste
    ├── train/
    │   ├── 0/                          # Classe 0: Normal
    │   ├── 1/                          # Classe 1: Intermediário
    │   └── 2/                          # Classe 2: Falha
    └── test/
        ├── 0/                          # Classe 0: Normal
        ├── 1/                          # Classe 1: Intermediário
        └── 2/                          # Classe 2: Falha
```

## Geração de Amostras

O script `gerador_amostras_magnetostriccao.py` cria amostras de áudio sintéticas que simulam três estados de magnetostricção. 

A partir da versão mais recente, o processo de geração exibe uma **barra de progresso** no terminal, mostrando quantos arquivos de treino e teste já foram criados, facilitando o acompanhamento da execução.

- **Classe 0**: Estado normal (padrão harmônico equilibrado)
- **Classe 1**: Estado intermediário (alterações nas harmonicas pares)
- **Classe 2**: Estado de falha (alterações nas harmonicas ímpares e maior ruído)

### Como Funciona o Gerador

O gerador usa os seguintes processos para criar amostras realistas:

1. **Geração do Sinal Base**: Cria harmônicas de 60Hz com diferentes ganhos por classe
2. **Aplicação de Ruído Real**: (Opcional) Adiciona ruído ambiente de gravações reais
3. **Convolução com Resposta Impulsiva**: (Opcional) Aplica características acústicas de ambientes
4. **Equalização de Microfone**: (Opcional) Simula resposta em frequência de diferentes microfones

### Como Usar o Gerador

O script `gerador_amostras_magnetostriccao.py` pode ser executado diretamente pela linha de comando e aceita diversos parâmetros para personalização:

#### Principais argumentos

- `--fatorial`           : Gera todas as combinações possíveis entre sinais e todos os ruídos externos (modo fatorial)
- `--destino`            : Diretório de saída das amostras (default: ./samples_realistas)
- `--n_sinais`           : Número de sinais base por classe (no modo fatorial, default: 20)
- `--total`              : Número total de amostras (modo normal, default: 200)
- `--proporcao_treino`   : Proporção das amostras para treino (default: 0.3)
- `--proporcao_ruido`    : Proporção das amostras com ruído externo (default: 0.3)
- `--nivel-ruido` ou `-n`: Nível de amplificação do ruído externo (default: 1.5 = +50%)

#### Exemplos de uso

**Modo normal (aleatório, com proporção de ruído):**

```bash
python gerador_amostras_magnetostriccao.py --total 100 --proporcao_ruido 0.5 --nivel-ruido 2.0 --destino ./samples_realistas
```

**Modo fatorial (todas as combinações entre cada sinal e cada ruído externo):**

```bash
python gerador_amostras_magnetostriccao.py --fatorial --n_sinais 20 --nivel-ruido 1.5 --destino ./samples_fatorial
```

#### Observações importantes

- **Formatos aceitos para ruído externo:** `.wav` e `.mp3` (coloque os arquivos em `ruidos_externos/`)
- **Duração das amostras:** Quando ruído externo é utilizado, a duração da amostra gerada será igual à do arquivo de ruído externo correspondente.
- **Amplificação do ruído:** O parâmetro `--nivel-ruido` multiplica o volume do ruído externo (1.0 = original, 2.0 = dobro, 1.5 = +50%).
- **Saída:** O script cria pastas `train` e `test` no diretório de destino, cada uma com subpastas para as classes 0, 1 e 2.

#### Ajuda

Para ver todos os parâmetros disponíveis:

```bash
python gerador_amostras_magnetostriccao.py --help
```

### Personalização da Geração

Para personalizar a geração de amostras:

1. **Ruídos Externos**: Adicione arquivos de áudio WAV em `ruidos_externos/` (crie o diretório manualmente se não existir: `mkdir ruidos_externos`)
2. **Respostas Impulsivas**: Adicione IRs de ambientes em `impulse_responses/`
3. **Curvas de EQ**: Adicione perfis de equalização em `eq_curvas_microfone/`

## Uso do Analisador

O analisador (`simulador.py`) é uma aplicação gráfica para:
- Captura e reprodução de áudio em tempo real
- Processamento e visualização de características espectrais
- Classificação de sinais usando redes neurais treinadas

### Execução do Analisador

```bash
# Formato básico
python simulador.py <diretório_raiz>

# Exemplo com modelo pré-treinado
python simulador.py ./samples_realistas --model model_CNN.h5

# O diretório_raiz deve conter subpastas:
#   train/0, train/1, train/2 (para treinamento)
#   test/0, test/1, test/2 (para teste)
```

### Funcionalidades da Interface

![Interface do Analisador (representação)](https://exemplo.com/interface.png)

1. **Seleção de Dispositivos**:
   - Escolha de entrada de áudio (microfone)
   - Escolha de saída de áudio

2. **Controles de Áudio**:
   - Ganho de entrada
   - Ganho de saída
   - VU-meter para visualização do nível RMS

3. **Visualizações em Tempo Real**:
   - Espectrograma (até 1 kHz)
   - Bandas Mel
   - FFT

4. **Operações de Classificação**:
   - Treinamento de modelos
   - Classificação em tempo real
   - Classificação de arquivos de teste
   - Indicador visual do estado detectado

### Fluxo de Trabalho na Interface

1. **Treinar Modelo**:
   - Clique em "Treinar" para criar um novo modelo
   - Selecione a arquitetura desejada na lista
   - Aguarde o treinamento e salvamento do modelo

2. **Classificação em Tempo Real**:
   - Clique em "Iniciar" para começar a captura
   - Observe os gráficos e o status em tempo real
   - Clique em "Parar" para interromper

3. **Classificação de Testes**:
   - Clique em "Testar" para classificar arquivos de teste
   - Observe os resultados no console
   - A classificação pode ser interrompida a qualquer momento

## Redes Neurais Utilizadas

O sistema possui uma interface flexível para treino e teste de diferentes arquiteturas de redes neurais para classificação dos sinais de áudio. As opções disponíveis na interface são:

- **CNN** (Rede Neural Convolucional)
- **RNN** (Rede Neural Recorrente)
- **SVM** (Support Vector Machine)
- **RandomForest** (Floresta Aleatória)
- **XGBoost** (Extreme Gradient Boosting)

> **Nota:** No momento, apenas a arquitetura CNN está implementada e pode ser treinada diretamente pela interface. As demais (RNN, SVM, RandomForest, XGBoost) estão presentes para fins de comparação futura e seleção visual, mas ainda não estão disponíveis para treino.

### CNN (Rede Neural Convolucional)
A CNN é a arquitetura padrão utilizada para classificação dos MFCCs extraídos dos sinais de áudio. Ela é composta pelas seguintes camadas:

```
Entrada: MFCCs [n_mfcc=40, frames]
↓
Convolução 2D (16 filtros 3x3, ReLU)
↓
MaxPooling 2D (2x2)
↓
Convolução 2D (32 filtros 3x3, ReLU)
↓
MaxPooling 2D (2x2)
↓
Flatten
↓
Densa (64 neurônios, ReLU)
↓
Saída (3 classes, Softmax)
```

- **Entrada:** MFCCs extraídos do áudio (formato imagem 2D)
- **Conv2D/MaxPooling:** Capturam padrões espectrais e temporais relevantes
- **Flatten/Dense:** Realizam a classificação final em 3 classes

### Outras Arquiteturas Previstas

- **RNN:** Indicada para sequências temporais, pode capturar dependências de longo prazo nos sinais.
- **SVM:** Classificador tradicional eficiente para problemas lineares e não-lineares com poucos dados.
- **RandomForest:** Método de ensemble baseado em árvores de decisão, robusto a ruídos e sobreajuste.
- **XGBoost:** Algoritmo de boosting de árvores, eficiente para grandes conjuntos de dados tabulares.

Essas opções estão presentes na interface para facilitar testes comparativos e futuras expansões do sistema.

### Características e Pré-processamento

- **MFCCs**: Extração de 40 Coeficientes Cepstrais de Frequência Mel
- **Filtragem**: Passa-baixa até 1 kHz usando filtro Butterworth de 4ª ordem
- **Padronização**: Todas as amostras são normalizadas e padronizadas para o mesmo tamanho

### Vantagens da CNN para Este Problema

1. **Captura de padrões espectrais**: As camadas convolucionais identificam padrões relevantes nos espectrogramas de MFCC
2. **Robustez a variações temporais**: O pooling fornece certa invariância a pequenas diferenças temporais
3. **Compacidade**: A rede é pequena o suficiente para rodar em tempo real em hardware comum
4. **Generalização**: Boa capacidade de detectar padrões em condições de ruído variável

### Limitações

1. **Necessidade de dados representativos**: A qualidade da classificação depende da variedade do conjunto de treinamento
2. **Sensibilidade a ruídos não vistos**: Ruídos muito diferentes dos usados no treinamento podem afetar o desempenho
3. **Falta de explicabilidade**: Como em muitas redes neurais, as decisões não são totalmente interpretáveis
4. **Limitação a 1 kHz**: A análise é limitada a baixas frequências, podendo perder informações em frequências mais altas

## Fluxo de Trabalho Completo

Para utilizar o sistema completo:

1. **Gerar ruídos urbanos** (opcional, para enriquecer os testes):
   ```bash
   python gerar_ruidos_urbanos.py
   ```

2. **Gerar amostras sintéticas**:
   ```bash
   python gerador_amostras_magnetostriccao.py
   ```

3. **Opcional**: Adicionar gravações reais às pastas train/test correspondentes

4. **Executar o analisador**:
   ```bash
   python simulador.py ./samples_realistas
   ```

5. **Treinar um modelo** na interface gráfica
   - Ao trocar o modelo na lista, os botões de análise e classificação serão desativados automaticamente. É necessário treinar novamente para habilitá-los.
   - O arquivo salvo terá sempre o formato `model_<nome_modelo>.h5` (ex: `model_CNN.h5`).

6. **Testar o modelo** com amostras de teste

7. **Usar em tempo real** com um microfone para monitoramento contínuo

---

## Uso de TinyML no STM32

Este projeto oferece um script para converter modelos Keras (.h5) em formatos compatíveis com TinyML para microcontroladores STM32 (TFLite e array C).

### Passos para Converter e Integrar

1. **Converta o modelo .h5 para TFLite e array C:**

   ```bash
   python h5_to_stm32_tinyml.py caminho/do/modelo.h5
   # Saídas: modelo.tflite e modelo.h (array C)
   ```
   
   Parâmetros opcionais:
   - `--tflite`: Caminho de saída do arquivo .tflite
   - `--c`: Caminho de saída do arquivo .h (array C)
   - `--var`: Nome da variável C (default: model_tflite)

2. **Projeto Base STM32:**
   - Um esqueleto de projeto está disponível em `stm32_base/`.
   - Copie o arquivo `.h` gerado para `stm32_base/Core/Inc/model_tflite.h`.
   - Siga o guia em `stm32_base/README_STM32_PROJECT.md` para integração no seu projeto STM32.

3. **Integração no Código C:**
   - Inclua o arquivo `.h` no seu código STM32:
     ```c
     #include "model_tflite.h"
     // Use model_tflite e model_tflite_len
     ```
   - Importe e integre a biblioteca TensorFlow Lite Micro conforme instruções no projeto base.

### Dependências para Conversão
- Python 3.8+
- tensorflow
- numpy

Instale com:
```bash
pip install tensorflow numpy
```

---

Este projeto foi desenvolvido para auxiliar na detecção e classificação de problemas de magnetostricção em transformadores, fornecendo ferramentas para simulação, treinamento, aplicação em tempo real e agora integração com microcontroladores STM32 via TinyML.
