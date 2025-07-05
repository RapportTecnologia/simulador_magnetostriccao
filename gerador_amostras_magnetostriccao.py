"""Gerador de amostras sintéticas de magnetostricção."""

import os
import numpy as np
import random
from scipy.signal import fftconvolve
import soundfile as sf
from glob import glob
from pydub import AudioSegment

# Parâmetros globais
fs = 8000
duracao = 5
n_amostras = fs * duracao
t = np.linspace(0, duracao, n_amostras, endpoint=False)

# Diretórios dos ruídos e IRs
DIR_RUIDOS = "./ruidos_externos"
DIR_IRS = "./impulse_responses"
DIR_EQ_CURVES = "./eq_curvas_microfone"

def gerar_sinal_base(label, sem_ruido=False):
    """Gera sinal fundamental com harmônicas simulando magnetostricção."""

    base_freq = 60 + np.random.uniform(-2, 2)  # Pequena variação de frequência fundamental
    harmonicas = list(range(2, 16))
    sinal = np.zeros_like(t)
    # Sorteia fase e amplitude aleatória para cada harmônico
    for h in harmonicas:
        amp = np.random.uniform(0.5, 1.2) / h
        phase = np.random.uniform(0, 2 * np.pi)
        freq = base_freq * h + np.random.uniform(-0.5, 0.5)  # Variação por harmônico
        if label == 0:
            sinal += amp * np.sin(2 * np.pi * freq * t + phase)
        elif label == 1:
            ganho = amp * (np.random.uniform(0.8, 1.3) if h % 2 == 0 else np.random.uniform(0.6, 1.1))
            sinal += ganho * np.sin(2 * np.pi * freq * t + phase)
        elif label == 2:
            ganho = amp * (np.random.uniform(0.4, 0.9) if h % 2 == 1 else np.random.uniform(1.0, 1.6))
            sinal += ganho * np.sin(2 * np.pi * freq * t + phase)
    if not sem_ruido:
        # Ruído de fundo leve
        if label == 1:
            sinal += 0.01 * np.random.randn(*t.shape)
        elif label == 2:
            sinal += 0.03 * np.random.randn(*t.shape)
        sinal = sinal / np.max(np.abs(sinal) + 1e-6)
        sinal += 0.005 * np.random.randn(*t.shape)
    else:
        sinal = sinal / np.max(np.abs(sinal) + 1e-6)
    return sinal

def carregar_audio_pydub(path):
    """Carrega áudio usando pydub e converte para numpy."""

    audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(fs)
    arr = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    dur = len(arr) / fs
    return arr, dur

def aplicar_ruido_real(sinal, nivel_ruido=1.5):
    """Combina o sinal base com um ruído externo real."""

    arquivos_ruido = glob(os.path.join(DIR_RUIDOS, "*.wav")) + glob(os.path.join(DIR_RUIDOS, "*.mp3"))
    if not arquivos_ruido:
        return sinal, "sem_ruido"
    arquivo = random.choice(arquivos_ruido)
    ruido, dur = carregar_audio_pydub(arquivo)
    n_amostras = len(ruido)
    t_ruido = np.linspace(0, dur, n_amostras, endpoint=False)
    # Gere o sinal base do mesmo tamanho do ruído
    sinal_base = gerar_sinal_base_com_duracao(len(ruido))
    sinal = sinal_base + 0.2 * nivel_ruido * ruido
    return sinal, os.path.splitext(os.path.basename(arquivo))[0]

def aplicar_convolucao_IR(sinal):
    """Aplica resposta impulsiva para simular ambiente."""

    arquivos_ir = glob(os.path.join(DIR_IRS, "*.wav"))
    if not arquivos_ir:
        return sinal
    ir = carregar_audio_pydub(random.choice(arquivos_ir))
    return fftconvolve(sinal, ir, mode='same')

def aplicar_equalizacao(sinal):
    """Aplica curva de equalização de microfone ao sinal."""

    arquivos_eq = glob(os.path.join(DIR_EQ_CURVES, "*.wav"))
    if not arquivos_eq:
        return sinal
    curva_eq = carregar_audio_pydub(random.choice(arquivos_eq))
    curva_eq = np.resize(curva_eq, len(sinal))
    return sinal * curva_eq

def salvar_wav(sinal, path):
    """Salva array numpy como arquivo WAV normalizado."""

    wav_data = np.int16(sinal / np.max(np.abs(sinal)) * 32767)
    sf.write(path, wav_data, fs)

from tqdm import tqdm

def gerar_amostras_fatorial(destino, n_sinais=20, nivel_ruido=1.5):
    """
    Para cada label (0, 1, 2), gera n_sinais de magnetostricção e combina cada um com TODOS os ruídos externos,
    salvando cada combinação como um arquivo de teste. O diretório de treino permanece vazio.
    """
    treino_dir = os.path.join(destino, "train")
    teste_dir = os.path.join(destino, "test")
    for d in [treino_dir, teste_dir]:
        for label in ['0', '1', '2']:
            os.makedirs(os.path.join(d, label), exist_ok=True)

    arquivos_ruido = glob(os.path.join(DIR_RUIDOS, "*.wav")) + glob(os.path.join(DIR_RUIDOS, "*.mp3"))
    if not arquivos_ruido:
        print("Nenhum ruído externo encontrado em", DIR_RUIDOS)
        return

    idx = 0
    total = 3 * n_sinais * len(arquivos_ruido)
    with tqdm(total=total, desc="Gerando amostras fatorial (test)") as pbar:
        for label in [0, 1, 2]:
            for i in range(n_sinais):
                for ruido_path in arquivos_ruido:
                    ruido, dur = carregar_audio_pydub(ruido_path)
                    n_amostras = len(ruido)
                    sinal_base = gerar_sinal_base_com_duracao(n_amostras)
                    sinal = sinal_base + 0.2 * nivel_ruido * ruido
                    sinal = sinal / np.max(np.abs(sinal))
                    ruido_nome = os.path.splitext(os.path.basename(ruido_path))[0]
                    nome = f"sinal_{idx:05d}_label{label}_{i:02d}_{ruido_nome}.wav"
                    salvar_wav(sinal, os.path.join(teste_dir, str(label), nome))
                    idx += 1
                    pbar.update(1)
    print(f"Foram geradas {idx} amostras combinando sinais e ruídos externos.")

# Função original mantida para compatibilidade e geração padrão

def gerar_amostras(destino, total, proporcao_treino, proporcao_ruido=0.3, nivel_ruido=1.5):
    """Gera dataset de treino/teste com opção de ruído externo."""

    treino_dir = os.path.join(destino, "train")
    teste_dir = os.path.join(destino, "test")
    for d in [treino_dir, teste_dir]:
        for label in ['0', '1', '2']:
            os.makedirs(os.path.join(d, label), exist_ok=True)

    num_treino = int(total * proporcao_treino)

    with tqdm(total=total, desc="Gerando amostras") as pbar:
        for i in range(total):
            label = random.choice([0, 1, 2])
            # Amostras de treino SEM ruído, amostras de teste com ruído
            if i < num_treino:
                sinal = gerar_sinal_base(label, sem_ruido=True)
                ruido_nome = "sem_ruido"
            else:
                sinal = gerar_sinal_base(label)
                ruido_nome = "sem_ruido"

            if i >= num_treino and random.random() < proporcao_ruido:
                arquivos_ruido = glob(os.path.join(DIR_RUIDOS, "*.wav")) + glob(os.path.join(DIR_RUIDOS, "*.mp3"))
                if arquivos_ruido:
                    arquivo = random.choice(arquivos_ruido)
                    ruido, dur = carregar_audio_pydub(arquivo)
                    n_amostras = len(ruido)
                    sinal_base = gerar_sinal_base_com_duracao(n_amostras)
                    sinal = sinal_base + 0.2 * nivel_ruido * ruido
                    ruido_nome = os.path.splitext(os.path.basename(arquivo))[0]
                else:
                    sinal = gerar_sinal_base(label)
                    ruido_nome = "sem_ruido"
                if random.random() < 0.5:
                    sinal = aplicar_convolucao_IR(sinal)
                    sinal = aplicar_equalizacao(sinal)

            sinal = sinal / np.max(np.abs(sinal))

            nome = f"sinal_{i:04d}_label{label}_{ruido_nome}.wav"
            destino_final = treino_dir if i < num_treino else teste_dir
            salvar_wav(sinal, os.path.join(destino_final, str(label), nome))
            pbar.update(1)

# Geração de amostras fatorial: cada sinal com cada ruído externo
# Exemplo: para 20 sinais por label e todos os ruídos em ./ruidos_externos
# O diretório de saída será ./samples_fatorial
def gerar_sinal_base_com_duracao(n_amostras, label=None):
    """Versão de ``gerar_sinal_base`` que respeita um tamanho arbitrário."""

    base_freq = 60 + np.random.uniform(-2, 2)
    harmonicas = list(range(2, 16))
    t_local = np.linspace(0, n_amostras / fs, n_amostras, endpoint=False)
    if label is None:
        label = random.choice([0, 1, 2])
    sinal = np.zeros_like(t_local)
    for h in harmonicas:
        amp = np.random.uniform(0.5, 1.2) / h
        phase = np.random.uniform(0, 2 * np.pi)
        freq = base_freq * h + np.random.uniform(-0.5, 0.5)
        if label == 0:
            sinal += amp * np.sin(2 * np.pi * freq * t_local + phase)
        elif label == 1:
            ganho = amp * (np.random.uniform(0.8, 1.3) if h % 2 == 0 else np.random.uniform(0.6, 1.1))
            sinal += ganho * np.sin(2 * np.pi * freq * t_local + phase)
        elif label == 2:
            ganho = amp * (np.random.uniform(0.4, 0.9) if h % 2 == 1 else np.random.uniform(1.0, 1.6))
            sinal += ganho * np.sin(2 * np.pi * freq * t_local + phase)
    if label == 1:
        sinal += 0.01 * np.random.randn(*t_local.shape)
    elif label == 2:
        sinal += 0.03 * np.random.randn(*t_local.shape)
    sinal = sinal / np.max(np.abs(sinal) + 1e-6)
    sinal += 0.005 * np.random.randn(*t_local.shape)
    return sinal

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gerador de amostras de magnetostricção com ruídos externos.")
    parser.add_argument('--fatorial', action='store_true', help='Gera todas as combinações fatorias de sinais e ruídos externos')
    parser.add_argument('--destino', type=str, default="./samples_realistas", help='Diretório de saída')
    parser.add_argument('--n_sinais', type=int, default=20, help='Número de sinais base por label (fatorial)')
    parser.add_argument('--total', type=int, default=200, help='Total de amostras (modo normal)')
    parser.add_argument('--proporcao_treino', type=float, default=0.3, help='Proporção de treino (modo normal)')
    parser.add_argument('--proporcao_ruido', type=float, default=0.3, help='Proporção de ruído (modo normal)')
    parser.add_argument('--nivel-ruido', '-n', type=float, default=1.5, help='Nível de aumento do ruído externo (1.0 = original, 1.5 = +50%)')
    args = parser.parse_args()

    if args.fatorial:
        gerar_amostras_fatorial(args.destino, n_sinais=args.n_sinais, nivel_ruido=args.nivel_ruido)
    else:
        gerar_amostras(args.destino, total=args.total, proporcao_treino=args.proporcao_treino, proporcao_ruido=args.proporcao_ruido, nivel_ruido=args.nivel_ruido)

# Geração de 10 amostras para teste
# gerar_amostras("./samples_realistas", total=200, proporcao_treino=0.3, proporcao_ruido=0.3)
