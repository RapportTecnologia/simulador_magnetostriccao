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

def gerar_sinal_base(label):
    base_freq = 60
    harmonicas = list(range(2, 16))
    sinal = np.zeros_like(t)

    if label == 0:
        for h in harmonicas:
            sinal += (1 / h) * np.sin(2 * np.pi * base_freq * h * t)
    elif label == 1:
        for h in harmonicas:
            ganho = 1 / (h * (1.2 if h % 2 == 0 else 0.8))
            sinal += ganho * np.sin(2 * np.pi * base_freq * h * t)
        sinal += 0.01 * np.random.randn(*t.shape)
    elif label == 2:
        for h in harmonicas:
            ganho = 1 / (h * (0.6 if h % 2 == 1 else 1.5))
            sinal += ganho * np.sin(2 * np.pi * base_freq * h * t)
        sinal += 0.03 * np.random.randn(*t.shape)

    sinal = sinal / np.max(np.abs(sinal))
    sinal += 0.005 * np.random.randn(*t.shape)
    return sinal

def carregar_audio_pydub(path):
    audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(fs)
    audio = audio[:duracao * 1000]
    return np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

def aplicar_ruido_real(sinal):
    arquivos_ruido = glob(os.path.join(DIR_RUIDOS, "*.wav"))
    if not arquivos_ruido:
        return sinal, "sem_ruido"
    arquivo = random.choice(arquivos_ruido)
    ruido = carregar_audio_pydub(arquivo)
    ruido = np.resize(ruido, len(sinal))
    sinal += 0.2 * ruido
    return sinal, os.path.splitext(os.path.basename(arquivo))[0]

def aplicar_convolucao_IR(sinal):
    arquivos_ir = glob(os.path.join(DIR_IRS, "*.wav"))
    if not arquivos_ir:
        return sinal
    ir = carregar_audio_pydub(random.choice(arquivos_ir))
    return fftconvolve(sinal, ir, mode='same')

def aplicar_equalizacao(sinal):
    arquivos_eq = glob(os.path.join(DIR_EQ_CURVES, "*.wav"))
    if not arquivos_eq:
        return sinal
    curva_eq = carregar_audio_pydub(random.choice(arquivos_eq))
    curva_eq = np.resize(curva_eq, len(sinal))
    return sinal * curva_eq

def salvar_wav(sinal, path):
    wav_data = np.int16(sinal / np.max(np.abs(sinal)) * 32767)
    sf.write(path, wav_data, fs)

def gerar_amostras(destino, total, proporcao_treino, proporcao_ruido=0.3):
    treino_dir = os.path.join(destino, "train")
    teste_dir = os.path.join(destino, "test")
    for d in [treino_dir, teste_dir]:
        for label in ['0', '1', '2']:
            os.makedirs(os.path.join(d, label), exist_ok=True)

    num_treino = int(total * proporcao_treino)

    for i in range(total):
        label = random.choice([0, 1, 2])
        sinal = gerar_sinal_base(label)
        ruido_nome = "sem_ruido"

        if i >= num_treino and random.random() < proporcao_ruido:
            sinal, ruido_nome = aplicar_ruido_real(sinal)
            if random.random() < 0.5:
                sinal = aplicar_convolucao_IR(sinal)
                sinal = aplicar_equalizacao(sinal)

        sinal = sinal / np.max(np.abs(sinal))

        nome = f"sinal_{i:04d}_label{label}_{ruido_nome}.wav"
        destino_final = treino_dir if i < num_treino else teste_dir
        salvar_wav(sinal, os.path.join(destino_final, str(label), nome))

# Geração de 10 amostras para teste
gerar_amostras("./samples_realistas", total=200, proporcao_treino=0.3, proporcao_ruido=0.3)
