#!/usr/bin/env python3
"""
Gerador de Ruídos Urbanos Sintéticos

Este script gera arquivos WAV simulando diferentes tipos de ruídos urbanos.
Os arquivos são salvos no diretório ruidos_externos e podem ser usados como
complemento na geração de amostras de magnetostricção.
"""

import os
import numpy as np
from scipy import signal
import soundfile as sf
import random

# Diretório para salvar os ruídos
RUIDOS_DIR = "./ruidos_externos"
os.makedirs(RUIDOS_DIR, exist_ok=True)

# Taxa de amostragem (Hz)
SR = 8000

# Duração dos arquivos (segundos)
DURACAO = 5

def normalizar(sinal):
    """Normaliza o sinal para range [-1, 1]"""
    if np.max(np.abs(sinal)) > 0:
        return sinal / np.max(np.abs(sinal))
    return sinal

def gerar_ruido_branco(amplitude=0.1):
    """Gera ruído branco básico com amplitude específica"""
    return amplitude * np.random.randn(SR * DURACAO)

def gerar_ruido_filtrado(tipo, freq_corte, ordem=4, amplitude=0.1):
    """Gera ruído com filtro específico (lowpass, highpass, bandpass)"""
    ruido = gerar_ruido_branco(amplitude=1.0)
    nyquist = 0.5 * SR
    
    if tipo == 'lowpass':
        b, a = signal.butter(ordem, freq_corte / nyquist, btype='lowpass')
    elif tipo == 'highpass':
        b, a = signal.butter(ordem, freq_corte / nyquist, btype='highpass')
    elif tipo == 'bandpass':
        # Para bandpass, freq_corte deve ser uma tupla (low, high)
        low, high = freq_corte
        b, a = signal.butter(ordem, [low / nyquist, high / nyquist], btype='bandpass')
    else:
        raise ValueError(f"Tipo de filtro desconhecido: {tipo}")
        
    ruido_filtrado = signal.filtfilt(b, a, ruido)
    return amplitude * normalizar(ruido_filtrado)

def gerar_pulsos_aleatorios(frequencia=2.0, amplitude=0.3, largura=0.05):
    """Gera pulsos aleatórios (simula sons impulsivos como buzinas)"""
    samples = SR * DURACAO
    t = np.linspace(0, DURACAO, samples, endpoint=False)
    sinal = np.zeros_like(t)
    
    # Número médio de pulsos baseado na frequência
    n_pulsos = int(frequencia * DURACAO)
    
    # Posições aleatórias para os pulsos
    posicoes = np.random.uniform(0, DURACAO, n_pulsos)
    
    # Largura do pulso em segundos
    largura_samples = int(largura * SR)
    
    # Gerar cada pulso
    for pos in posicoes:
        pos_sample = int(pos * SR)
        inicio = max(0, pos_sample - largura_samples // 2)
        fim = min(samples, pos_sample + largura_samples // 2)
        
        # Forma de onda do pulso (gaussiana)
        meio = (inicio + fim) // 2
        indice = np.arange(inicio, fim)
        pulso = amplitude * np.exp(-0.5 * ((indice - meio) / (largura_samples / 6)) ** 2)
        sinal[inicio:fim] += pulso
    
    return sinal

def gerar_som_trafego():
    """Simula ruído de tráfego urbano"""
    # Base: ruído marrom (filtrado) + ruído filtrado em banda
    ruido_base = gerar_ruido_filtrado('lowpass', 200, amplitude=0.4)
    ruido_medio = gerar_ruido_filtrado('bandpass', (100, 800), amplitude=0.3)
    
    # Adicionar alguns pulsos aleatórios (carros passando)
    pulsos = gerar_pulsos_aleatorios(frequencia=1.5, amplitude=0.2)
    
    # Combinar componentes
    sinal = ruido_base + ruido_medio + pulsos
    return normalizar(sinal)

def gerar_som_construcao():
    """Simula ruído de canteiro de obras"""
    # Base: ruído mais intenso nas frequências médias
    ruido_base = gerar_ruido_filtrado('bandpass', (300, 1500), amplitude=0.4)
    
    # Adicionar pulsos mais intensos e frequentes (martelos, etc)
    pulsos = gerar_pulsos_aleatorios(frequencia=4.0, amplitude=0.6)
    
    # Adicionar componente irregular
    ruido_irregular = gerar_ruido_filtrado('highpass', 1000, amplitude=0.2)
    
    # Combinar componentes
    sinal = ruido_base + pulsos + ruido_irregular
    return normalizar(sinal)

def gerar_som_restaurante():
    """Simula ambiente de restaurante/café"""
    # Base: murmúrio de vozes (ruído filtrado em bandas específicas)
    murmurinho = gerar_ruido_filtrado('bandpass', (200, 800), amplitude=0.3)
    
    # Adicionar pulsos esparsos (copos, risadas)
    pulsos = gerar_pulsos_aleatorios(frequencia=0.8, amplitude=0.2)
    
    # Adicionar fundo musical suave
    ruido_fundo = gerar_ruido_filtrado('bandpass', (400, 600), amplitude=0.1)
    
    # Combinar componentes
    sinal = murmurinho + pulsos + ruido_fundo
    return normalizar(sinal)

def gerar_som_chuva():
    """Simula chuva em ambiente urbano"""
    # Base: ruído branco para a chuva geral
    ruido_base = gerar_ruido_branco(amplitude=0.3)
    
    # Filtrar para dar característica de chuva
    b, a = signal.butter(2, 2000 / (0.5 * SR), btype='lowpass')
    ruido_filtrado = signal.filtfilt(b, a, ruido_base)
    
    # Adicionar algumas gotas individuais mais fortes
    pulsos = gerar_pulsos_aleatorios(frequencia=6.0, amplitude=0.1, largura=0.02)
    
    # Combinar componentes
    sinal = ruido_filtrado + pulsos
    return normalizar(sinal)

def gerar_som_metro():
    """Simula estação de metrô"""
    # Base: ruído constante e grave de fundo
    ruido_base = gerar_ruido_filtrado('lowpass', 150, amplitude=0.2)
    
    # Adicionar ruído mecânico de média frequência
    ruido_medio = gerar_ruido_filtrado('bandpass', (400, 1200), amplitude=0.15)
    
    # Adicionar pulsos para simular sons de freios e portas
    pulsos = gerar_pulsos_aleatorios(frequencia=0.5, amplitude=0.4)
    
    # Combinar componentes
    sinal = ruido_base + ruido_medio + pulsos
    return normalizar(sinal)

def salvar_wav(sinal, nome):
    """Salva o sinal como arquivo WAV"""
    caminho = os.path.join(RUIDOS_DIR, nome)
    sf.write(caminho, sinal, SR)
    print(f"Arquivo criado: {caminho}")

def main():
    """Função principal - gera todos os tipos de ruídos urbanos"""
    print(f"Gerando ruídos urbanos em {RUIDOS_DIR}...")
    
    # Gerar e salvar ruídos
    salvar_wav(gerar_som_trafego(), "ruido_urbano_trafego.wav")
    salvar_wav(gerar_som_construcao(), "ruido_urbano_construcao.wav")
    salvar_wav(gerar_som_restaurante(), "ruido_urbano_restaurante.wav")
    salvar_wav(gerar_som_chuva(), "ruido_urbano_chuva.wav")
    salvar_wav(gerar_som_metro(), "ruido_urbano_metro.wav")
    
    # Gerar alguns ruídos aleatórios com diferentes filtros
    tipos_filtro = ['lowpass', 'highpass', 'bandpass']
    for i in range(3):
        tipo = random.choice(tipos_filtro)
        if tipo == 'bandpass':
            low = random.uniform(100, 500)
            high = low + random.uniform(500, 1500)
            freq = (low, high)
        else:
            freq = random.uniform(200, 1500)
        
        sinal = gerar_ruido_filtrado(tipo, freq, amplitude=0.5)
        salvar_wav(sinal, f"ruido_urbano_aleatorio_{i+1}.wav")
    
    print(f"\nTotal de {5 + 3} arquivos de ruído urbano gerados com sucesso.")

if __name__ == "__main__":
    main()
