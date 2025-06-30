"""Rotina de análise de áudio em tempo real.

Este módulo define uma QThread responsável por capturar som do microfone,
processá-lo e emitir resultados intermediários (espectrograma, Mel bands, FFT
e classificação) sem bloquear a interface gráfica.
"""

# analysis_thread.py - Thread para análise de arquivos de áudio
import time

import numpy as np
import scipy.signal
import librosa
import sounddevice as sd

from PyQt5.QtCore import QThread, pyqtSignal

# Constants duplicated from main application for analysis
CUTOFF_FREQ = 1000
N_MFCC = 40

class FileAnalysisThread(QThread):
    """
    Thread dedicado para processar arquivos de áudio de teste sem bloquear a GUI.
    Emite sinais com dados já processados: espectrograma, Mel bands, FFT e classificação.
    """
    # Thread de análise em tempo real não usa progresso de arquivos

    # Sinal para espectrograma pronto: frequências, matriz em dB
    spec_ready = pyqtSignal(np.ndarray, np.ndarray)

    # Sinal para Mel bands pronto: vetor de potência média em dB
    mel_ready = pyqtSignal(np.ndarray)

    # Sinal para FFT pronta: frequências, magnitudes
    fft_ready = pyqtSignal(np.ndarray, np.ndarray)

    # Sinal para classe prevista: código_da_classe
    class_ready = pyqtSignal(int)

    def __init__(self, b, a, sr, duration, model, max_time):
        """Inicializa a thread de análise de áudio.

        Parameters
        ----------
        b, a : ndarray
            Coeficientes do filtro Butterworth usado no pré-processamento.
        sr : int
            Taxa de amostragem utilizada na captura.
        duration : float
            Duração máxima de cada captura em segundos.
        model : keras.Model
            Modelo carregado responsável pela classificação.
        max_time : int
            Número de quadros esperados pelo modelo (eixo temporal dos MFCCs).
        """

        super().__init__()

        # Coeficientes do filtro passa-baixo (Butterworth)
        self.b = b
        self.a = a

        # Taxa de amostragem em Hertz
        self.sr = sr

        # Duração máxima de cada arquivo em segundos
        self.duration = duration

        # Flag para controle de execução
        self._running = True

        # Modelo e tempo máximo de quadros
        self.model = model
        self.max_time = max_time

    def run(self):
        """
        Método principal: monitora entrada de áudio e emite sinais em tempo real.
        """
        while self._running:
            # Grava áudio
            audio = sd.rec(int(self.duration * self.sr), samplerate=self.sr, channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()

            # Aplica filtro passa-baixo Butterworth
            filtered = scipy.signal.filtfilt(self.b, self.a, audio)

            # 1) Espectrograma
            S = librosa.stft(filtered)
            freqs = librosa.fft_frequencies(sr=self.sr)
            mask = freqs <= CUTOFF_FREQ
            S_db = librosa.amplitude_to_db(np.abs(S[mask, :]))
            self.spec_ready.emit(freqs[mask], S_db)

            # 2) Mel Spectrogram
            M = librosa.feature.melspectrogram(y=filtered, sr=self.sr, fmax=CUTOFF_FREQ)
            mel_db = librosa.power_to_db(M).mean(axis=1)
            self.mel_ready.emit(mel_db)

            # 3) FFT
            F = np.abs(np.fft.rfft(filtered))
            freqs_fft = np.fft.rfftfreq(len(filtered), 1 / self.sr)
            mask_fft = freqs_fft <= CUTOFF_FREQ
            self.fft_ready.emit(freqs_fft[mask_fft], F[mask_fft])

            # 4) Classificação
            mfccs = librosa.feature.mfcc(y=filtered, sr=self.sr, n_mfcc=N_MFCC, fmax=CUTOFF_FREQ)
            if mfccs.shape[1] >= self.max_time:
                mfccs = mfccs[:, :self.max_time]
            else:
                pad = np.zeros((N_MFCC, self.max_time - mfccs.shape[1]))
                mfccs = np.hstack((mfccs, pad))
            x = mfccs[np.newaxis, ..., np.newaxis]
            pred = self.model.predict(x)
            cls = int(np.argmax(pred))
            self.class_ready.emit(cls)

            time.sleep(0.05)

    def stop(self):
        """
        Desativa flag para parar a thread.
        """
        self._running = False

    @property
    def MODEL(self):
        """
        Acesso ao modelo carregado na thread principal.
        Deve ser configurado antes de iniciar a thread.
        """
        return self.parent().model

    @property
    def model_input_frames(self):
        """
        Retorna número de quadros temporais esperados pelo modelo.
        Obtido de model.input_shape.
        """
        return self.MODEL.input_shape[2]
