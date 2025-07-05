# analysis_thread.py - Thread para análise de arquivos de áudio
import time

import numpy as np
import scipy.signal
import librosa
import sounddevice as sd

from PyQt5.QtCore import QThread, pyqtSignal
from .analyzers.factory import AnalyzerFactory

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

    def __init__(self, b, a, sr, duration, model, max_time, analysis_method='fft', analysis_order=None):
        super().__init__()

        # Lista de caminhos para arquivos de teste
        # Não armazena lista de arquivos, usa entrada de áudio ao vivo

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
        self.analysis_method = analysis_method
        self.analysis_order = analysis_order

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

            # 3) Análise de frequências via método selecionado
            analyzer = AnalyzerFactory.create(self.analysis_method, self.sr, CUTOFF_FREQ, order=self.analysis_order)
            freqs_an, mags_an = analyzer.analyze(filtered)
            self.fft_ready.emit(freqs_an, mags_an)

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
