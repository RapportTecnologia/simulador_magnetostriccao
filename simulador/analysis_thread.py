# analysis_thread.py - Thread para análise de arquivos de áudio
import time

import numpy as np
import scipy.signal
import librosa

from PyQt5.QtCore import QThread, pyqtSignal

class FileAnalysisThread(QThread):
    """
    Thread dedicado para processar arquivos de áudio de teste sem bloquear a GUI.
    Emite sinais com dados já processados: espectrograma, Mel bands, FFT e classificação.
    """
    # Sinal para progresso de arquivo: índice_atual, total_de_arquivos
    chunk_ready = pyqtSignal(int, int)

    # Sinal para espectrograma pronto: frequências, matriz em dB
    spec_ready = pyqtSignal(np.ndarray, np.ndarray)

    # Sinal para Mel bands pronto: vetor de potência média em dB
    mel_ready = pyqtSignal(np.ndarray)

    # Sinal para FFT pronta: frequências, magnitudes
    fft_ready = pyqtSignal(np.ndarray, np.ndarray)

    # Sinal para classe prevista: código_da_classe
    class_ready = pyqtSignal(int)

    def __init__(
        self,
        file_list,
        b,
        a,
        sr,
        duration
    ):
        super().__init__()

        # Lista de caminhos para arquivos de teste
        self.files = file_list

        # Coeficientes do filtro passa-baixo (Butterworth)
        self.b = b
        self.a = a

        # Taxa de amostragem em Hertz
        self.sr = sr

        # Duração máxima de cada arquivo em segundos
        self.duration = duration

        # Flag para controle de execução
        self._running = True

    def run(self):
        """
        Método principal executado quando a thread é iniciada.
        Processa cada arquivo em sequência e emite sinais.
        """
        # Número total de arquivos a processar
        total_files = len(self.files)

        # Itera sobre cada arquivo
        for index, file_path in enumerate(self.files, start=1):
            # Verifica se deve interromper
            if not self._running:
                break

            # Emite sinal de progresso
            self.chunk_ready.emit(index, total_files)

            # Carrega o áudio (limitado a DURATION segundos)
            audio, _ = librosa.load(
                file_path,
                sr=self.sr,
                duration=self.duration
            )

            # Aplica filtro passa-baixo Butterworth
            filtered = scipy.signal.filtfilt(
                self.b,
                self.a,
                audio
            )

            # ------------------------------------------------
            # 1) Espectrograma
            # ------------------------------------------------
            # Calcula STFT
            S = librosa.stft(filtered)

            # Frequências correspondentes
            freqs = librosa.fft_frequencies(sr=self.sr)

            # Máscara para frequências até CUTOFF_FREQ
            mask = freqs <= CUTOFF_FREQ

            # Converte amplitudes para dB
            S_db = librosa.amplitude_to_db(np.abs(S[mask, :]))

            # Emite sinal com espectrograma pronto
            self.spec_ready.emit(freqs[mask], S_db)

            # ------------------------------------------------
            # 2) Mel Spectrogram
            # ------------------------------------------------
            # Calcula mel spectrogram
            M = librosa.feature.melspectrogram(
                y=filtered,
                sr=self.sr,
                fmax=CUTOFF_FREQ
            )

            # Média em dB por banda
            mel_db = librosa.power_to_db(M).mean(axis=1)

            # Emite sinal com Mel bands pronto
            self.mel_ready.emit(mel_db)

            # ------------------------------------------------
            # 3) FFT
            # ------------------------------------------------
            # Calcula FFT
            F = np.abs(np.fft.rfft(filtered))

            # Frequências da FFT
            freqs_fft = np.fft.rfftfreq(len(filtered), 1 / self.sr)

            # Máscara para frequências até CUTOFF_FREQ
            mask_fft = freqs_fft <= CUTOFF_FREQ
            
            # Emite sinal com FFT pronta
            self.fft_ready.emit(freqs_fft[mask_fft], F[mask_fft])
            
            # ------------------------------------------------
            # 4) Classificação via modelo externo
            # ------------------------------------------------
            # Extrai MFCCs
            mfccs = librosa.feature.mfcc(
                y=filtered,
                sr=self.sr,
                n_mfcc=N_MFCC,
                fmax=CUTOFF_FREQ
            )

            # Ajusta dimensão temporal para o modelo
            max_time = self.model_input_frames
            if mfccs.shape[1] >= max_time:
                mfccs = mfccs[:, :max_time]
            else:
                padding = np.zeros((N_MFCC, max_time - mfccs.shape[1]))
                mfccs = np.hstack((mfccs, padding))

            # Formato para previsão: (1, n_mfcc, T, 1)
            x = mfccs[np.newaxis, ..., np.newaxis]

            # Realiza predição
            pred = self.MODEL.predict(x)
            cls = int(np.argmax(pred))

            # Emite sinal com classe prevista
            self.class_ready.emit(cls)

            # Pausa breve para simular streaming
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
