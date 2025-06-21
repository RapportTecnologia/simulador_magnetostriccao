from PyQt5.QtCore import QThread, pyqtSignal
import os
import time
import librosa
import scipy.signal
import numpy as np


class ClassificationThread(QThread):
    progress = pyqtSignal(int, int)
    spec_ready = pyqtSignal(np.ndarray, np.ndarray)
    mel_ready = pyqtSignal(np.ndarray)
    fft_ready = pyqtSignal(np.ndarray, np.ndarray)
    class_ready = pyqtSignal(int)
    finished = pyqtSignal()
    processing_metrics = pyqtSignal(float, float, int)

    def __init__(self, test_files, model, b, a, sr, duration, n_mfcc, fmax):
        super().__init__()
        self.test_files = test_files
        self.model = model
        self.b = b
        self.a = a
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.fmax = fmax
        self._running = True

    def run(self):
        total_files = len(self.test_files)
        report_path = os.path.join('relatorio_classificacao.csv')

        # Cria arquivo CSV se não existir
        if not os.path.exists(report_path):
            with open(report_path, 'w', newline='') as f:
                importer = __import__('csv')
                writer = importer.writer(f)
                writer.writerow([
                    'arquivo',
                    'classe_predita',
                    'probabilidade',
                    'data_hora'
                ])

        for index, file_path in enumerate(self.test_files, start=1):
            start_time = time.time()
            if not self._running:
                break

            # Carrega e filtra
            y_test, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
            y_test = scipy.signal.filtfilt(self.b, self.a, y_test)
            # Emit spectrogram data
            S = librosa.stft(y_test)
            freqs = librosa.fft_frequencies(sr=self.sr)
            mask = freqs <= self.fmax
            S_db = librosa.amplitude_to_db(np.abs(S[mask, :]))
            self.spec_ready.emit(freqs[mask], S_db)

            # Emit Mel bands data
            M = librosa.feature.melspectrogram(y=y_test, sr=self.sr, fmax=self.fmax)
            mel_db = librosa.power_to_db(M).mean(axis=1)
            self.mel_ready.emit(mel_db)

            # Emit FFT data
            F = np.abs(np.fft.rfft(y_test))
            freqs_fft = np.fft.rfftfreq(len(y_test), 1 / self.sr)
            mask_fft = freqs_fft <= self.fmax
            self.fft_ready.emit(freqs_fft[mask_fft], F[mask_fft])

            # Extrai MFCC para classificação
            mfcc_test = librosa.feature.mfcc(y=y_test, sr=self.sr, n_mfcc=self.n_mfcc, fmax=self.fmax)
            T = self.model.input_shape[2]
            if mfcc_test.shape[1] >= T:
                mfcc_test = mfcc_test[:, :T]
            else:
                pad = np.zeros((40, T - mfcc_test.shape[1]))
                mfcc_test = np.hstack((mfcc_test, pad))

            # Prediz classe
            x_input = mfcc_test[np.newaxis, ..., np.newaxis]
            pred = self.model.predict(x_input)
            predicted_class = int(np.argmax(pred))
            # Emit classification result
            self.class_ready.emit(predicted_class)
            probability = float(np.max(pred))

            # Timestamp
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            # Escreve no CSV
            with open(report_path, 'a', newline='') as f:
                importer = __import__('csv')
                writer = importer.writer(f)
                writer.writerow([
                    os.path.basename(file_path),
                    predicted_class,
                    probability,
                    timestamp
                ])

            # Emite sinal de progresso
            file_duration = len(y_test) / self.sr
            file_time = time.time() - start_time
            file_size = os.path.getsize(file_path)
            self.processing_metrics.emit(file_time, file_duration, file_size)
            self.progress.emit(index, total_files)

        # Emite sinal de finalização
        self.finished.emit()

    def stop(self):
        self._running = False
