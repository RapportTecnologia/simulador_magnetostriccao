from PyQt5.QtCore import QThread, pyqtSignal
import os
import time
import librosa
import scipy.signal
import numpy as np


"""Thread de classificação de arquivos de áudio."""


class ClassificationThread(QThread):
    """Executa classificação de forma assíncrona.

    A thread percorre todos os arquivos recebidos, realiza o pré-processamento
    (filtro, extração de espectrograma/MFCC e FFT) e, em seguida, envia os
    resultados de inferência para a GUI através de sinais.  As métricas de
    tempo de carregamento, pré-processamento e inferência são emitidas para
    monitoramento de desempenho.
    """

    progress = pyqtSignal(int, int)
    spec_ready = pyqtSignal(np.ndarray, np.ndarray)
    mel_ready = pyqtSignal(np.ndarray)
    fft_ready = pyqtSignal(np.ndarray, np.ndarray)
    class_ready = pyqtSignal(int)
    finished = pyqtSignal()
    processing_metrics = pyqtSignal(float, float, int)
    # Sinal para tempos de I/O, pré-processamento e inferência (segundos)
    detailed_metrics = pyqtSignal(float, float, float)

    def __init__(self, test_files, model, b, a, sr, duration, n_mfcc, fmax):
        """Configura a thread de classificação.

        Parâmetros
        ----------
        test_files : list[str]
            Lista de caminhos para os arquivos de teste.
        model : keras.Model
            Modelo já carregado utilizado na inferência.
        b, a : ndarray
            Coeficientes do filtro Butterworth utilizado para pré-processar o
            áudio.
        sr : int
            Taxa de amostragem dos arquivos.
        duration : int or float
            Duração máxima lida de cada arquivo (segundos).
        n_mfcc : int
            Número de coeficientes MFCC extraídos.
        fmax : int
            Frequência máxima considerada na análise.
        """

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
        """Processa cada arquivo de teste e envia resultados via sinais."""

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
            # Inicia medição de I/O
            load_start = start_time
            if not self._running:
                break

            # Carrega e filtra
            y_test, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
            load_time = time.time() - load_start
            # Inicia medição de pré-processamento
            proc_start = time.time()
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
            # Finaliza pré-processamento
            preproc_time = time.time() - proc_start

            # Prediz classe
            x_input = mfcc_test[np.newaxis, ..., np.newaxis]
            # Mede tempo de inferência
            inf_start = time.time()
            pred = self.model.predict(x_input)
            inference_time = time.time() - inf_start
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
            # Emite métricas detalhadas
            self.detailed_metrics.emit(load_time, preproc_time, inference_time)
            self.processing_metrics.emit(file_time, file_duration, file_size)
            self.progress.emit(index, total_files)

        # Emite sinal de finalização
        self.finished.emit()

    def stop(self):
        """Solicita o encerramento da thread."""

        self._running = False
