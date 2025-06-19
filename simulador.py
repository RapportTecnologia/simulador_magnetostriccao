"""
Analisador de Magnetostricção - Aplicação Python

Este script implementa uma aplicação gráfica GUI com PyQt5 para:

1. Capturar áudio em tempo real de um microfone selecionado.
2. Reproduzir simultaneamente esse áudio na saída de áudio escolhida.
3. Aplicar filtro passa-baixo de 4ª ordem (Butterworth) com corte em 1 kHz.
4. Visualizar em tempo real:
   a. Espectrograma limitado a 1 kHz.
   b. Mel Bands com frequência máxima de 1 kHz.
   c. FFT até 1 kHz.
5. Classificar o sinal em três categorias: Normal, Intermediário e Falha,
   usando uma CNN treinada em MFCCs.
6. Treinar a CNN e salvar o modelo em arquivo .h5 com sufixo identificando o tipo.
7. Iniciar e parar a classificação de arquivos de teste com um único botão.
8. Mostrar na barra de status cada passo do treinamento e o arquivo sendo processado.

Requisitos:
    - Python 3.8+
    - Bibliotecas: numpy, scipy, librosa, sounddevice,
      PyQt5, matplotlib, tensorflow

Uso:
    python simulador.py <diretório_raiz> [--model model.h5]

    <diretório_raiz> deve conter:
      - pasta 'train' com subpastas '0','1','2'.
      - opcionalmente pasta 'test'; se faltar, usa todos os arquivos em qualquer subpasta.
"""

import sys
import os
import argparse
import csv
import datetime
import signal
import numpy as np
import scipy.signal
import librosa
import sounddevice
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QProgressBar,
    QSlider,
    QLabel,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QListWidget,
    QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)
from tensorflow.keras.utils import to_categorical

# ------------------------------------------------
# Parâmetros de processamento
# ------------------------------------------------
SR = 44100  # Taxa de amostragem (Hz)
CUTOFF_FREQ = 1000  # Frequência de corte (Hz)
DURATION = 3  # Duração para MFCC (s)
N_MFCC = 40  # Número de MFCCs
EPOCHS = 10  # Épocas de treino
BATCH_SIZE = 16  # Batch size

# ------------------------------------------------
# Configuração de GPU
# ------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("[INFO] GPU não encontrada. Usando CPU.")

# ------------------------------------------------
# VU Widget: mostra RMS
# ------------------------------------------------
class VUWidget(QProgressBar):
    """
    Barra vertical exibindo RMS do sinal.
    RMS = sqrt((1/N)*sum(x[i]^2)).
    """
    def __init__(self):
        super().__init__()
        self.setOrientation(Qt.Vertical)
        self.setRange(0, 1000)

    def update_level(self, rms_value: float):
        display_value = int(rms_value * 1000)
        self.setValue(display_value)

# ------------------------------------------------
# Color Box: mostra classe
# ------------------------------------------------
class ColorBox(QLabel):
    """
    Label colorido conforme classe:
      0: Normal (verde)
      1: Intermediário (amarelo)
      2: Falha (vermelho)
    """
    COLORS = {0: '#4caf50', 1: '#ffeb3b', 2: '#f44336'}
    LABELS = {0: 'Normal', 1: 'Intermediário', 2: 'Falha'}

    def __init__(self):
        super().__init__('Status: -', alignment=Qt.AlignCenter)
        self.setFixedHeight(40)
        self.setStyleSheet('border: 1px solid black;')

    def update_status(self, cls: int):
        color = self.COLORS.get(cls, '#ffffff')
        text = self.LABELS.get(cls, '-')
        self.setText(f'Status: {text}')
        self.setStyleSheet(f'background-color: {color}; border: 1px solid black;')

# ------------------------------------------------
# Aplicação principal
# ------------------------------------------------
class AnalyzerApp(QMainWindow):
    """
    GUI para captura, visualização, treino e classificação.
    """
    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        model_path: str
    ):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_path = model_path
        self.model = None
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        nyquist = SR / 2.0
        self.b, self.a = scipy.signal.butter(4, CUTOFF_FREQ/nyquist, btype='low')
        self.audio_buffer = []
        self.classifying = False
        self.stop_classify = False
        self._init_ui()
        self._init_stream()
        self._init_timer()
        signal.signal(signal.SIGINT, self._exit)

    def _on_model_changed(self, idx):
        """Desabilita botões ao trocar modelo."""
        self.btn_analysis.setEnabled(False)
        self.btn_classify.setEnabled(False)
        self.model = None
        self.model_path = ''

    def _init_ui(self):
        """Monta UI com PyQt5 widgets e layouts."""
        self.setWindowTitle('Analisador de Magnetostricção')
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Controles à esquerda
        ctrl = QVBoxLayout()
        ctrl.addWidget(QLabel('Entrada (Mic):'))
        self.cmb_input = QComboBox()
        for idx, dev in enumerate(sounddevice.query_devices()):
            if dev['max_input_channels'] > 0:
                self.cmb_input.addItem(f"{idx}: {dev['name']}", idx)
        ctrl.addWidget(self.cmb_input)

        ctrl.addWidget(QLabel('Saída (Speaker):'))
        self.cmb_output = QComboBox()
        for idx, dev in enumerate(sounddevice.query_devices()):
            if dev['max_output_channels'] > 0:
                self.cmb_output.addItem(f"{idx}: {dev['name']}", idx)
        default_out = sounddevice.default.device[1]
        pos = self.cmb_output.findData(default_out)
        if pos >= 0:
            self.cmb_output.setCurrentIndex(pos)
        ctrl.addWidget(self.cmb_output)

        ctrl.addWidget(QLabel('VU-meter:'))
        self.vu = VUWidget()
        ctrl.addWidget(self.vu)

        ctrl.addWidget(QLabel('Ganho Entrada (%)'))
        self.sld_in = QSlider(Qt.Horizontal)
        self.sld_in.setRange(0, 200)
        self.sld_in.setValue(100)
        ctrl.addWidget(self.sld_in)

        ctrl.addWidget(QLabel('Ganho Saída (%)'))
        self.sld_out = QSlider(Qt.Horizontal)
        self.sld_out.setRange(0, 200)
        self.sld_out.setValue(100)
        ctrl.addWidget(self.sld_out)

        # Botões
        self.btn_analysis = QPushButton('Iniciar Análise')
        self.btn_train = QPushButton('Treinar Modelo')
        self.btn_classify = QPushButton('Iniciar Classificação')
        ctrl.addWidget(self.btn_analysis)
        ctrl.addWidget(self.btn_train)
        ctrl.addWidget(self.btn_classify)

        # Matriz ilustrativa
        ctrl.addWidget(QLabel('Matriz Treino:'))
        self.tbl = QTableWidget(3, 3)
        for r in range(3):
            for c in range(3):
                itm = QTableWidgetItem('•')
                itm.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(r, c, itm)
        ctrl.addWidget(self.tbl)

        # Lista de modelos
        ctrl.addWidget(QLabel('Modelos freq:'))
        self.lst = QListWidget()
        for m in ['CNN', 'RNN', 'SVM', 'RandomForest', 'XGBoost']:
            self.lst.addItem(m)
        self.lst.setCurrentRow(0)
        self.lst.currentRowChanged.connect(self._on_model_changed)
        ctrl.addWidget(self.lst)
        ctrl.addStretch()

        # Display à direita
        disp = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        disp.addWidget(self.progress_bar)

        disp.addWidget(QLabel('Espectrograma (até 1kHz)'))
        self.fig_spec = Figure()
        self.canv_spec = FigureCanvas(self.fig_spec)
        disp.addWidget(self.canv_spec)

        disp.addWidget(QLabel('Mel Bands (fmax=1kHz)'))
        self.fig_mel = Figure()
        self.canv_mel = FigureCanvas(self.fig_mel)
        disp.addWidget(self.canv_mel)

        disp.addWidget(QLabel('FFT (até 1kHz)'))
        self.fig_fft = Figure()
        self.canv_fft = FigureCanvas(self.fig_fft)
        disp.addWidget(self.canv_fft)

        self.color_box = ColorBox()
        disp.addWidget(self.color_box)

        self.lbl_file = QLabel('Arquivo: -', alignment=Qt.AlignCenter)
        disp.addWidget(self.lbl_file)

        self.lbl_model = QLabel('Modelo: CNN', alignment=Qt.AlignCenter)
        disp.addWidget(self.lbl_model)

        main_layout.addLayout(ctrl)
        main_layout.addLayout(disp)

        self.statusBar().showMessage('Pronto')

        # Conexões
        self.btn_analysis.clicked.connect(self._toggle_analysis)
        self.btn_train.clicked.connect(self._train_model)
        self.btn_classify.clicked.connect(self._toggle_classification)

        if not self.model:
            self.btn_analysis.setEnabled(False)
            self.btn_classify.setEnabled(False)

    def _init_stream(self):
        """Configura stream full-duplex para captura/reprodução."""
        in_dev = self.cmb_input.currentData()
        out_dev = self.cmb_output.currentData()
        self.stream = sounddevice.Stream(
            device=(in_dev, out_dev),
            samplerate=SR,
            channels=(1, 1),
            dtype='float32',
            callback=self._audio_callback
        )

    def _toggle_analysis(self):
        """Inicia ou pára análise ao vivo; exige modelo treinado."""
        if not self.model:
            QMessageBox.warning(self, 'Aviso', 'Treine um modelo antes.')
            return
        if not self.stream.active:
            self.stream.start()
            self.btn_analysis.setText('Parar Análise')
            self.statusBar().showMessage('Análise ao vivo iniciada')
            self.lbl_file.setText('Arquivo: Áudio ao vivo')
        else:
            self.stream.stop()
            self.btn_analysis.setText('Iniciar Análise')
            self.statusBar().showMessage('Análise ao vivo parada')

    def _audio_callback(self, indata, outdata, frames, time, status):
        """Callback de áudio: filtro, VU, replay e buffer."""
        try:
            data_in = indata[:, 0] * (self.sld_in.value() / 100)
        except RuntimeError:
            return
        filtered = scipy.signal.filtfilt(self.b, self.a, data_in)
        rms = np.sqrt(np.mean(filtered**2))
        self.vu.update_level(rms)
        outdata[:, 0] = filtered * (self.sld_out.value() / 100)
        self.audio_buffer.append(filtered)

    def _init_timer(self):
        """Timer para atualização dos gráficos cada 200ms."""
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._process)
        self.timer.start()

    def _process(self):
        """Processa buffer: espectrograma, Mel, FFT e classe."""
        if not self.audio_buffer or not self.model:
            return
        signal_block = np.concatenate(self.audio_buffer)
        self.audio_buffer.clear()

        # Espectrograma
        self.fig_spec.clear()
        ax1 = self.fig_spec.add_subplot(111)
        S = librosa.stft(signal_block)
        freqs = librosa.fft_frequencies(sr=SR)
        mask = freqs <= CUTOFF_FREQ
        ax1.imshow(librosa.amplitude_to_db(np.abs(S[mask, :])), origin='lower', aspect='auto')
        self.canv_spec.draw()

        # Mel Bands
        self.fig_mel.clear()
        ax2 = self.fig_mel.add_subplot(111)
        M = librosa.feature.melspectrogram(y=signal_block, sr=SR, fmax=CUTOFF_FREQ)
        ax2.plot(librosa.power_to_db(M).mean(axis=1))
        self.canv_mel.draw()

        # FFT
        self.fig_fft.clear()
        ax3 = self.fig_fft.add_subplot(111)
        F = np.abs(np.fft.rfft(signal_block))
        freqs_fft = np.fft.rfftfreq(len(signal_block), 1/SR)
        idx = freqs_fft <= CUTOFF_FREQ
        ax3.plot(freqs_fft[idx], F[idx])
        self.canv_fft.draw()

        # Atualiza rótulo do modelo
        self.lbl_model.setText(f"Modelo: {self.lst.currentItem().text()}")

        # Classificação via MFCC
        mfccs = librosa.feature.mfcc(y=signal_block, sr=SR, n_mfcc=N_MFCC, fmax=CUTOFF_FREQ)
        T = self.model.input_shape[2]
        if mfccs.shape[1] >= T:
            mfccs = mfccs[:, :T]
        else:
            pad = np.zeros((N_MFCC, T - mfccs.shape[1]))
            mfccs = np.hstack((mfccs, pad))
        x = mfccs[np.newaxis, ..., np.newaxis]
        pred = self.model.predict(x)
        cls = int(np.argmax(pred))
        self.color_box.update_status(cls)

    def _collect_data(self, directory: str):
        """
        Carrega áudios de diretório e extrai MFCC para treino.
        Retorna tupla (X, y) prontos para model.fit().
        """
        X = []
        labels = []
        T = None
        files = []
        for root, _, filenames in os.walk(directory):
            for fname in filenames:
                if fname.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                    files.append(os.path.join(root, fname))
        # Barra de progresso para carga dos arquivos
        total = len(files)
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(0)
        for i, full in enumerate(files, 1):
            fname = os.path.basename(full)
            self.statusBar().showMessage(f"Treinamento: carregando {fname} ({i}/{total})")
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(i)
            QApplication.processEvents()
            y, _ = librosa.load(full, sr=SR, duration=DURATION)
            y = scipy.signal.filtfilt(self.b, self.a, y)
            mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, fmax=CUTOFF_FREQ)
            if T is None:
                T = mf.shape[1]
            if mf.shape[1] >= T:
                mf = mf[:, :T]
            else:
                pad = np.zeros((N_MFCC, T - mf.shape[1]))
                mf = np.hstack((mf, pad))
            X.append(mf)
            label = os.path.basename(os.path.dirname(full))
            labels.append(int(label) if label.isdigit() else 0)
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(False)
        X = np.array(X, dtype=np.float32)[..., np.newaxis]
        y = to_categorical(labels, num_classes=3).astype(np.float32)
        self.statusBar().showMessage('Treinamento: dados prontos')
        return X, y

    def _train_model(self):
        """
        Treina CNN e salva em model_<modelo>.h5.
        """
        model_name = self.lst.currentItem().text()
        # Reset barra de progresso
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(0)  # modo indeterminado
            self.progress_bar.setValue(0)
        X, y = self._collect_data(self.train_dir)
        # Verificação de formato antes de criar o modelo
        if len(X.shape) != 4:
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(False)
            QMessageBox.critical(self, 'Erro de dados', f'Os dados de entrada para a CNN devem ter 4 dimensões (amostras, n_mfcc, T, 1), mas a forma recebida foi {X.shape}. Corrija o pré-processamento.')
            return
        if X.shape[-1] != 1:
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(False)
            QMessageBox.critical(self, 'Erro de dados', f'O canal final dos dados de entrada deve ser 1 (shape: {X.shape}). Corrija o pré-processamento.')
            return
        self.statusBar().showMessage('Treinamento: montando arquitetura')
        QApplication.processEvents()
        inp = Input(shape=(X.shape[1], X.shape[2], 1))
        x = Conv2D(16, (3, 3), activation='relu')(inp)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(3, activation='softmax')(x)
        model = Model(inp, out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.statusBar().showMessage('Treinamento: ajustando pesos')
        QApplication.processEvents()
        # Atualiza barra de progresso para o número de épocas
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setMaximum(EPOCHS)
            self.progress_bar.setValue(0)
        # Treinamento com callback para atualizar barra
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
            def on_epoch_end(self, epoch, logs=None):
                if self.progress_bar:
                    self.progress_bar.setValue(epoch + 1)
                    QApplication.processEvents()
        callbacks = []
        if hasattr(self, 'progress_bar'):
            callbacks = [ProgressCallback(self.progress_bar)]
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(False)
        # Corrigido: sempre gerar model_<nome>.h5
        new_path = f"model_{model_name}.h5"
        self.statusBar().showMessage('Treinamento: salvando modelo')
        QApplication.processEvents()
        model.save(new_path)
        self.model_path = new_path
        self.model = tf.keras.models.load_model(self.model_path)
        self.btn_analysis.setEnabled(True)
        self.btn_classify.setEnabled(True)
        self.statusBar().showMessage('Treinamento concluído')
        QMessageBox.information(self, 'Treinamento', f'Modelo salvo em: {self.model_path}')

    def _toggle_classification(self):
        """
        Alterna iniciar/parar classificação de arquivos de teste.
        """
        if self.classifying:
            self.stop_classify = True
            self.statusBar().showMessage('Interrompendo classificação...')
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(False)
        else:
            if not self.model:
                QMessageBox.warning(self, 'Aviso', 'Treine um modelo antes.')
                return
            self.classifying = True
            self.stop_classify = False
            # Exemplo de barra de progresso para classificação
            test_files = []
            for root, _, filenames in os.walk(self.test_dir):
                for fname in filenames:
                    if fname.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                        test_files.append(os.path.join(root, fname))
            total = len(test_files)
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(True)
                self.progress_bar.setMaximum(total)
                self.progress_bar.setValue(0)
            for i, file in enumerate(test_files, 1):
                if self.stop_classify:
                    break
                self.statusBar().showMessage(f"Classificando: {os.path.basename(file)} ({i}/{total})")
                # Aqui entraria a lógica real de classificação do arquivo
                if hasattr(self, 'progress_bar'):
                    self.progress_bar.setValue(i)
                QApplication.processEvents()
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(False)
            self.classifying = False
            self.statusBar().showMessage('Classificação concluída')

    def _exit(self, signum, frame):
        """
        Trata SIGINT (Ctrl+C) para sair graciosamente.
        """
        print('Saindo graciosamente...')
        self.close()
        QApplication.quit()
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analisador GUI')
    parser.add_argument('root_dir', help='Diretório raiz de dados')
    parser.add_argument('--model', default='model.h5', help='Caminho para modelo .h5')
    args = parser.parse_args()

    train_folder = os.path.join(args.root_dir, 'train')
    test_folder = os.path.join(args.root_dir, 'test')

    app = QApplication(sys.argv)
    window = AnalyzerApp(train_folder, test_folder, args.model)
    window.show()
    sys.exit(app.exec_())
