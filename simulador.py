"""
Analisador de Magnetostricção - Aplicação Python

Este script implementa uma aplicação gráfica para:

1. Capturar áudio em tempo real de um microfone selecionado.
2. Reproduzir simultaneamente o áudio capturado na saída de áudio selecionada.
3. Filtrar o sinal até 1 kHz usando um filtro Butterworth de 4ª ordem.
4. Exibir em tempo real:
   - Espectrograma limitado a 1 kHz.
   - Mel Bands com frequência máxima de 1 kHz.
   - FFT até 1 kHz.
5. Classificar o sinal em três categorias (Normal, Intermediário, Falha) usando uma CNN treinada com MFCCs.
6. Treinar o modelo e salvá‑lo em arquivo `.h5` nomeado conforme a arquitetura usada.
7. Classificar arquivos de teste, com opção de interrupção a qualquer momento.

Requisitos:
    Python 3.8+
    numpy, scipy, librosa, sounddevice, PyQt5, matplotlib, tensorflow

Uso:
    python simulador.py <diretório_raiz> [--model model.h5]
    onde <diretório_raiz> contém pastas:
        train/0, train/1, train/2
        test/0, test/1, test/2
"""

import sys
import os
import argparse
import signal
import numpy as np
import scipy.signal as sps
import librosa
import sounddevice as sd
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
SR = 44100             # Taxa de amostragem (Hz)
CUTOFF_FREQ = 1000     # Frequência de corte do filtro lowpass (Hz)
DURATION = 3           # Duração (s) para MFCC de arquivos de teste
N_MFCC = 40            # Número de coeficientes MFCC
EPOCHS = 10            # Número de épocas de treinamento
BATCH_SIZE = 16        # Tamanho do batch de treinamento

# ------------------------------------------------
# Configuração de GPU
# ------------------------------------------------
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("[INFO] GPU não encontrada. Usando CPU.")

# ------------------------------------------------
# Widget para VU-meter (nível RMS)
# ------------------------------------------------
class VUWidget(QProgressBar):
    """
    Barra vertical que exibe o valor RMS do áudio.
    Fórmula RMS:
        RMS = sqrt( (1/N) * sum(x[i]^2) )
    onde x[i] são amostras de áudio.
    """
    def __init__(self):
        super().__init__()
        self.setOrientation(Qt.Vertical)
        self.setRange(0, 1000)

    def update_level(self, rms_value: float):
        # Converte rms 0.0-1.0 para escala 0-1000
        self.setValue(int(rms_value * 1000))

# ------------------------------------------------
# Quadro colorido para status de classificação
# ------------------------------------------------
class ColorBox(QLabel):
    """
    Exibe cor e texto conforme classe:
        0: Normal  (verde)
        1: Interm. (amarelo)
        2: Falha   (vermelho)
    """
    COLORS = {
        0: '#4caf50',
        1: '#ffeb3b',
        2: '#f44336'
    }
    LABELS = {
        0: 'Normal',
        1: 'Intermediário',
        2: 'Falha'
    }

    def __init__(self):
        super().__init__('Status: -', alignment=Qt.AlignCenter)
        self.setFixedHeight(40)
        self.setStyleSheet('border: 1px solid black;')

    def update_status(self, cls: int):
        color = self.COLORS.get(cls, '#ffffff')
        text = self.LABELS.get(cls, '-')
        self.setText(f'Status: {text}')
        self.setStyleSheet(
            f'background-color: {color}; border: 1px solid black;'
        )

# ------------------------------------------------
# Classe principal da aplicação
# ------------------------------------------------
class AnalyzerApp(QMainWindow):
    """
    Janela principal que:
      - Captura e reproduz áudio ao vivo
      - Processa e exibe gráficos (espectro, Mel, FFT)
      - Treina e carrega modelos (.h5)
      - Classifica sinais em tempo real ou arquivos de teste
    """
    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        model_path: str
    ):
        super().__init__()

        # Caminhos e modelo
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_path = model_path
        self.model = None

        # Se existir arquivo .h5, carrega modelo
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)

        # Filtro Butterworth passa-baixo
        nyquist = SR / 2
        self.b, self.a = sps.butter(
            4,
            CUTOFF_FREQ / nyquist,
            btype='low'
        )

        # Buffer de áudio para processamento
        self.audio_buffer = []

        # Flag para interromper classificação de testes
        self.stop_classify = False

        # Monta interface e inicia stream/timer
        self._init_ui()
        self._init_stream()
        self._init_timer()

        # Captura Ctrl-C para saída graciosa
        signal.signal(signal.SIGINT, self._exit)

    def _init_ui(self):
        """
        Cria widgets e layouts:
          - Seletor de dispositivos
          - Controles de ganho
          - Botões de ação
          - Área de gráficos e indicadores
        """
        self.setWindowTitle('Analisador de Magnetostricção')

        # Widget principal
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # === Controles à esquerda ===
        controls = QVBoxLayout()

        # Seletor de entrada (microfone)
        controls.addWidget(QLabel('Entrada (Mic):'))
        self.cmb_input = QComboBox()
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                self.cmb_input.addItem(
                    f"{idx}: {dev['name']}", idx
                )
        controls.addWidget(self.cmb_input)

        # Seletor de saída (alto-falante)
        controls.addWidget(QLabel('Saída (Speaker):'))
        self.cmb_output = QComboBox()
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_output_channels'] > 0:
                self.cmb_output.addItem(
                    f"{idx}: {dev['name']}", idx
                )
        # Define saída padrão
        default_out = sd.default.device[1]
        pos = self.cmb_output.findData(default_out)
        if pos >= 0:
            self.cmb_output.setCurrentIndex(pos)
        controls.addWidget(self.cmb_output)

        # VU-meter
        controls.addWidget(QLabel('VU-meter:'))
        self.vu = VUWidget()
        controls.addWidget(self.vu)

        # Controle de ganho de entrada
        controls.addWidget(QLabel('Ganho Entrada (%)'))
        self.sld_in = QSlider(Qt.Horizontal)
        self.sld_in.setRange(0, 200)
        self.sld_in.setValue(100)
        controls.addWidget(self.sld_in)

        # Controle de ganho de saída
        controls.addWidget(QLabel('Ganho Saída (%)'))
        self.sld_out = QSlider(Qt.Horizontal)
        self.sld_out.setRange(0, 200)
        self.sld_out.setValue(100)
        controls.addWidget(self.sld_out)

        # Botões de ação
        self.btn_toggle = QPushButton('Iniciar')
        self.btn_train = QPushButton('Treinar Modelo')
        self.btn_classify = QPushButton('Classificar Testes')
        self.btn_stop = QPushButton('Parar Classificação')
        controls.addWidget(self.btn_toggle)
        controls.addWidget(self.btn_train)
        controls.addWidget(self.btn_classify)
        controls.addWidget(self.btn_stop)

        # Matriz ilustrativa de treinamento
        controls.addWidget(QLabel('Matriz Treino:'))
        self.tbl = QTableWidget(3, 3)
        for r in range(3):
            for c in range(3):
                item = QTableWidgetItem('•')
                item.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(r, c, item)
        controls.addWidget(self.tbl)

        # Lista de modelos disponíveis
        controls.addWidget(QLabel('Modelos freq:'))
        self.lst = QListWidget()
        for m in ['CNN', 'RNN', 'SVM', 'RandomForest', 'XGBoost']:
            self.lst.addItem(m)
        self.lst.setCurrentRow(0)
        controls.addWidget(self.lst)
        controls.addStretch()

        # === Área de exibição à direita ===
        display = QVBoxLayout()

        # Espectrograma até 1 kHz
        display.addWidget(QLabel('Espectrograma (até 1 kHz)'))
        self.fig_spec = Figure()
        self.canvas_spec = FigureCanvas(self.fig_spec)
        display.addWidget(self.canvas_spec)

        # Mel Bands com fmax=1 kHz
        display.addWidget(QLabel('Mel Bands (fmax = 1 kHz)'))
        self.fig_mel = Figure()
        self.canvas_mel = FigureCanvas(self.fig_mel)
        display.addWidget(self.canvas_mel)

        # FFT até 1 kHz
        display.addWidget(QLabel('FFT (até 1 kHz)'))
        self.fig_fft = Figure()
        self.canvas_fft = FigureCanvas(self.fig_fft)
        display.addWidget(self.canvas_fft)

        # Indicadores finais
        self.status = ColorBox()
        display.addWidget(self.status)

        self.lbl_file = QLabel('Arquivo: -', alignment=Qt.AlignCenter)
        display.addWidget(self.lbl_file)

        self.lbl_model = QLabel('Modelo: CNN', alignment=Qt.AlignCenter)
        display.addWidget(self.lbl_model)

        # Monta layout principal
        main_layout.addLayout(controls)
        main_layout.addLayout(display)

        # Conecta sinais a métodos
        self.btn_toggle.clicked.connect(self._toggle)
        self.btn_train.clicked.connect(self._train)
        self.btn_classify.clicked.connect(self._classify_test_gui)
        self.btn_stop.clicked.connect(self._stop_classify)

        # Desabilita botões se não há modelo carregado
        if self.model is None:
            self.btn_toggle.setEnabled(False)
            self.btn_classify.setEnabled(False)

    def _init_stream(self):
        """
        Inicializa stream full-duplex para captura e reprodução simultâneas.
        """
        in_dev = self.cmb_input.currentData()
        out_dev = self.cmb_output.currentData()
        self.stream = sd.Stream(
            device=(in_dev, out_dev),
            samplerate=SR,
            channels=(1, 1),
            dtype='float32',
            callback=self._audio_callback
        )

    def _toggle(self):
        """
        Inicia ou para o stream. Exibe aviso se não houver modelo treinado.
        """
        if self.model is None:
            QMessageBox.warning(
                self,
                'Aviso',
                'Treine um modelo antes de iniciar análise ao vivo.'
            )
            return
        if not self.stream.active:
            self.stream.start()
            self.btn_toggle.setText('Parar')
            self.lbl_file.setText('Arquivo: Áudio ao vivo')
        else:
            self.stream.stop()
            self.btn_toggle.setText('Iniciar')

    def _stop_classify(self):
        """
        Interrompe a rotina de classificação de arquivos de teste.
        """
        self.stop_classify = True

    def _audio_callback(self, indata, outdata, frames, time, status):
        """
        Callback executado para cada buffer:
        1. Aplica ganho de entrada
        2. Filtra sinal abaixo de 1 kHz
        3. Calcula RMS e atualiza VU-meter
        4. Reproduz sinal na saída com ganho
        5. Armazena buffer para processamento posterior
        """
        try:
            audio = indata[:, 0] * (self.sld_in.value() / 100)
        except RuntimeError:
            return
        audio = sps.filtfilt(self.b, self.a, audio)
        rms = np.sqrt(np.mean(audio ** 2))
        self.vu.update_level(rms)
        outdata[:, 0] = audio * (self.sld_out.value() / 100)
        self.audio_buffer.append(audio)

    def _init_timer(self):
        """
        Inicia timer para atualizar gráficos a cada 200 ms.
        """
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._process)
        self.timer.start()

    def _process(self):
        """
        Processa áudio acumulado:
        1. Calcula e exibe espectrograma até 1 kHz
        2. Calcula e exibe Mel Bands com fmax=1 kHz
        3. Calcula e exibe FFT até 1 kHz
        4. Extrai MFCC e realiza classificação
        """
        if not self.audio_buffer or self.model is None:
            return
        data = np.concatenate(self.audio_buffer)
        self.audio_buffer.clear()

        # Espectrograma
        self.fig_spec.clear()
        ax1 = self.fig_spec.add_subplot(111)
        S = librosa.stft(data)
        freqs = librosa.fft_frequencies(sr=SR)
        mask = freqs <= CUTOFF_FREQ
        ax1.imshow(
            librosa.amplitude_to_db(np.abs(S[mask, :])),
            origin='lower', aspect='auto'
        )
        self.canvas_spec.draw()

        # Mel Bands
        self.fig_mel.clear()
        ax2 = self.fig_mel.add_subplot(111)
        M = librosa.feature.melspectrogram(
            y=data, sr=SR, fmax=CUTOFF_FREQ
        )
        ax2.plot(librosa.power_to_db(M).mean(axis=1))
        self.canvas_mel.draw()

        # FFT
        self.fig_fft.clear()
        ax3 = self.fig_fft.add_subplot(111)
        F = np.abs(np.fft.rfft(data))
        freqs_fft = np.fft.rfftfreq(len(data), 1 / SR)
        idx = freqs_fft <= CUTOFF_FREQ
        ax3.plot(freqs_fft[idx], F[idx])
        self.canvas_fft.draw()

        # Atualiza rótulo de modelo
        self.lbl_model.setText(
            f"Modelo: {self.lst.currentItem().text()}"
        )

        # Classificação por MFCC
        mfcc = librosa.feature.mfcc(
            y=data, sr=SR,
            n_mfcc=N_MFCC, fmax=CUTOFF_FREQ
        )
        T = self.model.input_shape[2]
        if mfcc.shape[1] >= T:
            mfcc = mfcc[:, :T]
        else:
            pad = np.zeros((N_MFCC, T - mfcc.shape[1]))
            mfcc = np.hstack((mfcc, pad))
        x = mfcc[np.newaxis, ..., np.newaxis]
        pred = self.model.predict(x)
        cls = int(np.argmax(pred))
        self.status.update_status(cls)

    def _collect(self, directory: str):
        """
        Carrega arquivos de 'directory' e extrai MFCC:
        - Aplica filtro passa-baixo
        - Extrai MFCC com fmax=1 kHz
        - Pad/trunc para mesmo comprimento T
        Retorna X (float32) e y (one-hot float32)
        """
        X = []
        labels = []
        T = None
        for label in ['0', '1', '2']:
            path = os.path.join(directory, label)
            for fname in os.listdir(path):
                y_audio, _ = librosa.load(
                    os.path.join(path, fname),
                    sr=SR, duration=DURATION
                )
                y_audio = sps.filtfilt(self.b, self.a, y_audio)
                mf = librosa.feature.mfcc(
                    y=y_audio, sr=SR,
                    n_mfcc=N_MFCC, fmax=CUTOFF_FREQ
                )
                if T is None:
                    T = mf.shape[1]
                if mf.shape[1] >= T:
                    mf = mf[:, :T]
                else:
                    pad = np.zeros((N_MFCC, T - mf.shape[1]))
                    mf = np.hstack((mf, pad))
                X.append(mf)
                labels.append(int(label))
        X = np.array(X, dtype=np.float32)[..., np.newaxis]
        y = to_categorical(labels, num_classes=3).astype(np.float32)
        return X, y

    def _train(self):
        """
        Treina CNN e salva em <base>_<modelo>.h5.
        Habilita análise após treinamento.
        """
        model_name = self.lst.currentItem().text()
        X, y = self._collect(self.train_dir)
        inp = Input(shape=X.shape[1:])
        x = Conv2D(16, (3, 3), activation='relu')(inp)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(3, activation='softmax')(x)
        model = Model(inp, out)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        base, ext = os.path.splitext(self.model_path)
        new_path = f"{base}_{model_name}{ext}"
        model.save(new_path)
        self.model_path = new_path
        self.model = tf.keras.models.load_model(self.model_path)
        self.btn_toggle.setEnabled(True)
        self.btn_classify.setEnabled(True)
        QMessageBox.information(
            self,
            'Treinamento',
            f'Modelo salvo em: {self.model_path}'
        )

    def _classify_test_gui(self):
        """
        Classifica arquivos de teste com atualização de gráficos.
        Permite interrupção por botão.
        """
        if self.model is None:
            QMessageBox.warning(
                self,
                'Aviso',
                'Treine um modelo antes de classificar testes.'
            )
            return
        self.stop_classify = False
        for label in ['0', '1', '2']:
            folder = os.path.join(self.test_dir, label)
            for fname in os.listdir(folder):
                if self.stop_classify:
                    QMessageBox.information(
                        self,
                        'Classificação',
                        'Classificação interrompida.'
                    )
                    return
                self.lbl_file.setText(f'Arquivo: {fname}')
                QApplication.processEvents()
                full = os.path.join(folder, fname)
                y_audio, _ = librosa.load(
                    full, sr=SR, duration=DURATION
                )
                y_audio = sps.filtfilt(self.b, self.a, y_audio)
                self.audio_buffer = [y_audio]
                self._process()
                mf = librosa.feature.mfcc(
                    y=y_audio, sr=SR,
                    n_mfcc=N_MFCC, fmax=CUTOFF_FREQ
                )
                T = self.model.input_shape[2]
                if mf.shape[1] >= T:
                    mf = mf[:, :T]
                else:
                    pad = np.zeros((N_MFCC, T - mf.shape[1]))
                    mf = np.hstack((mf, pad))
                x = mf[np.newaxis, ..., np.newaxis]
                pred = self.model.predict(x)
                cls = int(np.argmax(pred))
                self.status.update_status(cls)
        QMessageBox.information(
            self,
            'Classificação',
            'Classificação de testes concluída.'
        )

    def _exit(self, *args):
        """
        Interrompe streams e encerra aplicação.
        """
        if hasattr(self, 'stream') and self.stream.active:
            self.stream.stop()
            self.stream.close()
        QApplication.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analisador de Magnetostricção'
    )
    parser.add_argument(
        'root_dir',
        help='Diretório com subpastas train/ e test/'
    )
    parser.add_argument(
        '--model', default='model.h5',
        help='Caminho para arquivo .h5'
    )
    args = parser.parse_args()
    train_dir = os.path.join(args.root_dir, 'train')
    test_dir = os.path.join(args.root_dir, 'test')
    app = QApplication(sys.argv)
    window = AnalyzerApp(train_dir, test_dir, args.model)
    window.show()
    sys.exit(app.exec_())
