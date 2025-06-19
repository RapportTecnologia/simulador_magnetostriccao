# simulador.py - Aplicação GUI para análise de magnetostricção
import sys
import os
import argparse
import signal
import time

import numpy as np
import scipy.signal
import librosa
import sounddevice
import tensorflow as tf

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
    QListWidget,
    QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Importa módulo de thread de análise com sinais estendidos
from .analysis_thread import FileAnalysisThread
# Importa widgets personalizados de UI
from .ui_widgets import VUWidget, ColorBox
# Importa thread de classificação
from .classification_thread import ClassificationThread

# ------------------------------------------------
# Parâmetros globais da aplicação
# ------------------------------------------------
# Taxa de amostragem em Hz
SR = 44100

# Frequência de corte para filtro passa-baixo em Hz
CUTOFF_FREQ = 1000

# Duração máxima (em segundos) para carregar cada arquivo de áudio
DURATION = 3

# Número de coeficientes MFCC a extrair
N_MFCC = 40

# Épocas de treinamento para a CNN
EPOCHS = 10

# Tamanho do batch para o treinamento
BATCH_SIZE = 16

class AnalyzerApp(QMainWindow):
    """
    GUI principal para captura, visualização,
    treinamento e classificação de sinais de magnetostricção.
    """
    def __init__(
        self,
        train_dir,
        test_dir,
        model_path
    ):
        super().__init__()

        # Diretórios de treino e teste
        self.train_dir = train_dir
        self.test_dir = test_dir

        # Caminho para o modelo .h5
        self.model_path = model_path

        # Variável para armazenar o modelo carregado/treinado
        self.model = None

        # Thread de análise de arquivos de teste
        self.thread = None

        # Thread de classificação
        self.classification_thread = None

        # Buffer para armazenar pedaços de áudio capturados
        self.audio_buffer = []

        # Se o modelo já existir no disco, carrega-o
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)

        # Configura coeficientes do filtro Butterworth de 4ª ordem
        nyquist = SR / 2.0
        self.b, self.a = scipy.signal.butter(
            4,
            CUTOFF_FREQ / nyquist,
            btype='low'
        )

        # Monta a interface gráfica
        self._init_ui()

        # Configura stream de áudio full-duplex
        self._init_stream()

        # Atalho para fechar a aplicação com Ctrl+C
        signal.signal(signal.SIGINT, lambda *args: self.close())

    def _init_ui(self):
        """
        Monta todos os widgets e layouts da interface.
        Conecta sinais e slots.
        Define estado inicial de botões.
        """
        # Define título da janela
        self.setWindowTitle('Analisador de Magnetostricção')

        # Widget principal
        main_widget = QWidget()

        # Layout horizontal principal
        main_layout = QHBoxLayout(main_widget)

        # Define o widget central da janela
        self.setCentralWidget(main_widget)

        # -----------------------------
        # Painel de Controle (esquerda)
        # -----------------------------
        ctrl_layout = QVBoxLayout()

        # Seletor de dispositivo de entrada (microfone)
        ctrl_layout.addWidget(QLabel('Entrada (Mic):'))
        self.cmb_input = QComboBox()
        devices = sounddevice.query_devices()
        for index, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                display = f"{index}: {dev['name']}"
                self.cmb_input.addItem(display, index)
        ctrl_layout.addWidget(self.cmb_input)

        # Seletor de dispositivo de saída (alto-falante)
        ctrl_layout.addWidget(QLabel('Saída (Speaker):'))
        self.cmb_output = QComboBox()
        for index, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                display = f"{index}: {dev['name']}"
                self.cmb_output.addItem(display, index)
        ctrl_layout.addWidget(self.cmb_output)

        # VU-meter para exibir nível RMS em tempo real
        ctrl_layout.addWidget(QLabel('VU-meter:'))
        self.vu = VUWidget()
        ctrl_layout.addWidget(self.vu)

        # Slider para ajuste de ganho de entrada
        ctrl_layout.addWidget(QLabel('Ganho Entrada (%)'))
        self.sld_in = QSlider(Qt.Horizontal)
        self.sld_in.setRange(0, 200)
        self.sld_in.setValue(100)
        ctrl_layout.addWidget(self.sld_in)

        # Slider para ajuste de ganho de saída
        ctrl_layout.addWidget(QLabel('Ganho Saída (%)'))
        self.sld_out = QSlider(Qt.Horizontal)
        self.sld_out.setRange(0, 200)
        self.sld_out.setValue(100)
        ctrl_layout.addWidget(self.sld_out)

        # Botão para iniciar/parar análise de arquivos de teste
        self.btn_analysis = QPushButton('Iniciar Análise')
        ctrl_layout.addWidget(self.btn_analysis)

        # Botão para treinar o modelo
        self.btn_train = QPushButton('Treinar Modelo')
        ctrl_layout.addWidget(self.btn_train)

        # Botão para iniciar classificação de testes
        self.btn_classify = QPushButton('Iniciar Classificação')
        ctrl_layout.addWidget(self.btn_classify)

        # Lista de algoritmos de classificação disponíveis
        ctrl_layout.addWidget(QLabel('Modelos:'))
        self.lst_models = QListWidget()
        for model_name in ['CNN', 'RNN', 'SVM', 'RandomForest', 'XGBoost']:
            self.lst_models.addItem(model_name)
        self.lst_models.setCurrentRow(0)
        ctrl_layout.addWidget(self.lst_models)

        # Habilita os botões de análise/classificação apenas se o modelo estiver carregado
        model_loaded = (self.model is not None)
        self.btn_analysis.setEnabled(model_loaded)
        self.btn_classify.setEnabled(model_loaded)

        # Embala o painel de controle em um QWidget
        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl_layout)
        ctrl_widget.setMaximumWidth(300)

        # Adiciona ao layout principal
        main_layout.addWidget(ctrl_widget)

        # -----------------------------
        # Painel de Visualização (direita)
        # -----------------------------
        disp_layout = QVBoxLayout()

        # Barra de progresso para arquivos
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        disp_layout.addWidget(self.progress_bar)

        # Gráfico de espectrograma
        disp_layout.addWidget(QLabel(f'Espectrograma (até {CUTOFF_FREQ} Hz)'))
        self.fig_spec = Figure()
        self.canvas_spec = FigureCanvas(self.fig_spec)
        disp_layout.addWidget(self.canvas_spec)

        # Gráfico de Mel bands
        disp_layout.addWidget(QLabel(f'Mel Bands (fmax={CUTOFF_FREQ} Hz)'))
        self.fig_mel = Figure()
        self.canvas_mel = FigureCanvas(self.fig_mel)
        disp_layout.addWidget(self.canvas_mel)

        # Gráfico de FFT
        disp_layout.addWidget(QLabel(f'FFT (até {CUTOFF_FREQ} Hz)'))
        self.fig_fft = Figure()
        self.canvas_fft = FigureCanvas(self.fig_fft)
        disp_layout.addWidget(self.canvas_fft)

        # Caixa de cor para status de classificação
        self.color_box = ColorBox()
        disp_layout.addWidget(self.color_box)

        # Rótulo para exibir arquivo atual
        self.lbl_file = QLabel('Arquivo: -', alignment=Qt.AlignCenter)
        disp_layout.addWidget(self.lbl_file)

        # Rótulo para exibir modelo atual
        current_model = self.lst_models.currentItem().text()
        self.lbl_model = QLabel(f'Modelo: {current_model}', alignment=Qt.AlignCenter)
        disp_layout.addWidget(self.lbl_model)

        # Empacota painel de visualização
        disp_widget = QWidget()
        disp_widget.setLayout(disp_layout)

        # Adiciona ao layout principal (expande)
        main_layout.addWidget(disp_widget, stretch=1)

        # -----------------------------
        # Conexões de sinais e slots
        # -----------------------------
        self.btn_analysis.clicked.connect(self._toggle_analysis)
        self.btn_train.clicked.connect(self._train_model)
        self.btn_classify.clicked.connect(self._toggle_classification)
        self.lst_models.currentRowChanged.connect(self._on_model_changed)

    def _init_stream(self):
        """
        Inicializa o stream full-duplex usando sounddevice.
        """
        in_device = self.cmb_input.currentData()
        out_device = self.cmb_output.currentData()

        try:
            self.stream = sounddevice.Stream(
                device=(in_device, out_device),
                samplerate=SR,
                channels=(1, 1),
                dtype='float32',
                callback=self._audio_callback
            )
        except Exception as e:
            print(f'Erro ao inicializar stream de áudio: {e}')

    def _toggle_analysis(self):
        """
        Callback do botão de análise: inicia ou para a FileAnalysisThread.
        """
        # Se já existe e está rodando, para a thread
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.btn_analysis.setText('Iniciar Análise')
            self.statusBar().showMessage('Análise de arquivos parada')
            return

        # Se não há modelo, avisa e retorna
        if self.model is None:
            QMessageBox.warning(self, 'Aviso', 'Treine ou carregue um modelo primeiro.')
            return

        # Lista todos os arquivos de teste válidos
        test_files = []
        for root, _, files in os.walk(self.test_dir):
            for filename in files:
                if filename.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                    test_files.append(os.path.join(root, filename))

        # Se não encontrou nenhum arquivo, avisa e retorna
        if not test_files:
            QMessageBox.warning(self, 'Aviso', 'Nenhum arquivo de teste encontrado.')
            return

        # Prepara barra de progresso
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(test_files))
        self.progress_bar.setValue(0)

        # Atualiza texto do botão e status bar
        self.btn_analysis.setText('Parar Análise')
        self.statusBar().showMessage('Análise de arquivos iniciada')

        # Cria e configura a FileAnalysisThread
        self.thread = FileAnalysisThread(
            test_files,
            self.b,
            self.a,
            SR,
            DURATION
        )

        # Conecta sinais para atualizar UI
        self.thread.chunk_ready.connect(self._update_progress)
        self.thread.spec_ready.connect(self._update_spectrogram)
        self.thread.mel_ready.connect(self._update_mel_bands)
        self.thread.fft_ready.connect(self._update_fft)
        self.thread.class_ready.connect(self.color_box.update_status)
        self.thread.finished.connect(lambda: self.progress_bar.setVisible(False))

        # Inicia thread
        self.thread.start()

    def _update_progress(self, index, total):
        """
        Recebe progresso de arquivo da thread e atualiza UI.
        """
        self.lbl_file.setText(f'Arquivo: {index}/{total}')
        self.statusBar().showMessage(f'Analisando: {index}/{total}')
        self.progress_bar.setValue(index)

    def _update_spectrogram(self, freqs, S_db):
        """
        Atualiza o gráfico de espectrograma com dados pré-processados.
        """
        self.fig_spec.clear()
        ax = self.fig_spec.add_subplot(111)
        ax.imshow(
            S_db,
            origin='lower',
            aspect='auto',
            extent=[0, DURATION, freqs[0], freqs[-1]]
        )
        self.canvas_spec.draw()
        QApplication.processEvents()

    def _update_mel_bands(self, mel_db):
        """
        Atualiza o gráfico de Mel bands com dados pré-processados.
        """
        self.fig_mel.clear()
        ax = self.fig_mel.add_subplot(111)
        ax.plot(mel_db)
        self.canvas_mel.draw()
        QApplication.processEvents()

    def _update_fft(self, freqs, magnitudes):
        """
        Atualiza o gráfico de FFT com dados pré-processados.
        """
        self.fig_fft.clear()
        ax = self.fig_fft.add_subplot(111)
        ax.plot(freqs, magnitudes)
        self.canvas_fft.draw()
        QApplication.processEvents()

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """
        Callback do stream de áudio ao vivo.
        Atualiza VU-meter e buffer não utilizado pela thread.
        """
        # Aplica ganho de entrada
        input_signal = indata[:, 0] * (self.sld_in.value() / 100)
        # Filtra o sinal
        filtered_signal = scipy.signal.filtfilt(
            self.b,
            self.a,
            input_signal
        )
        # Calcula RMS
        rms_value = np.sqrt(np.mean(filtered_signal ** 2))
        # Atualiza VU-meter
        self.vu.update_level(rms_value)
        # Envia para saída com ganho de saída
        outdata[:, 0] = filtered_signal * (self.sld_out.value() / 100)

    def _on_model_changed(self, index):
        """
        Slot chamado quando o usuário muda de modelo na lista.
        Reseta modelo carregado e desabilita botões de análise/classificação.
        """
        self.model = None
        self.btn_analysis.setEnabled(False)
        self.btn_classify.setEnabled(False)
        # Atualiza rótulo de modelo
        selected_model = self.lst_models.item(index).text()
        self.lbl_model.setText(f'Modelo: {selected_model}')

    def _collect_data(self, directory):
        """
        Carrega arquivos de treinamento, extrai MFCC e retorna dados.
        """
        X = []
        y = []
        T = None

        # Lista arquivos
        files = []
        for root, _, filenames in os.walk(directory):
            for fn in filenames:
                if fn.lower().endswith(('.wav','.flac','.mp3','.ogg')):
                    files.append(os.path.join(root, fn))

        # Barra de progresso
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(files))

        for i, file_path in enumerate(files, start=1):
            self.progress_bar.setValue(i)
            QApplication.processEvents()

            # Carrega áudio
            audio, _ = librosa.load(
                file_path,
                sr=SR,
                duration=DURATION
            )
            # Filtra
            audio = scipy.signal.filtfilt(
                self.b,
                self.a,
                audio
            )
            # Extrai MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=SR,
                n_mfcc=N_MFCC,
                fmax=CUTOFF_FREQ
            )
            # Define T (comprimento mínimo)
            if T is None:
                T = mfcc.shape[1]
            # Ajusta comprimento para T
            if mfcc.shape[1] >= T:
                mfcc = mfcc[:, :T]
            else:
                padding = np.zeros((N_MFCC, T - mfcc.shape[1]))
                mfcc = np.hstack((mfcc, padding))

            X.append(mfcc)

            # Obtém label a partir do nome da pasta
            label_str = os.path.basename(os.path.dirname(file_path))
            if label_str.isdigit():
                label = int(label_str)
            else:
                label = 0
            y.append(label)

        self.progress_bar.setVisible(False)

        # Converte em arrays e one-hot encode
        X = np.array(X)[..., np.newaxis]
        y = to_categorical(y, num_classes=3)

        return X, y

    def _train_model(self):
        """
        Treina CNN com dados de treinamento e salva o modelo.
        """
        X_train, y_train = self._collect_data(self.train_dir)

        # Monta arquitetura CNN
        input_layer = Input(
            shape=(X_train.shape[1], X_train.shape[2], 1)
        )
        conv1 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation='relu'
        )(input_layer)
        pool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu'
        )(pool1)
        pool2 = MaxPooling2D((2, 2))(conv2)
        flat = Flatten()(pool2)
        dense = Dense(64, activation='relu')(flat)
        output_layer = Dense(3, activation='softmax')(dense)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Treina modelo
        model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        # Salva modelo treinado
        model_file = f"model_{self.lst_models.currentItem().text()}.h5"
        model.save(model_file)

        # Recarrega modelo para uso imediato
        self.model = tf.keras.models.load_model(model_file)

        # Habilita botões agora que há modelo
        self.btn_analysis.setEnabled(True)
        self.btn_classify.setEnabled(True)

        QMessageBox.information(
            self,
            'Treinamento Concluído',
            f'Modelo salvo em: {model_file}'
        )

    def _toggle_classification(self):
        """
        Classifica todos os arquivos de teste e cria relatório CSV.
        """
        # Alterna estado de classificação
        running = getattr(self, 'classifying', False)
        if running:
            self.classifying = False
            self.btn_classify.setText('Parar Classificação')
            self.btn_classify.setStyleSheet('background-color: red;')
            self.btn_analysis.setEnabled(True)
            self.btn_train.setEnabled(True)
            if self.classification_thread.isRunning():
                self.classification_thread.stop()
                self.classification_thread.wait()
            return
        else:
            self.classifying = True

        # Lista arquivos de teste
        test_files = []
        for root, _, filenames in os.walk(self.test_dir):
            for fn in filenames:
                if fn.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                    test_files.append(os.path.join(root, fn))

        self.current_files = test_files

        # Inicia a thread de classificação
        self.btn_classify.setText('Parar Classificação')
        self.btn_classify.setStyleSheet('background-color: red;')
        self.btn_analysis.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(test_files))
        self.progress_bar.setValue(0)
        self.classification_thread = ClassificationThread(
            test_files,
            self.model,
            self.b,
            self.a,
            SR,
            DURATION,
            N_MFCC,
            CUTOFF_FREQ
        )
        self.classification_thread.progress.connect(self._update_progress)
        self.classification_thread.spec_ready.connect(self._update_spectrogram)
        self.classification_thread.mel_ready.connect(self._update_mel_bands)
        self.classification_thread.fft_ready.connect(self._update_fft)
        self.classification_thread.class_ready.connect(self.color_box.update_status)
        self.classification_thread.finished.connect(self._classification_finished)
        self.classification_thread.start()

    def _classification_finished(self):
        QMessageBox.information(
            self,
            'Classificação',
            'Relatório de classificação gerado com sucesso.'
        )
        self.classifying = False
        self.btn_classify.setText('Iniciar Classificação')
        self.btn_classify.setStyleSheet('background-color: green;')
        self.btn_analysis.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _update_progress(self, index, total):
        """
        Recebe progresso de arquivo da thread e atualiza UI.
        """
        file_name = os.path.basename(self.current_files[index-1])
        self.lbl_file.setText(f'Arquivo: {file_name}')
        self.statusBar().showMessage(f'Classificando: {index}/{total}')
        self.progress_bar.setValue(index)

    def closeEvent(self, event):
        """
        Garante que a thread de análise seja finalizada
        antes de fechar a aplicação.
        """
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        event.accept()
