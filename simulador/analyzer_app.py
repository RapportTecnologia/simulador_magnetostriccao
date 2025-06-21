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
    QDialog,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QProgressBar,
    QSlider,
    QLabel,
    QComboBox,
    QListWidget, QGroupBox, QTabWidget,
    QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
import statistics
import psutil
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Importa módulo de thread de análise com sinais estendidos
from .analysis_thread import FileAnalysisThread
# Importa widgets personalizados de UI
from .ui_widgets import VUWidget, ColorBox, ClickableLabel
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
        self.btn_analysis.setStyleSheet('background-color: green;')
        ctrl_layout.addWidget(self.btn_analysis)

        # Botão para treinar o modelo
        self.btn_train = QPushButton('Treinar Modelo')
        self.btn_train.setStyleSheet('background-color: green;')
        ctrl_layout.addWidget(self.btn_train)

        # Botão para iniciar classificação de testes
        self.btn_classify = QPushButton('Iniciar Classificação')
        self.btn_classify.setStyleSheet('background-color: green;')
        ctrl_layout.addWidget(self.btn_classify)
        self.btn_play_noises = QPushButton('Tocar Ruídos')
        self.btn_play_noises.setStyleSheet('background-color: cyan;')
        ctrl_layout.addWidget(self.btn_play_noises)
        self.btn_play_noises.clicked.connect(self._open_play_noises_dialog)

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
        # Inicializa labels de estatísticas
        self.lbl_time_stats = QLabel('Tempo total: 0.00s | Tempo médio: 0.00s/s', alignment=Qt.AlignCenter)
        self.lbl_file_stats = QLabel('Duração: 0.00s | Tamanho: 0 bytes', alignment=Qt.AlignCenter)
        self.lbl_total_stats = QLabel('Total duração: 0.00s | Total bytes: 0 bytes', alignment=Qt.AlignCenter)
        self.lbl_overall_stats = ClickableLabel('Arquivos: 0 | Média bytes/s: 0.00 | Média bytes/arquivo: 0', alignment=Qt.AlignCenter)
        self.lbl_overall_stats.clicked.connect(self._show_stats_dialog)
        # Estatísticas agrupadas
        group_tempo = QGroupBox('Tempo')
        tempo_lay = QVBoxLayout()
        tempo_lay.addWidget(self.lbl_time_stats)
        group_tempo.setLayout(tempo_lay)
        group_audio = QGroupBox('Áudio')
        audio_lay = QVBoxLayout()
        audio_lay.addWidget(self.lbl_file_stats)
        audio_lay.addWidget(self.lbl_total_stats)
        group_audio.setLayout(audio_lay)
        group_geral = QGroupBox('Geral')
        geral_lay = QVBoxLayout()
        geral_lay.addWidget(self.lbl_overall_stats)
        group_geral.setLayout(geral_lay)
        stats_grid = QGridLayout()
        stats_grid.addWidget(group_tempo, 0, 0)
        stats_grid.addWidget(group_audio, 0, 1)
        stats_grid.addWidget(group_geral, 1, 0)
        disp_layout.addLayout(stats_grid)










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
        Callback do botão de análise: inicia ou para a FileAnalysisThread em streaming ao vivo.
        """
        # Se já existe e está rodando, para a thread
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.btn_analysis.setText('Iniciar Análise')
            self.btn_analysis.setStyleSheet('background-color: green;')
            self.statusBar().showMessage('Análise de streaming parada')
            return

        # Se não há modelo, avisa e retorna
        if self.model is None:
            QMessageBox.warning(self, 'Aviso', 'Treine ou carregue um modelo primeiro.')
            return

        # Atualiza texto do botão e status bar
        self.btn_analysis.setText('Parar Análise')
        self.btn_analysis.setStyleSheet('background-color: red;')
        self.statusBar().showMessage('Análise de streaming iniciada')

        # Cria e configura a FileAnalysisThread para áudio ao vivo
        self.thread = FileAnalysisThread(
            self.b,
            self.a,
            SR,
            DURATION,
            self.model,
            self.model.input_shape[2]
        )

        # Conecta sinais para atualizar UI
        self.thread.spec_ready.connect(self._update_spectrogram)
        self.thread.mel_ready.connect(self._update_mel_bands)
        self.thread.fft_ready.connect(self._update_fft)
        self.thread.class_ready.connect(self.color_box.update_status)

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
        # Inicializa métricas de tempo
        self.load_times = []
        self.preproc_times = []
        self.inf_times = []
        self.total_load_time = 0.0
        self.total_preproc_time = 0.0
        self.total_inf_time = 0.0
        self.cpu_usages = []
        self.mem_usages = []
        self.total_cpu_usage = 0.0
        self.total_mem_usage = 0.0
        # Conecta sinal de métricas detalhadas
        self.classification_thread.detailed_metrics.connect(self._update_detailed_stats)
        self.total_files = 0
        self.lbl_overall_stats.setText('Arquivos: 0 | Média bytes/s: 0.00 | Média bytes/arquivo: 0')
        self.total_proc_time = 0.0
        self.total_audio_dur = 0.0
        self.total_bytes = 0
        self.lbl_time_stats.setText('Tempo total: 0.00s | Tempo médio: 0.00s/s')
        self.lbl_file_stats.setText('Duração: 0.00s | Tamanho: 0 bytes')
        self.lbl_total_stats.setText('Total duração: 0.00s | Total bytes: 0 bytes')
        # Conecta sinal de métricas de processamento
        self.classification_thread.processing_metrics.connect(self._update_time_stats)
        # Conecta sinal de métricas detalhadas
        self.classification_thread.progress.connect(self._update_progress)
        self.classification_thread.detailed_metrics.connect(self._update_detailed_stats)
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

    def _update_time_stats(self, file_time, file_duration, file_size):
        """
        Atualiza tempo total, duração e coleta de recursos por arquivo processado.
        """
        # Incrementa métricas gerais
        self.total_files += 1
        self.total_proc_time += file_time
        self.total_audio_dur += file_duration
        self.total_bytes += file_size
        # Atualiza widgets de tempo e bytes
        avg_time = self.total_proc_time / self.total_audio_dur if self.total_audio_dur else 0
        self.lbl_time_stats.setText(f'Tempo total: {self.total_proc_time:.2f}s | Tempo médio: {avg_time:.2f}s/s')
        self.lbl_file_stats.setText(f'Duração: {file_duration:.2f}s | Tamanho: {file_size} bytes')
        self.lbl_total_stats.setText(f'Total duração: {self.total_audio_dur:.2f}s | Total bytes: {self.total_bytes} bytes')
        avg_bytes_per_sec = self.total_bytes / self.total_audio_dur if self.total_audio_dur else 0
        avg_bytes_per_file = self.total_bytes / self.total_files if self.total_files else 0
        self.lbl_overall_stats.setText(f'Arquivos: {self.total_files} | Média bytes/s: {avg_bytes_per_sec:.2f} | Média bytes/arquivo: {avg_bytes_per_file:.0f}')
        # Coleta uso de CPU e memória ao final do processamento de cada arquivo
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        self.cpu_usages.append(cpu)
        self.mem_usages.append(mem)

    def _update_detailed_stats(self, load_time, preproc_time, inference_time):
        """Recebe métricas detalhadas de tempo de cada etapa e armazena."""
        # Armazena tempos de cada etapa
        self.load_times.append(load_time)
        self.preproc_times.append(preproc_time)
        self.inf_times.append(inference_time)
        # Acumula tempos totais
        self.total_load_time += load_time
        self.total_preproc_time += preproc_time
        self.total_inf_time += inference_time

    def _show_stats_dialog(self):
        # Monta métricas dinâmicas propostas
        proposed_metrics = {}
        # Latências por etapa
        avg_load = sum(self.load_times)/len(self.load_times) if self.load_times else 0
        avg_pre = sum(self.preproc_times)/len(self.preproc_times) if self.preproc_times else 0
        avg_inf = sum(self.inf_times)/len(self.inf_times) if self.inf_times else 0
        proposed_metrics['Latências por etapa'] = {
            'I/O média (s)': f"{avg_load:.3f}",
            'Pré-processamento média (s)': f"{avg_pre:.3f}",
            'Inferência média (s)': f"{avg_inf:.3f}"
        }
        # Estatísticas de latência
        stats = {}
        stats['I/O mediana (s)'] = f"{statistics.median(self.load_times):.3f}" if self.load_times else "0.000"
        stats['I/O p90 (s)'] = f"{statistics.quantiles(self.load_times, n=100)[89]:.3f}" if self.load_times else "0.000"
        stats['Pré mediana (s)'] = f"{statistics.median(self.preproc_times):.3f}" if self.preproc_times else "0.000"
        stats['Pré p90 (s)'] = f"{statistics.quantiles(self.preproc_times, n=100)[89]:.3f}" if self.preproc_times else "0.000"
        stats['Inf mediana (s)'] = f"{statistics.median(self.inf_times):.3f}" if self.inf_times else "0.000"
        stats['Inf p90 (s)'] = f"{statistics.quantiles(self.inf_times, n=100)[89]:.3f}" if self.inf_times else "0.000"
        proposed_metrics['Estatísticas de latência'] = stats
        # Throughput
        files_per_sec = self.total_files/self.total_proc_time if self.total_proc_time else 0
        bytes_per_sec = self.total_bytes/self.total_proc_time if self.total_proc_time else 0
        proposed_metrics['Throughput'] = {
            'Arquivos/s': f"{files_per_sec:.3f}",
            'Bytes/s': f"{bytes_per_sec:.0f}"
        }
        # Uso de recursos
        # Valores coletados durante processamento
        if self.cpu_usages:
            avg_cpu = sum(self.cpu_usages)/len(self.cpu_usages)
            median_cpu = statistics.median(self.cpu_usages)
            p90_cpu = statistics.quantiles(self.cpu_usages, n=100)[89]
        else:
            avg_cpu = median_cpu = p90_cpu = 0.0
        if self.mem_usages:
            avg_mem = sum(self.mem_usages)/len(self.mem_usages)
            median_mem = statistics.median(self.mem_usages)
            p90_mem = statistics.quantiles(self.mem_usages, n=100)[89]
        else:
            avg_mem = median_mem = p90_mem = 0.0
        proposed_metrics['Uso de recursos'] = {
            'CPU média (%)': f"{avg_cpu:.1f}",
            'CPU mediana (%)': f"{median_cpu:.1f}",
            'CPU p90 (%)': f"{p90_cpu:.1f}",
            'Memória média (%)': f"{avg_mem:.1f}",
            'Memória mediana (%)': f"{median_mem:.1f}",
            'Memória p90 (%)': f"{p90_mem:.1f}"
        }
        # Métricas do modelo
        param_count = self.model.count_params() if hasattr(self.model, 'count_params') else 'N/A'
        model_size = os.path.getsize(self.model_path)/1024/1024 if os.path.exists(self.model_path) else 0
        proposed_metrics['Métricas do modelo'] = {
            'Parâmetros': str(param_count),
            'Tamanho do modelo (MB)': f"{model_size:.2f}"
        }
        # Qualidade de classificação
        proposed_metrics['Qualidade de classificação'] = {
            'Matriz de confusão': 'N/A',
            'Precision/Recall/F1': 'N/A',
            'ROC/AUC': 'N/A'
        }
        # Monta diálogo
        current_stats = {
            'Tempo total (s)': f"{self.total_proc_time:.2f}",
            'Tempo médio (s/s)': f"{(self.total_proc_time/self.total_audio_dur if self.total_audio_dur else 0):.2f}",
            'Duração total (s)': f"{self.total_audio_dur:.2f}",
            'Total bytes': f"{self.total_bytes}",
            'Total arquivos': f"{self.total_files}"
        }
        dialog = StatsDialog(self, current_stats, proposed_metrics)
        dialog.exec_()


    def _play_noise(self, file_path):
        """Toca arquivo de áudio especificado."""
        try:
            # Stop previous playback
            if hasattr(self, 'playback_timer'):
                self.playback_timer.stop()
            sounddevice.stop()
            # Load audio
            y, sr = librosa.load(file_path, sr=SR)
            # Setup playback variables
            self.playback_data = y
            self.playback_sr = sr
            self.playback_total = len(y) / sr
            self.playback_offset = 0.0
            self.playback_start_time = time.time()
            # Timer for tracker
            if not hasattr(self, 'playback_timer'):
                self.playback_timer = QTimer(self)
                self.playback_timer.timeout.connect(self._update_playback_time)
            self.playback_timer.start(100)
            # Start audio
            sounddevice.play(self.playback_data, self.playback_sr)
            # Setup pause/resume button
            self.btn_play_pause.setText("Parar")
            self.btn_play_pause.setEnabled(True)
            # Start classification of noise
            try:
                if hasattr(self, 'noise_class_thread') and self.noise_class_thread.isRunning():
                    self.noise_class_thread.stop()
                    self.noise_class_thread.wait()
                self.noise_class_thread = ClassificationThread([file_path], self.model, self.b, self.a, SR, DURATION, N_MFCC, CUTOFF_FREQ)
                self.noise_class_thread.class_ready.connect(self._update_noise_class)
                self.noise_class_thread.start()
            except Exception as e:
                print(f"Erro ao iniciar classificação de ruído: {e}")
        except Exception as e:
            QMessageBox.warning(self, 'Erro ao tocar áudio', str(e))

    def _open_play_noises_dialog(self):
        """Abre diálogo para selecionar e tocar ruídos de treino e teste em abas, agrupados por classe."""
        dialog = QDialog(self)
        dialog.setWindowTitle('Tocar Ruídos')
        layout = QVBoxLayout(dialog)
        # Playback tracker and control
        tracker_layout = QHBoxLayout()
        self.lbl_play_time = QLabel("00:00 / 00:00")
        self.lbl_noise_class = QLabel("Classe: N/A")
        self.btn_play_pause = QPushButton("Parar")
        tracker_layout.addWidget(self.lbl_play_time)
        tracker_layout.addWidget(self.lbl_noise_class)
        tracker_layout.addWidget(self.btn_play_pause)
        layout.addLayout(tracker_layout)
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)
        tabs = QTabWidget(dialog)
        mapping = {'0': '0 (Normal)', '1': '1 (Intermediário)', '2': '2 (Falha)'}
        for label, dir_path in [('Treino', self.train_dir), ('Teste', self.test_dir)]:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            class_tabs = QTabWidget(tab)
            if os.path.isdir(dir_path):
                for class_name in sorted(os.listdir(dir_path)):
                    class_path = os.path.join(dir_path, class_name)
                    if os.path.isdir(class_path):
                        class_tab = QWidget()
                        class_layout = QVBoxLayout(class_tab)
                        list_widget = QListWidget(class_tab)
                        for fname in sorted(os.listdir(class_path)):
                            if fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                                list_widget.addItem(fname)
                        list_widget.itemClicked.connect(lambda item, p=class_path: self._play_noise(os.path.join(p, item.text())))
                        class_layout.addWidget(list_widget)
                        class_tab.setLayout(class_layout)
                        class_tabs.addTab(class_tab, mapping.get(class_name, class_name))
            tab_layout.addWidget(class_tabs)
            tab.setLayout(tab_layout)
            tabs.addTab(tab, label)
        layout.addWidget(tabs)
        dialog.resize(800, 600)
        # Stop playback timer immediately when dialog finishes
        dialog.finished.connect(lambda _: self.playback_timer.stop() if hasattr(self, 'playback_timer') else None)
        dialog.exec_()

    def _toggle_play_pause(self):
        """Pauses or resumes audio playback."""
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            # pause
            self.playback_timer.stop()
            elapsed = time.time() - self.playback_start_time
            self.playback_offset = elapsed
            sounddevice.stop()
            self.btn_play_pause.setText("Continuar")
        else:
            # resume
            start_idx = int(self.playback_offset * self.playback_sr)
            remaining = self.playback_data[start_idx:]
            sounddevice.play(remaining, self.playback_sr)
            self.playback_start_time = time.time() - self.playback_offset
            self.playback_timer.start(100)
            self.btn_play_pause.setText("Parar")

    def _update_playback_time(self):
        """Updates playback time tracker label."""
        # Safely update label; stop timer if label deleted
        try:
            elapsed = time.time() - self.playback_start_time
            total = getattr(self, 'playback_total', 0)
            if elapsed >= total:
                elapsed = total
                self.playback_timer.stop()
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            total_str = time.strftime("%M:%S", time.gmtime(total))
            self.lbl_play_time.setText(f"{elapsed_str} / {total_str}")
        except Exception:
            # QLabel deleted, stop timer
            if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
                self.playback_timer.stop()

    def _update_noise_class(self, predicted_class):
        """Atualiza label de classificação do ruído."""
        labels = {0: "Normal", 1: "Intermediário", 2: "Falha"}
        self.lbl_noise_class.setText(f"Classe: {labels.get(predicted_class, predicted_class)}")

    def closeEvent(self, event):
        """
        Garante que a thread de análise seja finalizada
        antes de fechar a aplicação.
        """
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        event.accept()


class StatsDialog(QDialog):
    def __init__(self, parent, current_stats, proposed_metrics):
        super().__init__(parent)
        self.setWindowTitle('Detalhes de Estatísticas')
        layout = QVBoxLayout(self)
        # Métricas Atuais
        group_current = QGroupBox('Métricas Atuais')
        v1 = QVBoxLayout()
        for k, v in current_stats.items():
            v1.addWidget(QLabel(f"{k}: {v}"))
        group_current.setLayout(v1)
        layout.addWidget(group_current)
        # Métricas Propostas Dinâmicas
        prop_grid = QGridLayout()
        for idx, (cat, items) in enumerate(proposed_metrics.items()):
            box = QGroupBox(cat)
            bl = QVBoxLayout()
            if isinstance(items, dict):
                for name, val in items.items():
                    bl.addWidget(QLabel(f"{name}: {val}"))
            else:
                for item in items:
                    bl.addWidget(QLabel(item))
            box.setLayout(bl)
            row = idx // 2
            col = idx % 2
            prop_grid.addWidget(box, row, col)
        layout.addLayout(prop_grid)
        self.resize(600, 800)

