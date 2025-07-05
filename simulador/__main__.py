import sys
import argparse
import os
import pkgutil
import numpy as np
import scipy.signal
import librosa
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from .analyzer_app import AnalyzerApp, SR, CUTOFF_FREQ, DURATION, N_MFCC, EPOCHS, BATCH_SIZE

def main(root_dir, model_path='model.h5', train=False, analyze=False, classify=False, analysis_method='fft'):
    """
    Função principal para iniciar a aplicação GUI.
    :param root_dir: Caminho para a pasta contendo train/ e test/
    :param model_path: Caminho para arquivo .h5 de modelo pré-treinado
    """
    # Define pastas de treino e teste
    # Define pastas de treino e teste; use root_dir if subfolder não existir
    default_train = os.path.join(root_dir, 'train')
    default_test = os.path.join(root_dir, 'test')
    train_folder = default_train if os.path.isdir(default_train) else root_dir
    test_folder = default_test if os.path.isdir(default_test) else root_dir

    # Inicia a aplicação Qt
    app = QApplication(sys.argv)
    window = AnalyzerApp(
        train_folder,
        test_folder,
        model_path,
        analysis_method
    )
    window.show()
    if train:
        window.pending_analyze = analyze
        window.pending_classify = classify
        QTimer.singleShot(0, window._train_model)
    else:
        if analyze:
            QTimer.singleShot(0, window._toggle_analysis)
        elif classify:
            QTimer.singleShot(0, window._toggle_classification)
    sys.exit(app.exec_())

if __name__ == "__main__":
    # Parser de argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description='Analisador de Magnetostricção GUI'
    )
    parser.add_argument(
        'root_dir',
        help='Caminho para a pasta contendo train/ e test/'
    )
    parser.add_argument(
        '--model',
        help='Caminho para arquivo .h5 de modelo pré-treinado',
        default='model.h5'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Treina a rede neural usando dados de treino e salva o modelo'
    )
    parser.add_argument(
        '--save-model',
        help='Caminho para salvar o modelo treinado; se omitido, grava sobre modelo padrão',
        default=None
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Inicia análise de arquivos de teste; requer modelo existente'
    )
    parser.add_argument(
        '--no_gpu',
        action='store_true',
        help='Ativa ou desativa o uso da GPU, o default é desativado',
    )
    parser.add_argument(
        '--classify',
        action='store_true',
        help='Inicia classificação de arquivos de teste; requer modelo existente'
    )
    # Descobre métodos de análise dinamicamente com base em módulos no diretório analyzers
    analyzer_dir = os.path.join(os.path.dirname(__file__), 'analyzers')
    methods = [name[:-len('_analyzer')] for _, name, _ in pkgutil.iter_modules([analyzer_dir]) if name.endswith('_analyzer')]
    parser.add_argument(
        '-a', '--analysis-method',
        choices=methods,
        default=methods[0] if methods else None,
        help='Método de análise de frequências a usar'
    )
    args = parser.parse_args()

    root_dir = args.root_dir
    model_path = args.model

    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if args.analyze and args.classify:
        parser.error('As flags --analyze e --classify não podem ser usadas juntas')
    main(root_dir, model_path, train=args.train, analyze=args.analyze, classify=args.classify, analysis_method=args.analysis_method)
