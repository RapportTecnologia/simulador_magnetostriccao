import sys
import argparse
import os
from PyQt5.QtWidgets import QApplication
from .analyzer_app import AnalyzerApp

def main(root_dir, model_path='model.h5'):
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
        model_path
    )
    window.show()
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
    args = parser.parse_args()

    main(args.root_dir, args.model)
