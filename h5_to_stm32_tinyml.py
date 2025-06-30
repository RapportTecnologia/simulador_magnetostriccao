"""Ferramenta para converter modelos Keras para formatos TinyML."""

import argparse
import os
import sys
import tensorflow as tf

try:
    import numpy as np
except ImportError:
    np = None

def convert_h5_to_tflite(h5_path, tflite_path):
    """Converte arquivo ``.h5`` em modelo TFLite."""

    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")

def tflite_to_c_array(tflite_path, c_path, var_name="model_tflite"):
    """Converte arquivo TFLite em array C para uso em microcontroladores."""

    with open(tflite_path, "rb") as f:
        tflite_bytes = f.read()
    with open(c_path, "w") as f:
        f.write(f"const unsigned char {var_name}[] = {{\n")
        for i, b in enumerate(tflite_bytes):
            if i % 12 == 0:
                f.write("    ")
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"const unsigned int {var_name}_len = {len(tflite_bytes)};\n")
    print(f"C array saved to {c_path}")

def main():
    """Função principal do script de conversão."""

    parser = argparse.ArgumentParser(description="Converte arquivo .h5 para formato STM32 TinyML (TFLite e C array)")
    parser.add_argument("input_h5", help="Caminho para o arquivo .h5 do modelo treinado")
    parser.add_argument("--tflite", help="Caminho de saída do arquivo .tflite", default=None)
    parser.add_argument("--c", help="Caminho de saída do arquivo .h (array C)", default=None)
    parser.add_argument("--var", help="Nome da variável C", default="model_tflite")
    args = parser.parse_args()

    h5_path = args.input_h5
    if not os.path.isfile(h5_path):
        print(f"Arquivo {h5_path} não encontrado.")
        sys.exit(1)
    tflite_path = args.tflite or h5_path.replace(".h5", ".tflite")
    c_path = args.c or h5_path.replace(".h5", ".h")

    convert_h5_to_tflite(h5_path, tflite_path)
    tflite_to_c_array(tflite_path, c_path, args.var)

if __name__ == "__main__":
    main()
