import sounddevice as sd
import numpy as np
import time

# Configurações de áudio
fs = 44100  # Taxa de amostragem
duração = 1  # Duração do tom em segundos
freq = 440  # Frequência do tom (Lá 440Hz)

# Gera 1 segundo de tom senoidal a 440 Hz
t = np.linspace(0, duração, int(fs * duração), False)
tone = 0.2 * np.sin(2 * np.pi * freq * t)

# Obter todos os dispositivos de áudio
dispositivos = sd.query_devices()

print("\n=== DISPOSITIVOS DE SAÍDA DE ÁUDIO DISPONÍVEIS ===\n")

# Lista e testa apenas dispositivos de saída
dispositivos_saida = []
for i, dispositivo in enumerate(dispositivos):
    # Verificar se o dispositivo é de saída (tem canais de saída)
    if dispositivo['max_output_channels'] > 0:
        print(f"Índice: {i}")
        print(f"Nome: {dispositivo['name']}")
        print(f"Canais de saída: {dispositivo['max_output_channels']}")
        print(f"Taxa de amostragem padrão: {dispositivo['default_samplerate']} Hz")
        print("\n")
        dispositivos_saida.append(i)

# Testar cada dispositivo de saída com um tom
for idx in dispositivos_saida:
    try:
        print(f"\nTestando dispositivo {idx}: {dispositivos[idx]['name']}")
        print("Reproduzindo tom de teste (440Hz) por 1 segundo...")
        
        # Configurar o dispositivo de saída atual
        sd.default.device = idx
        
        # Reproduzir o tom e aguardar
        sd.play(tone, fs)
        sd.wait()
        
        print(f"Teste do dispositivo {idx} concluído!")
        print("-" * 50)
        time.sleep(0.5)  # Pausa entre testes
        
    except Exception as e:
        print(f"Erro ao testar dispositivo {idx}: {e}")

print("\nTodos os testes de saída de áudio foram concluídos!")
