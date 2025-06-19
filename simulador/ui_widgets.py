# ui_widgets.py - Widgets personalizados para a interface

from PyQt5.QtWidgets import QProgressBar, QLabel
from PyQt5.QtCore import Qt


class VUWidget(QProgressBar):
    """
    VUWidget: Barra vertical que exibe o nível RMS do sinal de áudio em tempo real.

    - Herda de QProgressBar para facilidade de uso.
    - Valores de 0 (silêncio) a 1000 (pico de RMS mapeado).
    """
    def __init__(self):
        super().__init__()

        # Define orientação vertical
        self.setOrientation(Qt.Vertical)

        # Define intervalo de exibição de 0 a 1000
        self.setRange(0, 1000)

        # Inicia com valor zero
        self.setValue(0)

    def update_level(self, rms_value: float):
        """
        Atualiza o nível exibido conforme o valor RMS calculado.

        Parâmetros:
        - rms_value: valor RMS normalizado (0.0 a 1.0).
        """
        # Converte para escala interna
        display = int(rms_value * 1000)

        # Atualiza barra
        self.setValue(display)


class ColorBox(QLabel):
    """
    ColorBox: Label que muda de cor conforme a classe prevista.

    - 0: Normal (verde)
    - 1: Intermediário (amarelo)
    - 2: Falha (vermelho)
    """
    # Mapas de cor e texto por classe
    COLORS = {
        0: '#4caf50',    # Verde
        1: '#ffeb3b',    # Amarelo
        2: '#f44336'     # Vermelho
    }
    LABELS = {
        0: 'Normal',
        1: 'Intermediário',
        2: 'Falha'
    }

    def __init__(self):
        super().__init__()

        # Alinha texto ao centro
        self.setAlignment(Qt.AlignCenter)

        # Texto inicial
        self.setText('Status: -')

        # Altura fixa para consistência
        self.setFixedHeight(40)

        # Estilo inicial
        self.setStyleSheet('border: 1px solid #000000; background-color: #ffffff;')

    def update_status(self, cls: int):
        """
        Atualiza cor de fundo e texto conforme a classe prevista.

        Parâmetros:
        - cls: inteiro (0, 1 ou 2).
        """
        # Obtém cor e rótulo
        color = self.COLORS.get(cls, '#ffffff')
        label = self.LABELS.get(cls, '-')

        # Atualiza texto
        self.setText(f'Status: {label}')

        # Atualiza cor de fundo
        self.setStyleSheet(
            f'border: 1px solid #000000; background-color: {color};'
        )
