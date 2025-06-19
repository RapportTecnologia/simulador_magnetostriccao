# Projeto Base STM32 TinyML

Este diretório contém um esqueleto para um projeto STM32 voltado para integração de modelos TinyML (TensorFlow Lite Micro).

## Estrutura Sugerida

```
stm32_base/
├── Core/
│   ├── Inc/
│   │   └── model_tflite.h        # Array C gerado do modelo
│   └── Src/
│       └── main.c                # Exemplo de uso
├── Drivers/                      # Drivers STM32 (HAL/LL)
├── Middleware/                   # TensorFlow Lite Micro
└── Makefile                      # Script de build (exemplo)
```

## Passos Básicos

1. **Converta seu modelo .h5** usando o script `h5_to_stm32_tinyml.py`:

   ```bash
   python h5_to_stm32_tinyml.py model_SVM.h5 --c stm32_base/Core/Inc/model_tflite.h
   ```

2. **Copie o arquivo gerado** para o diretório do seu projeto STM32, caso esteja usando outro repositório.

3. **Inclua o array** no seu código C:

   ```c
   #include "model_tflite.h"
   // Use model_tflite e model_tflite_len
   ```

4. **Integre o TensorFlow Lite Micro**
   - Baixe o [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro) e adicione à pasta `Middleware/`.
   - Inclua as fontes e headers no seu projeto STM32CubeIDE ou Makefile.

5. **Implemente a inferência**
   - Veja um exemplo mínimo em `main.c` (não implementado ainda).

## Observações
- O projeto base é apenas um ponto de partida. Adapte conforme sua placa STM32, periféricos e ambiente de build (CubeIDE, Makefile, PlatformIO).
- Para placas STM32H7/L4/F4/N6, verifique a RAM disponível para rodar modelos TinyML.
- Consulte a [documentação oficial do TFLite Micro](https://www.tensorflow.org/lite/microcontrollers) para detalhes de integração.
