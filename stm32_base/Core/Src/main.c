// Exemplo mínimo de inferência com TensorFlow Lite Micro em STM32
// Adapte para sua placa e ambiente (CubeIDE, Makefile, etc)
#include "model_tflite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Ajuste conforme sua RAM disponível
#define TENSOR_ARENA_SIZE 16 * 1024
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

int main(void) {
    // Inicialização mínima do ambiente STM32 (HAL, clock, etc)
    // HAL_Init();
    // SystemClock_Config();

    // 1. Inicializar o erro reporter
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // 2. Mapear o modelo
    const tflite::Model* model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Modelo TFLite schema version incompatível!");
        while (1);
    }

    // 3. Resolver de operadores
    static tflite::AllOpsResolver resolver;

    // 4. Criar o intérprete
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);

    // 5. Alocar tensores
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        error_reporter->Report("Falha ao alocar tensores!");
        while (1);
    }

    // 6. Obter ponteiros para input/output
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // Exemplo: preencher input (ajuste para seu modelo)
    for (int i = 0; i < input->bytes / sizeof(float); i++) {
        input->data.f[i] = 0.0f; // Preencha com dados reais
    }

    // 7. Executar inferência
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Falha na inferência!");
        while (1);
    }

    // 8. Ler saída
    for (int i = 0; i < output->bytes / sizeof(float); i++) {
        float result = output->data.f[i];
        // Use o resultado (ex: print via UART, LED, etc)
    }

    while (1) {
        // Loop principal
    }
}
