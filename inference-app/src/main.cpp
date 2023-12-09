#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/pwm.h"
extern "C" {
#include "pico/pdm_microphone.h"
}
#include "tflite_model.h"
#include "dsp_pipeline.h"
#include "ml_model.h"

// Konstanten und Mikrofonkonfiguration
#define SAMPLE_RATE       16000
#define FFT_SIZE          256
#define SPECTRUM_SHIFT    4
#define INPUT_BUFFER_SIZE ((FFT_SIZE / 2) * SPECTRUM_SHIFT)
#define INPUT_SHIFT       0

const struct pdm_microphone_config pdm_config = {
    .gpio_data = 2,
    .gpio_clk = 3,
    .pio = pio0,
    .pio_sm = 0,
    .sample_rate = SAMPLE_RATE,
    .sample_buffer_size = INPUT_BUFFER_SIZE,
};

q15_t capture_buffer_q15[INPUT_BUFFER_SIZE];
volatile int new_samples_captured = 0;

q15_t input_q15[INPUT_BUFFER_SIZE + (FFT_SIZE / 2)];

DSPPipeline dsp_pipeline(FFT_SIZE);
MLModel ml_model(tflite_model, 128 * 1024);

int8_t* scaled_spectrum = nullptr;
int32_t spectogram_divider;
float spectrogram_zero_point;

void on_pdm_samples_ready() {
    new_samples_captured = pdm_microphone_read(capture_buffer_q15, INPUT_BUFFER_SIZE);
}

int main(void) {
    stdio_init_all();
    printf("Wake Word Detection\n");

    gpio_set_function(PICO_DEFAULT_LED_PIN, GPIO_FUNC_PWM);
    uint pwm_slice_num = pwm_gpio_to_slice_num(PICO_DEFAULT_LED_PIN);
    uint pwm_chan_num = pwm_gpio_to_channel(PICO_DEFAULT_LED_PIN);

    pwm_set_wrap(pwm_slice_num, 256);
    pwm_set_enabled(pwm_slice_num, true);

    if (!ml_model.init()) {
        printf("Failed to initialize ML model!\n");
        while (1) { tight_loop_contents(); }
    }

    if (!dsp_pipeline.init()) {
        printf("Failed to initialize DSP Pipeline!\n");
        while (1) { tight_loop_contents(); }
    }

    scaled_spectrum = (int8_t*)ml_model.input_data();
    spectogram_divider = 64 * ml_model.input_scale();
    spectrogram_zero_point = ml_model.input_zero_point();

    if (pdm_microphone_init(&pdm_config) < 0) {
        printf("PDM microphone initialization failed!\n");
        while (1) { tight_loop_contents(); }
    }

    pdm_microphone_set_samples_ready_handler(on_pdm_samples_ready);

    if (pdm_microphone_start() < 0) {
        printf("PDM microphone start failed!\n");
        while (1) { tight_loop_contents(); }
    }

    while (1) {
        while (new_samples_captured == 0) {
            tight_loop_contents();
        }
        new_samples_captured = 0;

        dsp_pipeline.shift_spectrogram(scaled_spectrum, SPECTRUM_SHIFT, 124);
        memmove(input_q15, &input_q15[INPUT_BUFFER_SIZE], (FFT_SIZE / 2));
        arm_shift_q15(capture_buffer_q15, INPUT_SHIFT, input_q15 + (FFT_SIZE / 2), INPUT_BUFFER_SIZE);

        for (int i = 0; i < SPECTRUM_SHIFT; i++) {
            dsp_pipeline.calculate_spectrum(
                input_q15 + i * ((FFT_SIZE / 2)),
                scaled_spectrum + (129 * (124 - SPECTRUM_SHIFT + i)),
                spectogram_divider, spectrogram_zero_point
            );
        }

        std::vector<float> predictions = ml_model.predict();
        int max_index = std::distance(predictions.begin(), std::max_element(predictions.begin(), predictions.end()));

        if (max_index == 0) {
            //printf("Detected: Silence\n");
        } else if (max_index == 1) {
            printf("Detected: Unknown\n");
        } else if (max_index == 2) {
            printf("Detected: Yes\n");
        } else if (max_index == 3) {
            printf("Detected: No\n");
        } else {
            printf("Detected: Unknown class with index %d\n", max_index);
        }
    }

    return 0;
}
