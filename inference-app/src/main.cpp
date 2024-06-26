
#include "pico/time.h"
#include <iostream>
#include <chrono>

#include <malloc.h>

#include <algorithm>

#include "pico/stdlib.h"

#include "hardware/pwm.h"

extern "C" {
  #include "pico/pdm_microphone.h"

}

#include "tflite_model.h"

#include "dsp_pipeline.h"

#include "ml_model.h"

// constants
#define SAMPLE_RATE 16000
#define FFT_SIZE 256
#define SPECTRUM_SHIFT 32
#define INPUT_BUFFER_SIZE ((FFT_SIZE / 2) * SPECTRUM_SHIFT)
#define INPUT_SHIFT 0

// microphone configuration
const struct pdm_microphone_config pdm_config = {
  // GPIO pin for the PDM DAT signal
  .gpio_data = 2,

  // GPIO pin for the PDM CLK signal
  .gpio_clk = 3,

  // PIO instance to use
  .pio = pio0,

  // PIO State Machine instance to use
  .pio_sm = 0,

  // sample rate in Hz
  .sample_rate = SAMPLE_RATE,

  // number of samples to buffer
  .sample_buffer_size = INPUT_BUFFER_SIZE,
};

q15_t capture_buffer_q15[INPUT_BUFFER_SIZE];
volatile int new_samples_captured = 0;

q15_t input_q15[INPUT_BUFFER_SIZE + (FFT_SIZE / 2)];

DSPPipeline dsp_pipeline(FFT_SIZE);
MLModel ml_model(tflite_model, 64 * 1024);

int8_t * scaled_spectrum = nullptr;
int32_t spectogram_divider;
float spectrogram_zero_point;
uint32_t getTotalHeap(void);
uint32_t getFreeHeap(void);
void on_pdm_samples_ready();
#define DEBOUNCE_INTERVAL_MS 1000 // Mindestzeitspanne in Millisekunden

uint64_t last_go_time = 0;
uint64_t last_up_time = 0;
const int BAR_LENGTH = 20; // Maximale Länge des Balkens
int main(void) {
  // initialize stdio
  stdio_init_all();

  printf("hello pico fire alarm detection\n");

  gpio_set_function(PICO_DEFAULT_LED_PIN, GPIO_FUNC_PWM);

  uint pwm_slice_num = pwm_gpio_to_slice_num(PICO_DEFAULT_LED_PIN);
  uint pwm_chan_num = pwm_gpio_to_channel(PICO_DEFAULT_LED_PIN);

  // Set period of 256 cycles (0 to 255 inclusive)
  pwm_set_wrap(pwm_slice_num, 256);

  // Set the PWM running
  pwm_set_enabled(pwm_slice_num, true);

  if (!ml_model.init()) {
    printf("Failed to initialize ML model!\n");
    while (1) {
      tight_loop_contents();
    }
  }

  if (!dsp_pipeline.init()) {
    printf("Failed to initialize DSP Pipeline!\n");
    while (1) {
      tight_loop_contents();
    }
  }

  scaled_spectrum = (int8_t * ) ml_model.input_data();
  spectogram_divider = 64 * ml_model.input_scale();
  spectrogram_zero_point = ml_model.input_zero_point();

  // initialize the PDM microphone
  if (pdm_microphone_init( & pdm_config) < 0) {
    printf("PDM microphone initialization failed!\n");
    while (1) {
      tight_loop_contents();
    }
  }

  // set callback that is called when all the samples in the library
  // internal sample buffer are ready for reading
  pdm_microphone_set_samples_ready_handler(on_pdm_samples_ready);

  // start capturing data from the PDM microphone
  if (pdm_microphone_start() < 0) {
    printf("PDM microphone start failed!\n");
    while (1) {
      tight_loop_contents();
    }
  }

  while (1) {
    printf("\033[H");

    // wait for new samples
    while (new_samples_captured == 0) {}
    new_samples_captured = 0;

    dsp_pipeline.shift_spectrogram(scaled_spectrum, SPECTRUM_SHIFT, 124);

    // move input buffer values over by INPUT_BUFFER_SIZE samples
    memmove(input_q15, & input_q15[INPUT_BUFFER_SIZE], (FFT_SIZE / 2));

    // copy new samples to end of the input buffer with a bit shift of INPUT_SHIFT
    arm_shift_q15(capture_buffer_q15, INPUT_SHIFT, input_q15 + (FFT_SIZE / 2), INPUT_BUFFER_SIZE);

    for (int i = 0; i < SPECTRUM_SHIFT; i++) {
      dsp_pipeline.calculate_spectrum(
        input_q15 + i * ((FFT_SIZE / 2)),
        scaled_spectrum + (129 * (124 - SPECTRUM_SHIFT + i)),
        spectogram_divider, spectrogram_zero_point
      );
    }
    absolute_time_t start = get_absolute_time();
    std::vector < float > prediction = ml_model.predict();

    absolute_time_t end = get_absolute_time();
    int64_t time_taken_us = absolute_time_diff_us(start, end);
    uint64_t current_time = time_us_64();

    printf("up: %6.2f%% (%.4f) [", prediction[0] * 100, prediction[0]);
    int barSize = static_cast<int>(prediction[0] * BAR_LENGTH);
    for (int j = 0; j < barSize; j++) printf("#");
    for (int j = barSize; j < BAR_LENGTH; j++) printf(" ");
    printf("]\n");

    printf("go: %6.2f%% (%.4f) [", prediction[1] * 100, prediction[1]);
    barSize = static_cast<int>(prediction[1] * BAR_LENGTH);
    for (int j = 0; j < barSize; j++) printf("#");
    for (int j = barSize; j < BAR_LENGTH; j++) printf(" ");
    printf("]\n");

    printf("un: %6.2f%% (%.4f) [", prediction[2] * 100, prediction[2]);
    barSize = static_cast<int>(prediction[2] * BAR_LENGTH);
    for (int j = 0; j < barSize; j++) printf("#");
    for (int j = barSize; j < BAR_LENGTH; j++) printf(" ");
    printf("]\n");

    printf("bg: %6.2f%% (%.4f) [", prediction[3] * 100, prediction[3]);
    barSize = static_cast<int>(prediction[3] * BAR_LENGTH);
    for (int j = 0; j < barSize; j++) printf("#");
    for (int j = barSize; j < BAR_LENGTH; j++) printf(" ");
    printf("]\n");

    printf("Time taken by predict() is: %.8f sec\n", time_taken_us / 1e6);
    printf("Total Heap: %u\n", getTotalHeap());
    printf("Free Heap: %u\n", getFreeHeap());
  }

  return 0;
}

void on_pdm_samples_ready() {

  // read in the new samples
  new_samples_captured = pdm_microphone_read(capture_buffer_q15, INPUT_BUFFER_SIZE);

}

uint32_t getTotalHeap(void) {
  extern char __StackLimit, __bss_end__;

  return & __StackLimit - & __bss_end__;
}

uint32_t getFreeHeap(void) {
  struct mallinfo m = mallinfo();

  return getTotalHeap() - m.uordblks;
}