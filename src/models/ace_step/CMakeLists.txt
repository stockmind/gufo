# ACE-Step 1.5 XL (turbo / sft / base, music): Qwen3 tokenizer and 0.6B
# text encoder, condition encoder, DiT decoder with APG guidance, Oobleck VAE
# decoder and encoder, the sampler mathematics (CPU) and the music pipeline
# with the official tasks (text2music, repaint, cover, cover-nofsq, extract,
# lego, complete). gufo-studio-windows only (no upstream capability); added from
# src/cli/serve/media/media.cmake.
#
# CPU: tokenizer, prompts, noise, WAV and the model file layout (prompt.cpp),
# enough for request handling and the token oracle test.
add_library(gufo_ace_step STATIC EXCLUDE_FROM_ALL tokenizer.cpp prompt.cpp preview.cpp
  sampler.cpp audio_io.cpp)
target_include_directories(gufo_ace_step PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_ace_step PUBLIC gufo_core gufo_qwen_text ICU::i18n)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_ace_step PRIVATE weights.hip kernels.hip attention.hip
    model.hip vae.hip gemm.cpp pipeline.cpp parallel_read.cpp)
  target_compile_options(gufo_ace_step PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
  target_link_libraries(gufo_ace_step PUBLIC gufo_platform hip::host
    roc::hipblaslt)

  add_executable(gufo-ace-check EXCLUDE_FROM_ALL check_main.cpp)
  set_target_properties(gufo-ace-check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-ace-check PRIVATE gufo_ace_step)
  gufo_stage_hip_runtime(gufo-ace-check)

  add_executable(gufo-ace EXCLUDE_FROM_ALL music_main.cpp)
  set_target_properties(gufo-ace PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-ace PRIVATE gufo_ace_step)
  gufo_stage_hip_runtime(gufo-ace)
endif()

if (BUILD_TESTING)
  # Prompts and Qwen3 tokens against the torch oracle (ft/ace-step/oracle):
  # GUFO_ACE_ORACLE = oracle directory, GUFO_ACE_TOKENIZER = tokenizer.json.
  # Skipped (77) without them.
  add_executable(ace_step_tokens_oracle_test
    ${PROJECT_SOURCE_DIR}/tests/models/ace_step/tokens_oracle_test.cpp)
  target_link_libraries(ace_step_tokens_oracle_test PRIVATE gufo_ace_step)
  add_test(NAME ace_step_tokens_oracle_test
    COMMAND ace_step_tokens_oracle_test)
  set_tests_properties(ace_step_tokens_oracle_test PROPERTIES
    LABELS "cpu;models;ace_step;oracle" SKIP_RETURN_CODE 77)
endif()
