# ACE-Step 1.5 XL music extension (ported from gufo-studio-windows).

gufo_extension_target(
  NAME gufo_ace_step
  SOURCES
    tokenizer.cpp
    prompt.cpp
    preview.cpp
    sampler.cpp
    audio_io.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    gufo_qwen_text
    ICU::i18n
)

if(TARGET gufo_ace_step)
  target_include_directories(gufo_ace_step PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

gufo_extension_target(
  NAME gufo_ace_step_hip
  SOURCES
    gemm.cpp
    pipeline.cpp
    parallel_read.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_ace_step
    hip::host
    roc::hipblaslt
)

if(TARGET gufo_ace_step_hip)
  gufo_extension_hip_sources(gufo_ace_step_hip
    SOURCES
      weights.hip
      kernels.hip
      attention.hip
      model.hip
      vae.hip
  )
  target_compile_definitions(gufo_ace_step_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_ace_step_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()
