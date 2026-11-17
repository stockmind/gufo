#include <filesystem>
#include <gufo/ext/music.hpp>
#include <gufo/ext/registry.hpp>
#include <string>
#include <string_view>

#include "src/models/ace_step/pipeline.hpp"

namespace gufo::models::ace_step {

bool ProbeAceStep(std::string_view model_root) {
  ModelPaths paths;
  std::string error;
  return ModelPaths::FromRoot(std::filesystem::path(model_root), &paths,
                              &error);
}

bool ResolveAceStepArtifacts(std::string_view model_root, std::string* error) {
  ModelPaths paths;
  if (!ModelPaths::FromRoot(std::filesystem::path(model_root), &paths,
                            error)) {
    if (error != nullptr && error->empty())
      *error = "not an ACE-Step 1.5 directory: " + std::string(model_root);
    return false;
  }
  return true;
}

}  // namespace gufo::models::ace_step

void ace_step_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::ace_step;
  registry.AddMusicProvider({
      .id = "ace-step",
      .probe = &ProbeAceStep,
      .resolve = &ResolveAceStepArtifacts,
      .create_with_options = nullptr,
  });
}
