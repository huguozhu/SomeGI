#pragma once
#include "scene.h"
#include <filesystem>
#include <string>

namespace somegi {

bool loadGltf(const std::filesystem::path& path, SceneCpu& outScene, std::string& outErr);

}
