#pragma once

#include <vector>
#include <utility> // For std::pair
#include <optional>

using vec = std::vector<float>;

namespace FisherIris {

std::pair<std::vector<std::pair<int, vec>>, std::vector<std::pair<int, vec>>>
loadData(const std::string& filename = "resources/FisherIris.csv", std::optional<long> randomSeed = std::nullopt);

}