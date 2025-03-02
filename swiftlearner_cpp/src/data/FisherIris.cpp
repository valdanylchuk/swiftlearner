#include "data/FisherIris.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

namespace FisherIris {

std::pair<std::vector<std::pair<int, vec>>, std::vector<std::pair<int, vec>>>
loadData(const std::string& filename, std::optional<long> randomSeed) {
    std::vector<std::pair<int, vec>> data;

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int itemClass;
        vec parameters;

        // Extract itemClass
        std::getline(ss, value, ',');
        itemClass = std::stoi(value);

        // Extract parameters
        while (std::getline(ss, value, ',')) {
            parameters.push_back(std::stof(value));
        }

        data.push_back({itemClass, parameters});
    }

    // Shuffle the data
    std::mt19937 generator(randomSeed.value_or(std::random_device{}()));
    std::shuffle(data.begin(), data.end(), generator);

    // Split into training and test sets (2/3 training, 1/3 test)
    size_t trainingSize = data.size() * 2 / 3;
    std::vector<std::pair<int, vec>> trainingSet(data.begin(), data.begin() + trainingSize);
    std::vector<std::pair<int, vec>> testSet(data.begin() + trainingSize, data.end());

    return {trainingSet, testSet};
}

} // namespace FisherIris