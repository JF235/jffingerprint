#ifndef LOADERS_HPP
#define LOADERS_HPP

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <ostream>
#include <chrono>
#include <memory>
#include "ParentedFeature.hpp"
#include "Individual.hpp"
#include "../dependencies/npy.hpp"

namespace fs = std::filesystem;

/**
 * @brief Loads data from a .npy file and converts it into a list of features.
 * 
 * This function reads a specified .npy file, extracts the data, and converts it into a list of features.
 * If log_info is true, information about the loading process will be logged.
 * 
 * @param filename The name of the .npy file to be loaded.
 * @param log_info If true, logs information about the loading process.
 * @return A vector of features extracted from the .npy file.
 * @tparam F Type of the features to be loaded.
 */
template <typename F>
std::vector<F> loadNpy(std::string filename, bool log_info)
{
    std::vector<F> dataFeatures;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> data;
    std::vector<unsigned long> shape;

    // Load the data objects from the .npy file
    npy::npy_data<float> d = npy::read_npy<float>(filename);

    data = std::move(d.data);
    shape = std::move(d.shape);

    if (log_info)
        std::cout << "Loaded .npy with shape: " << shape[0] << "x" << shape[1] << "\n";

    // Reserve space for dataFeatures to avoid multiple reallocations
    dataFeatures.reserve(shape[0]);
    
    // Go through the lines of matrix
    for (uint32_t i = 0; i < shape[0]; i++)
    {
        // Use iterators to avoid copying data to a new vector
        auto startIt = data.begin() + i * shape[1];
        auto endIt = startIt + shape[1];

        dataFeatures.emplace_back(std::vector<float>(startIt, endIt));
    }

    if (log_info)
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = (end - start) * 1000; // milliseconds
        std::cout << "Added " << dataFeatures.size() << " features\n";
        std::cout << "Time: " << duration.count() << " ms\n\n";
    }

    return dataFeatures;
}

/**
 * @brief Loads individuals and their features from a specified directory.
 * 
 * This function iterates through all files in the given directory path, 
 * loads individuals from files with a ".npy" extension, and extracts their features.
 * It also associates each feature with the individual it belongs to and vice versa.
 * Finally, it calculates the mean and standard deviation for each individual.
 * 
 * @param directoryPath The path to the directory containing the files.
 * @param log_info If true, logs information about the loading process.
 * @return A pair consisting of a vector of individual pointers and a vector of features.
 */
std::pair<std::vector<std::shared_ptr<Individual<float>>>, std::vector<ParentedFeature<float>>> loadIndividuals(const std::string &directoryPath, bool log_info)
{
    using feature = ParentedFeature<float>;
    std::vector<std::shared_ptr<Individual<float>>> individuals;
    std::vector<feature> allFeatures;

    auto start = std::chrono::high_resolution_clock::now();

    // Iterate through all files in the directory and load individuals from .npy files
    for (const auto &entry : fs::directory_iterator(directoryPath))
    {
        if (entry.path().extension() == ".npy")
        {
            // For each .npy file, create an individual
            auto individual = std::make_shared<Individual<float>>();
            individual->name = entry.path().filename().string();

            // Load all features from the file
            std::vector<feature> fileFeatures = loadNpy<feature>(entry.path().string(), false);

            for (auto &f : fileFeatures)
            {
                // Associate all features with the individual
                f.representative = individual.get();
                individual->addFeature(f.getId());
                allFeatures.push_back(f);
            }

            // Calculate the mean and std for the individual
            individual->calculateMean(fileFeatures);
            individual->calculateStd(fileFeatures);
    
            individuals.push_back(individual);
        }
    }

    if (log_info)
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = (end - start) * 1000; // milliseconds
        std::cout << "Loaded " << individuals.size() << " individuals\n";
        std::cout << "Added " << allFeatures.size() << " features\n";
        std::cout << "Time: " << duration.count() << " ms\n\n";
    }

    return std::make_pair(individuals, allFeatures);
}

#endif // LOADERS_HPP