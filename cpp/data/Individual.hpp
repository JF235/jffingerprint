#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP

#include <vector>
#include <cmath>
#include <string>
#include <iostream>

template <typename F>
class Individual {
public:
    /**
     * @brief Default constructor that initializes an empty Individual.
     */
    Individual() : id(nextId++), mean(), stddev() {}

    /**
     * @brief Adds a feature to the Individual.
     * @param featureId ID of the feature to be added.
     */
    void addFeature(uint32_t featureId) {
        features.push_back(featureId);
    }

    /**
     * @brief Calculates the mean feature of the Individual.
     * @param features Vector containing all features.
     */
    void calculateMean(const std::vector<F>& features) {
        if (features.empty()) return;

        size_t featureSize = features[0].size();
        std::vector<float> meanValues(featureSize, 0);

        for (const auto& feature : features) {
            for (size_t i = 0; i < featureSize; ++i) {
                meanValues[i] += feature[i];
            }
        }

        for (size_t i = 0; i < featureSize; ++i) {
            meanValues[i] /= features.size();
        }

        mean = F(0, meanValues);
    }

    /**
     * @brief Calculates the standard deviation feature of the Individual.
     * @param features Vector containing all features.
     */
    void calculateStd(const std::vector<F>& features) {
        if (features.empty()) return;

        size_t featureSize = features[0].size();
        std::vector<float> stdValues(featureSize, 0);

        for (const auto& feature : features) {
            for (size_t i = 0; i < featureSize; ++i) {
                float diff = feature[i] - mean[i];
                stdValues[i] += diff * diff;
            }
        }

        for (size_t i = 0; i < featureSize; ++i) {
            stdValues[i] = std::sqrt(stdValues[i] / features.size());
        }

        stddev = F(0, stdValues);
    }

    void print() const {
        std::cout << "Individual: " << name << "\n";
        std::cout << "ID: " << id << "\n";

        // Calculate the mean of the means
        float meanOfMeans = 0;
        for (const auto& val : mean.values) {
            meanOfMeans += val;
        }
        meanOfMeans /= mean.values.size();

        // Calculate the mean of the standard deviations
        float meanOfStds = 0;
        for (const auto& val : stddev.values) {
            meanOfStds += val;
        }
        meanOfStds /= stddev.values.size();

        // Increase the precision of the output
        std::cout.precision(10);
        std::cout << "Mean of Means: " << meanOfMeans << "\n";
        std::cout << "Mean of Stds: " << meanOfStds << "\n";
    }

    void printInline() const {
        std::cout << name << " (ID: " << id << ") ";

        // Calculate the mean of the means
        float meanOfMeans = 0;
        for (const auto& val : mean.values) {
            meanOfMeans += val;
        }
        meanOfMeans /= mean.values.size();

        // Calculate the mean of the standard deviations
        float meanOfStds = 0;
        for (const auto& val : stddev.values) {
            meanOfStds += val;
        }
        meanOfStds /= stddev.values.size();

        // Increase the precision of the output
        std::cout.precision(10);
        std::cout << meanOfMeans << "; ";
        std::cout << meanOfStds << "; ";
    }

    void printLong() const {
        // Print 8 decimal places
        std::cout.precision(9);
        std::cout << std::fixed;
        // Print all elements
        std::cout << "Individual: " << name << "\n";
        std::cout << "ID: " << id << "\n";
        std::cout << "Features: ";
        for (uint32_t featureId : features) {
            std::cout << featureId << " ";
        }
        std::cout << "\n";

        std::cout << "Mean: ";
        for (const auto& val : mean.values) {
            std::cout << val << " ";
        }
        std::cout << "\n";

        std::cout << "Std: ";
        for (const auto& val : stddev.values) {
            std::cout << val << " ";
        }
        std::cout << "\n\n";
    }

    uint32_t getId() const {
        return id;
    }

    uint32_t id;                    ///< Unique identifier of the Individual
    std::vector<uint32_t> features; ///< List of feature IDs associated with the Individual
    F mean;    ///< Mean feature
    F stddev;  ///< Standard deviation feature
    std::string name;               ///< Name of the Individual

private:
    static uint32_t nextId;         ///< Next unique identifier
};

// Initialize the static member
template <typename F>
uint32_t Individual<F>::nextId = 1;

#endif // INDIVIDUAL_HPP