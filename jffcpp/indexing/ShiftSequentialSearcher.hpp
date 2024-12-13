#ifndef SHIFT_SEQUENTIAL_SEARCHER_HPP
#define SHIFT_SEQUENTIAL_SEARCHER_HPP

#include "SequentialSearcher.hpp"
#include "../math/LinAlg.hpp"

/**
 * @brief A class for performing sequential k-nearest neighbors search, with shift by the mean and
 * scaling by std for each individual.
 *
 * @tparam F The type of the objects stored in dataObjects.
 * @tparam DistanceFunc The type of the distance function.
 */
template <typename F, typename DistanceFunc>
class ShiftSequentialSearcher : public SequentialSearcher<F, DistanceFunc>
{
public:
    /**
     * @brief Constructs a ShiftSequentialSearcher with the given distance function.
     *
     * @param distFunc The distance function to evaluate distance between objects.
     */
    ShiftSequentialSearcher(DistanceFunc &distFunc) : SequentialSearcher<F, DistanceFunc>(distFunc) {}

    /**
     * @brief Shifts and scales the query object based on the representative individual's mean and std.
     *
     * @param query The query object.
     * @param representative The representative individual.
     * @return F The shifted and scaled query object.
     */
    static F shift(F &feature, Individual<F> *representative)
    {

        F mean = representative->mean;
        F std = representative->stddev;
        F shifted_feature(feature.size());

        // f = f * std + mean
        for (size_t i = 0; i < feature.size(); ++i)
        {
            shifted_feature[i] = feature[i] * std[i] + mean[i];
        }

        shifted_feature.id = feature.id;
        shifted_feature.representative = representative;

        return shifted_feature;
    }

    /**
     * @brief Shifts and scales all features in the vector based on their representative individual's mean and std.
     *
     * @param features The vector of features to be shifted and scaled.
     */
    static void shiftAll(std::vector<F> &features)
    {
        for (size_t i = 0; i < features.size(); ++i)
        {
            features[i] = shift(features[i], features[i].representative);
        }
    }

    /**
     * @brief Performs k-nearest neighbors search.
     *
     * @param query The query object.
     * @param k The number of nearest neighbors to find.
     * @return NNList<F> The list of k-nearest neighbors.
     */
    NNList<F> knn(F &query, size_t k) const override
    {
        NNList<F> nnList(k);

        // Sequentially calculate the distance between the query object and all objects in dataObjects
        // Takes O(n) distance calculations
        for (const auto &obj : this->dataObjects)
        {
            F shiftQuery = shift(query, obj.representative);
            double dist = this->distanceFunc(shiftQuery, obj);

            nnList.insert(obj, dist);
        }

        return nnList;
    }
};

#endif // SHIFT_SEQUENTIAL_SEARCHER_HPP