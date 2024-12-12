#ifndef SHIFT_SEQUENTIAL_SEARCHER_HPP
#define SHIFT_SEQUENTIAL_SEARCHER_HPP

#include "SequentialSearcher.hpp"

/**
 * @brief A class for performing sequential k-nearest neighbors search, with shift by the mean and
 * scaling by std for each individual.
 *
 * @tparam T The type of the objects stored in dataObjects.
 * @tparam DistanceFunc The type of the distance function.
 */
template <typename T, typename DistanceFunc>
class ShiftSequentialSearcher : public SequentialSearcher<T, DistanceFunc>
{
public:
    /**
     * @brief Constructs a ShiftSequentialSearcher with the given distance function.
     *
     * @param distFunc The distance function to evaluate distance between objects.
     */
    ShiftSequentialSearcher(DistanceFunc &distFunc) : SequentialSearcher<T, DistanceFunc>(distFunc) {}

    /**
     * @brief Shifts and scales the query object based on the representative individual's mean and std.
     *
     * @param query The query object.
     * @param representative The representative individual.
     * @return T The shifted and scaled query object.
     */
    static T shift(const T &query, const Individual<float> *representative)
    {
        T shiftQuery(query.size());
        T mean = representative->mean;
        T std = representative->std;
        for (size_t i = 0; i < query.size(); i++)
        {
            shiftQuery[i] = mean[i] + query[i] * std[i];
        }
        return shiftQuery;
    }

    /**
     * @brief Shifts and scales all features in the vector based on their representative individual's mean and std.
     *
     * @param features The vector of features to be shifted and scaled.
     */
    static void shiftAll(std::vector<T> &features)
    {
        for (auto &f : features)
        {
            T mean = f.representative->mean;
            T std = f.representative->std;
            for (size_t i = 0; i < f.size(); i++)
            {
                f[i] = mean[i] + f[i] * std[i];
            }
        }
    }

    /**
     * @brief Performs k-nearest neighbors search.
     *
     * @param query The query object.
     * @param k The number of nearest neighbors to find.
     * @return NNList<T> The list of k-nearest neighbors.
     */
    NNList<T> knn(const T &query, size_t k) const override
    {
        NNList<T> nnList(k);

        // Sequentially calculate the distance between the query object and all objects in dataObjects
        // Takes O(n) distance calculations
        for (const auto &obj : this->dataObjects)
        {
            T shiftQuery = shift(query, obj.representative);
            double dist = this->distanceFunc(shiftQuery, obj);

            nnList.insert(obj, dist);
        }

        return nnList;
    }
};

#endif // SHIFT_SEQUENTIAL_SEARCHER_HPP