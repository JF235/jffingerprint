#ifndef LINALG_HPP
#define LINALG_HPP

#include <vector>
#include <cmath>
#include <numeric>

/**
 * @brief Namespace for linear algebra operations.
 */
namespace LinAlg {

/**
 * @brief Computes the 2-norm of a feature vector.
 * 
 * @tparam F The type of the feature vector.
 * @param feature The feature vector.
 * @return float The 2-norm of the feature vector.
 */
template <typename F>
float norm(const F &feature)
{
    return std::sqrt(std::inner_product(feature.begin(), feature.end(), feature.begin(), 0.0f));
}

/**
 * @brief Multiplies a feature vector by a scalar.
 * 
 * @tparam F The type of the feature vector.
 * @tparam numT The numeric type of the elements in the feature vector and the scalar.
 * @param feature The feature vector.
 * @param scalar The scalar value.
 * @return F The resulting feature vector after scalar multiplication.
 */
template <typename F, typename numT>
F scalarProd(const F &feature, numT scalar)
{
    F result(feature.size());

    for (size_t i = 0; i < feature.size(); ++i)
    {
        result[i] = feature[i] * scalar;
    }

    return result;
}

/**
 * @brief Sums two feature vectors element-wise.
 * 
 * @tparam F The type of the feature vectors.
 * @param feature1 The first feature vector.
 * @param feature2 The second feature vector.
 * @return F The resulting feature vector after element-wise summation.
 */
template <typename F>
F vecSum(const F &feature1, const F &feature2)
{
    if (feature1.size() != feature2.size())
    {
        throw std::invalid_argument("Vectors must be of the same size (" + std::to_string(feature1.size()) + " != " + std::to_string(feature2.size()) + ")");
    }

    F result(feature1.size());
    for (size_t i = 0; i < feature1.size(); ++i)
    {
        result[i] = feature1[i] + feature2[i];
    }
    return result;
}

/**
 * @brief Computes the dot product of two feature vectors.
 * 
 * @tparam F The type of the feature vectors.
 * @tparam numT The numeric type of the elements in the feature vectors.
 * @param feature1 The first feature vector.
 * @param feature2 The second feature vector.
 * @return numT The dot product of the two feature vectors.
 */
template <typename F, typename numT>
numT dotProd(const F &feature1, const F &feature2)
{
    if (feature1.size() != feature2.size())
    {
        throw std::invalid_argument("Vectors must be of the same size (" + std::to_string(feature1.size()) + " != " + std::to_string(feature2.size()) + ")");
    }

    return std::inner_product(feature1.begin(), feature1.end(), feature2.begin(), numT(0));
}

/**
 * @brief Adds a constant value to all components of a feature vector.
 * 
 * @tparam F The type of the feature vector.
 * @param feature The feature vector.
 * @param value The constant value to be added.
 * @return F The resulting feature vector after adding the constant value.
 */
template <typename F>
F shiftVec(const F &feature, float value)
{
    F result(feature.size());
    for (size_t i = 0; i < feature.size(); ++i)
    {
        result[i] = feature[i] + value;
    }
    return result;
}

/**
 * @brief Computes the pairwise product of two feature vectors.
 * 
 * @tparam F The type of the feature vectors.
 * @param feature1 The first feature vector.
 * @param feature2 The second feature vector.
 * @return F The resulting feature vector after pairwise multiplication.
 */
template <typename F>
F vecPairwiseProd(const F &feature1, const F &feature2)
{
    if (feature1.size() != feature2.size())
    {
        throw std::invalid_argument("Vectors must be of the same size (" + std::to_string(feature1.size()) + " != " + std::to_string(feature2.size()) + ")");
    }

    F result(feature1.size());
    for (size_t i = 0; i < feature1.size(); ++i)
    {
        result[i] = feature1[i] * feature2[i];
    }
    return result;
}

} // namespace LinAlg

#endif // LINALG_HPP