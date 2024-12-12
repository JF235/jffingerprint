#ifndef PARENTEDFEATURE_HPP
#define PARENTEDFEATURE_HPP

#include "Feature.hpp"

template <typename NumT>
class Individual;

/**
 * @brief Class representing a feature with a pointer to the representative Individual (parent).
 * 
 * A feature is a vector of values that can be used to represent an object.
 * 
 * @tparam NumT Type of the values of the feature.
 */
template <typename NumT>
class ParentedFeature : public Feature<NumT> {
public:
    /**
     * @brief Default constructor that initializes an empty Feature.
     */
    ParentedFeature() : Feature<NumT>(), representative(nullptr) {}

    /**
     * @brief Constructor that initializes a Feature with an ID and a vector of values.
     * 
     * If the ID is 0, it is not assigned.
     * 
     * @param id Unique identifier of the Feature.
     * @param vals Vector of values of the Feature.
     * @param rep Pointer to the representative Individual.
     */
    ParentedFeature(uint32_t id, const std::vector<NumT>& vals, Individual<NumT>* rep = nullptr) 
        : Feature<NumT>(id, vals), representative(rep) {
        if (representative) {
            representative->addFeature(this->id);
        }
    }

    /**
     * @brief Constructor that initializes a Feature with a vector of values.
     * 
     * The ID is automatically assigned.
     * 
     * @param vals Vector of values of the Feature.
     * @param rep Pointer to the representative Individual.
     */
    ParentedFeature(std::vector<NumT>& vals, Individual<NumT>* rep = nullptr) 
        : Feature<NumT>(vals), representative(rep) {
        if (representative) {
            representative->addFeature(this->id);
        }
    }

    /** 
     * @brief Other constructor for the case of rvalue reference
     * 
     * ID is automatically assigned.
     * 
     * @param vals rvalue reference to the vector of values of the Feature.
     * @param rep Pointer to the representative Individual.
     */
    ParentedFeature(std::vector<NumT>&& vals, Individual<NumT>* rep = nullptr) 
        : Feature<NumT>(std::move(vals)), representative(rep) {
        if (representative) {
            representative->addFeature(this->id);
        }
    }

    /**
     * @brief Constructor that initializes a Feature with a size.
     * 
     * The values are initialized to 0.
     * 
     * @param size Size of the Feature.
     * @param rep Pointer to the representative Individual.
     */
    ParentedFeature(size_t size, Individual<NumT>* rep = nullptr) 
        : Feature<NumT>(size), representative(rep) {
        if (representative) {
            representative->addFeature(this->id);
        }
    }

    /**
     * @brief Overload of the output operator to format the output as id:<idval>.
     * @param os Output stream.
     * @param f Feature object to be printed.
     * @return Output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const ParentedFeature& f) {
        os << "(id:" << f.id;
        if (f.representative != nullptr) {
            os << ", rep:" << f.representative->name << "[" << f.representative->id << "]";
        }
        os << ") ";
        return os;
    }

    Individual<NumT>* representative; ///< Pointer to the representative Individual
};

#endif // PARENTEDFEATURE_HPP