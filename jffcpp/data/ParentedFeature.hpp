#ifndef PARENTEDFEATURE_HPP
#define PARENTEDFEATURE_HPP

#include "Feature.hpp"
#include "Individual.hpp"

/**
 * @brief Class representing a feature with a pointer to the representative Individual (parent).
 */
class ParentedFeature : public Feature {
public:
    ParentedFeature() : Feature(), representative(nullptr) {}

    ParentedFeature(uint32_t id, const std::vector<float>& vals, Individual<ParentedFeature>* rep = nullptr)
        : Feature(id, vals), representative(rep) {
        if (representative) {
            representative->addFeature(this->id);
        }
    }

    ParentedFeature(std::vector<float>& vals, Individual<ParentedFeature>* rep = nullptr)
        : Feature(vals), representative(rep) {
        if (representative) {
            representative->addFeature(this->id);
        }
    }

    ParentedFeature(std::vector<float>&& vals, Individual<ParentedFeature>* rep = nullptr)
        : Feature(std::move(vals)), representative(rep) {
        if (representative) {
            representative->addFeature(this->id);
        }
    }

    ParentedFeature(size_t size, Individual<ParentedFeature>* rep = nullptr)
        : Feature(size), representative(rep) {
        if (representative) {
            representative->addFeature(this->id);
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const ParentedFeature& f) {
        os << "(id:" << f.id;
        if (f.representative != nullptr) {
            os << ", rep:" << f.representative->name << "[" << f.representative->id << "]";
        }
        os << ") ";
        return os;
    }

    Individual<ParentedFeature>* representative; ///< Pointer to the representative Individual
};

#endif // PARENTEDFEATURE_HPP