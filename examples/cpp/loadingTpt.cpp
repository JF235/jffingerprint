#include "jff.hpp"

typedef ParentedFeature<float> feature;

int main()
{
    std::string galleryPath = "C:/Users/jfcmp/Documentos/Griaule/data/tse1k_flat";
    std::string queryPath = "C:/Users/jfcmp/Documentos/Griaule/data/teste1/b101-9_l.tpt";

    auto [galleryIndividuals, gallery] = loadIndividuals(galleryPath, true);

    std::vector<feature> queries = loadTpt<feature>(queryPath, true);

    // Print features ids for each individual
    for (const auto& individual : galleryIndividuals)
    {
        std::cout << "Individual " << individual->id << " has " << individual->features.size() << " features" << std::endl;
        for (const auto& feature : individual->features)
        {
            std::cout << feature << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}