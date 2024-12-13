#include "jff.hpp"

typedef ParentedFeature<float> feature;

int main()
{
    std::string galleryPath = "C:/Users/jfcmp/Documentos/Griaule/data/teste2";
    std::string queryPath = "C:/Users/jfcmp/Documentos/Griaule/data/teste1/queries.npy";

    auto [galleryIndividuals, gallery] = loadIndividuals(galleryPath, true);

    std::vector<feature> queries = loadNpy<feature>(queryPath, true);

    return 0;
}