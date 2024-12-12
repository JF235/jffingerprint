#include "jff.hpp"

typedef ParentedFeature<float> feature;
typedef EuclideanDistance<feature> euclidean;
typedef ShiftSequentialSearcher<feature, euclidean> shift_searcher;

int main()
{   

    // 1. Load
    std::string galleryPath = "C:/Users/jfcmp/Documentos/Griaule/data/teste2";
    std::string queryPath = "C:/Users/jfcmp/Documentos/Griaule/data/teste1/queries.npy";

    auto [galleryIndividuals, gallery] = loadIndividuals(galleryPath, true);

    // 2. Shift All
    shift_searcher::shiftAll(gallery);

    // Print all shifted features
    for (const auto &f : gallery)
    {
        std::cout << f << " | ";
        for (size_t i = 0; i < f.size(); i++)
        {
            std::cout << f[i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // 3. Build Index
    euclidean d;
    shift_searcher searcher(d);
    searcher.addAll(gallery);
    std::cout << "Added: " << searcher.size() << "\n\n";

    // 4. Load queries
    std::vector<feature> queries = loadNpy<feature>(queryPath, true);

    // 5. Perform the queries
    std::vector<NNList<feature>> results;
    for (auto &q : queries){
        NNList<feature> nnList = searcher.knn(q, 3);
        results.push_back(nnList);

        std::cout << "Query: " << q << "\n";
        std::cout << "Results: " << nnList << "\n\n";
    }
    
    // 6. Evaluate
    NNResult<feature> nnResult(results);

    std::cout << nnResult << "\n\n";

    auto best = nnResult.pickBest(2, "frequency");
    std::cout << "Best: ";
    for (auto &b : best){
        std::cout << b.first << " " << b.second << "; ";
    }
    std::cout << "\n\n";

    auto best2 = nnResult.pickBest(2, "distance");
    std::cout << "Best: ";
    for (auto &b : best2){
        std::cout << b.first << " " << b.second << "; ";
    }
    std::cout << "\n\n";

    return 0;
}