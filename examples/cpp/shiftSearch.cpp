#include "jff.hpp"

typedef ParentedFeature feature;
typedef EuclideanDistance<feature> euclidean;
typedef ShiftSequentialSearcher<feature, euclidean> shift_searcher;

int main()
{   

    // 1. Load
    std::string galleryPath = "C:/Users/jfcmp/Documentos/Griaule/data/teste2";
    std::string queryPath = "C:/Users/jfcmp/Documentos/Griaule/data/teste1/b101-9_l.tpt";

    auto [galleryIndividuals, gallery] = loadIndividuals(galleryPath, true);

    // 2. Shift All
    shift_searcher::shiftAll(gallery);

    std::cout << "Shifted features: " << gallery.size() << "\n";

    // 3. Build Index
    euclidean d;
    shift_searcher searcher(d);
    searcher.addAll(gallery);
    std::cout << "Added: " << searcher.size() << "\n\n";

    // 4. Load queries
    std::vector<feature> queries = loadFile<feature>(queryPath, true);

    // 5. Perform the queries, using
    std::vector<NNList<feature>> results;
    for (auto &q : queries){
        NNList<feature> nnList = searcher.knn(q, 5);
        results.push_back(nnList);

        std::cout << "Query: " << q << "\n";
        std::cout << "Results: " << nnList << "\n\n";
    }
    
    // 6. Evaluate
    NNResult<feature> nnResult(results);

    // std::cout << nnResult << "\n\n";

    auto best = nnResult.pickBest(2, "frequency");
    std::cout << "Best: ";
    for (auto &b : best){
        std::cout << b.first << " (" << galleryIndividuals[b.first-1]->name << ") " << b.second << "; ";
    }
    std::cout << "\n\n";

    auto best2 = nnResult.pickBest(2, "distance");
    std::cout << "Best: ";
    for (auto &b : best2){
        std::cout << b.first << " (" << galleryIndividuals[b.first-1]->name << ") " << b.second << "; ";
    }
    std::cout << "\n\n";

    return 0;
}