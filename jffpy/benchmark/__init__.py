import numpy as np

def eval_rank(results, filenames, qfilenames, shapes, n = 5):
    # How many queries to evaluate
    all_I, all_D, times = results
    j = 0
    hit = 0
    for D, I, time in zip(all_D, all_I, times):
        # For every individual

        k = 16
        result = np.repeat(np.arange(len(shapes)), shapes)
        appereances = {}
        for i in range(I.shape[0]):
            # For all the features in the individual

            idxs = I[i]
            for idx in idxs:
                # For every nearest neighbor

                appereances[result[idx]] = appereances.get(result[idx], 0) + 1
        
        
        # Sort the appereances and select the n best ones 
        appereances = sorted(appereances.items(), key=lambda x: x[1], reverse=True)
        appereances = appereances[:n]
        print(f"{qfilenames[j]} [", end="")
        for idx, count in appereances:
            print(f"({filenames[idx]}, {count})", end=", ")
            if filenames[idx].split("_")[0].strip() == qfilenames[j].split("_")[0]:
                hit += 1
        print(f"]")

        j += 1

    print(f"Hit rate: {hit / len(all_D) * 100 :.2f}%")        

def pick_best(results, metric = "frequency"):
    if metric == "frequency":
        pass
    elif metric == "distance":
        pass
    else:
        pass