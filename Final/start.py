import evaluate
import numpy as np
import concurrent.futures
#import main

def go(my_iterator):
    results = []
    for rho in rho_list:
        for lambdaBB in lambdaBB_list:
            for lambdaCC in lambdaCC_list:
                print(f"starting _fold{my_iterator}_{rho}_{lambdaCC}_{lambdaBB}")
                #main.start(rho, lambdaBB, lambdaCC, t= 1000, target=f"_fold{my_iterator}", out=f"_fold{my_iterator}_{rho}_{lambdaCC}_{lambdaBB}")
                r5, r10, NDCG5, NDCG10 = evaluate.evaluate(mytest=f"_fold{my_iterator}", target=f"_fold{my_iterator}_{rho}_{lambdaCC}_{lambdaBB}")
                results.append([r5, r10, NDCG5, NDCG10, rho, lambdaBB, lambdaCC])

    for final in results:
        print(final)
    np.save(f"./results{my_iterator}.npy", np.array(results))


rho_list = [1000, 20000, 100000]
lambdaBB_list = [500, 1000]
lambdaCC_list = [1000, 10000, 30000]
folds = [0, 1, 2, 3, 4]

#with concurrent.futures.ProcessPoolExecutor() as executor:
#    executor.map(go, folds)

for fold in folds:
    go(fold)