import evaluate
import numpy as np
import concurrent.futures
import main

def go(my_iterator, thresh):
    results = []
    for rho in rho_list:
        for lambdaBB in lambdaBB_list:
            for lambdaCC in lambdaCC_list:
                print(f"starting test{my_iterator}_{rho}_{lambdaCC}_{lambdaBB}")
                main.start(rho, lambdaBB, lambdaCC, t=thresh, target=f"_old_data{my_iterator}", out=f"_old_data{my_iterator}_{rho}_{lambdaCC}_{lambdaBB}")
                r5, r10, NDCG5, NDCG10 = evaluate.evaluate(mytest=f"_old_data{my_iterator}", target=f"_old_data{my_iterator}_{rho}_{lambdaCC}_{lambdaBB}")
                results.append([r5, r10, NDCG5, NDCG10, rho, lambdaBB, lambdaCC])

    for final in results:
        print(final)
    np.save(f"./results_old_data{my_iterator}.npy", np.array(results))


rho_list = [1000, 20000, 100000]
lambdaBB_list = [500, 1000]
lambdaCC_list = [1000, 5000, 15000]
tests = [[1, 25], [2, 45], [3, 40], [4, 40]]

#with concurrent.futures.ProcessPoolExecutor() as executor:
#    executor.map(go, tests)

for test in tests:
    go(test[0], test[1])