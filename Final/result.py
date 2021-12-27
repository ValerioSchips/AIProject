import numpy as np

result = np.load(f"./results3.npy")

result = np.zeros((result.shape[0], result.shape[1], 5))


for x in range(0, 5):
    result[:, :, x] += np.load(f"./results{x}.npy")

mean = np.mean(result, axis=2)
print("Format [r5, r10, NDCG5, NDCG10, rho, lambdaBB, lambdaCC]")
print("Mean")
for x in mean:
    print(x)

print("Standrd deviation")
for x in np.std(result, axis=2):
    print(x)


for x in range(0, 5):
    print(np.load(f"./results{x}.npy")[0, 0:4])