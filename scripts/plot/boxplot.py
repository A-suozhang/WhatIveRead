
import matplotlib.pyplot as plt
import numpy as np
import pickle
data = {}
for name in ["gates"]:
    with open("./predict_results/{}.pkl".format(name), 'rb') as f:
        data[name] = pickle.load(f)
for name in ["gcn","lstm","mlp"]:
    with open("./predict_results_regression/{}.pkl".format(name), 'rb') as f:
        data[name] = pickle.load(f)

num_repeat = 200
for_plot = np.zeros([num_repeat,4])
K = 1
N = 50
min_ranks = np.array([])
min_ranks_global = np.array([])
for i in range(num_repeat):
    for idx,name in enumerate(data.keys()):
        
        true_scores = np.array(data[name][0])
        predict_scores = np.array(data[name][1])

        true_inds = np.argsort(true_scores)[::-1]
        reorder_true_scores = true_scores[true_inds.astype(int)]

        # Acquire Same Random Scores
        np.random.seed(i)
        random_ind = np.random.randint(0,len(predict_scores),size=K*N)
        random_scores = true_scores[random_ind]

        sort_random_inds = np.argsort(random_scores)[::-1]
        reorder_random_scores = random_scores[sort_random_inds.astype(int)]
        
        sampled_best_ind = np.argsort(predict_scores[random_ind])[-K:]
        sampled_ranks = []
        sampled_ranks_global = []
        for j in sampled_best_ind:
            sampled_ranks = np.concatenate([sampled_ranks,\
                        np.argwhere(reorder_random_scores == random_scores[j])[0]])
            sampled_ranks_global = np.concatenate([sampled_ranks_global,\
                        np.argwhere(reorder_true_scores == random_scores[j])[0]])
        min_ranks = np.min(sampled_ranks)
        min_ranks_global = np.min(sampled_ranks_global)
        for_plot[i,idx] = np.min(min_ranks_global)
# print(for_plot[1:].reshape(-1,4))

ax,fig = plt.subplots()
# print(for_plot)
fig.boxplot(for_plot,labels=data.keys(),showfliers=False)
