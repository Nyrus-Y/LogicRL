import pickle


filename = "../../datasets/max_score_compact_500.pkl"

with open(filename, "rb") as f:
    dataset = pickle.load(f)
print(len(dataset))

episode0 = dataset[0]
print(len(episode0))
