from dataset_helper import Dataset

dataset = Dataset('./dataset/dataset.csv', 250, 48, 350)
dataset.load_dataset()

for i in range(1000):
    batch = dataset.next_batch()
    print(str(dataset.start)+ ' ' +str(dataset.end))