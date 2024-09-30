from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label


# 假设有一些样本数据
data = [(1, 0), (2, 1), (3, 0), (4, 1), (5, 1)]
labels = [0, 1, 1, 2, 0]

# 创建 Dataset 和 DataLoader
dataset = MyDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

dataloader = iter(dataloader)
print(next(dataloader))  # [[tensor([5, 3]), tensor([1, 0])], tensor([0, 1])]
