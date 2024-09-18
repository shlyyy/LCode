import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

batch_size = 4
root = "./cache"

my_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5],  # mean=[0.485, 0.456, 0.406]
                                                        std=[0.5])])  # std=[0.229, 0.224, 0.225]

train_dataset = torchvision.datasets.MNIST(root=root,
                                           train=True,
                                           transform=my_transform,
                                           download=True)

val_dataset = torchvision.datasets.MNIST(root=root,
                                         train=False,
                                         transform=my_transform,
                                         download=True)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

print(len(train_dataset))
print(len(train_loader))

iterator = iter(train_loader)
image, label = next(iterator)
print(image.shape)
print(label)

for i in range(batch_size):
    plt.subplot(1, batch_size, i + 1)
    plt.title(label[i].item())
    plt.axis("off")
    plt.imshow(image[i].permute(1, 2, 0))

plt.show()
