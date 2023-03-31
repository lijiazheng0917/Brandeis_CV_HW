import torch
from torchvision import datasets, transforms

def get_loader(dataset, batch_size):
    if dataset =='lfw':
        train_dataset = datasets.LFWPeople(root='./datasets/',
                                    split='train',
                                    transform=transforms.ToTensor(),
                                    download=True)

        test_dataset = datasets.LFWPairs(root='./datasets/',
                                    split='test',
                                    transform=transforms.ToTensor(),
                                    download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)
        return train_loader, test_loader