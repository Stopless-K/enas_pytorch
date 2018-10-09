import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from IPython import embed

def make_cifar10_dataloader(root='.',train_size = 1.0,seed= 0, batch_size = 32,
                            download = True, shuffle_test =False, num_workers = 2
                            ):

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    trainset = torchvision.datasets.CIFAR10(root='{}/data'.format(root),
                                            train=True, download=download, transform=transform_train
                                            )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    testset = torchvision.datasets.CIFAR10(root='{}/data'.format(root),
                                            train=False, download=download, transform=transform_test
                                            )

    return _make_loaders(trainset, testset,
                         train_size, seed, batch_size, num_workers, shuffle_test)


def _make_loaders(trainset, testset, train_size, seed , batch_size,
                  num_workers, shuffle_test):

    #this func is copied from https://github.com/bkj/ripenet
    if train_size < 1:
        train_inds, val_inds = train_test_split(np.arange(len(trainset)), train_size=train_size, random_state=seed)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, num_workers=num_workers,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_inds),
        )

        valloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, num_workers=num_workers,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_inds),
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, num_workers=num_workers,
            shuffle=True,
        )

        valloader = None

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, num_workers=num_workers,
        shuffle=shuffle_test,
    )

    return {
        "train": trainloader,
        "test": testloader,
        "val": valloader,
    }



class Provider():

    def __init__(self, loader):

        self.batches_per_epoch = len(loader)
        self.epoch_batch =       0
        self.progress =          0
        self.epoch =             0

        self.loader = loader
        self._provider = self._make_provider()

    def _make_provider(self):

        while True:
            self.epoch_batch = 0
            for data, target in self.loader:
                yield  data, target

                self.epoch_batch += 1
                self.progress = self.epoch + (self.epoch_batch / self.batches_per_epoch)

            self.epoch +=1

    def __next__(self):

        return next(self._provider)




