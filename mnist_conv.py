import torchvision
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

from conv2d import myConv2d
from maxpool2d import myMaxPool2d
from torch.utils.data import DataLoader, Dataset

class DigitClassification(nn.Module):
    def __init__(self):
        super(DigitClassification, self).__init__()
        self.conv11 = myConv2d( 1, 16, [3,3], [1,1], 1)
        self.maxpool1 = myMaxPool2d([2,2], [2,2])
        self.conv12 = myConv2d( 16, 32, [3,3], [1,1], 1)
        self.maxpool2 = myMaxPool2d([2,2], [2,2])
        self.linear1 = nn.Linear(in_features=7*7*32, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv11(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)

        x = self.conv12(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = nn.functional.dropout(x, p=0.3)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        return x

def evaluation(model, criteria, batch_generator, cuda=True):
    print('In EVAL')
    # make sure you know the difference between trianing mode and eval model
    model.eval()
    total_loss = []
    for index, (batch, ref) in enumerate(train_batch_generator):
        print(f'is cuda: {cuda}')
        if cuda:
            batch = batch.cuda()
            ref = ref.cuda()

        model.zero_grad()
        hyp = model(batch)
        loss = criteria(hyp, ref)
        
        total_loss.append(loss.data.cpu().numpy())
    model.train()
    return np.mean(total_loss)

def accuracy(model, batch_generator):
    print('In Accuracy')
    model.eval()
    compared = []

    for index, (batch, ref) in enumerate(train_batch_generator):
        print(f'index: {index}')
        batch = batch.cuda()

        model.zero_grad()
        hyp = model(batch)
        hyp = torch.nn.functional.softmax(hyp, dim=1)
        hyp = torch.argmax(hyp, dim=1)
        temp = torch.eq(hyp.cpu(), ref)
        compared.extend(temp.data.tolist())

    model.train()
    return sum(compared)/len(compared)

def evaluation_package(model, criteria, dataloader, title):
    print('Eval Package')
    acc = accuracy(model, dataloader)
    loss = evaluation(model, criteria, dataloader)
    if title == "validation":
        print("{} loss: {}".format(title, loss))
    else: 
        print("====================================")
    print("{} acc: {}".format(title, acc))

def plot(data, refs, hyps, nrows=4, ncols=4):
    assert isinstance(data, torch.FloatTensor)
    assert isinstance(refs, torch.LongTensor)
    assert isinstance(hyps, torch.LongTensor)

#    data, refs, hyps = data[:nrows], refs[:nrows], hyps[:nrows]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    for i, (img, ref, hyp) in enumerate(zip(data, refs, hyps)):
        temp_handle = axes[int(i//ncols), int(i%ncols)]
        temp_handle.imshow(img.permute(1,2,0).data.numpy()[:,:,0])
        temp_handle.set_title("Numer: {}->{}".format(ref, hyp))
    #fig.show()
    plt.show()


if __name__ == '__main__':
    # Load data
    # this homework will not teach you how to create pytorch's dataset.
    # I have ensured that you do not need write any related code.
    dataset = torchvision.datasets.MNIST('mnist/', train=True, download=True, 
                                           transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    test_dataset = torchvision.datasets.MNIST('mnist/', train=False, download=True, 
                                          transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    # data visualization 
    train_batch_generator = DataLoader(train_dataset, batch_size=16, shuffle=True)
    data, labels = list(train_batch_generator)[0]
    plot(data, labels,  labels, 4, 4)

    # for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare dataloaders
    train_batch_generator = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_batch_generator = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_batch_generator = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # model training
    model = DigitClassification()
    # copy the model to GPU
    model.cuda()
    # switch to training mode
    model.train()

    # initialize loss function
    criteria = torch.nn.CrossEntropyLoss()
    # initialize optimization funciton
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # training, 1 epoch only
    for index, (batch, ref) in enumerate(train_batch_generator):
        print(f'[BATCH]: {index}')
        if index % 100 == 0:
            # evaluation training progress every 100 iteration
            evaluation_package(model, criteria, val_batch_generator, title="validation")

        if not model.training:
            model.train()

        batch = batch.cuda()
        ref = ref.cuda()

        model.zero_grad()
        optimizer.zero_grad()

        hyp = model(batch)

        loss = criteria(hyp, ref)
        # backpropagation
        loss.backward()
        # parameter update
        optimizer.step()

    evaluation_package(model, criteria, test_batch_generator, title="test")