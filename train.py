import torch
import torchvision
from model import Model


batch_size_train = 64
batch_size_test = 64
num_feats = 100
num_classes = 10
max_relative_distance = 28 * 28
lr = 0.01
log_interval = 10
n_epochs = 10
nheads = 6


def test(epoch, network, criterion, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    #test_losses.append(test_loss)
    print('\nTest set (e{}): Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))


def train(epoch, network, optimizer, criterion, train_loader):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(output)
            print(target)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # train_losses.append(loss.item())
            # train_counter.append(
            #  (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            # torch.save(network.state_dict(), '/results/model.pth')
            # torch.save(optimizer.state_dict(), '/results/optimizer.pth')


if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True, num_workers=3)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True, num_workers=3)

    network = Model(num_feats=num_feats, num_classes=num_classes, max_relative_distance=max_relative_distance, nhead=nheads, hidden_mlp_size=128)
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)

    for epoch in range(1, n_epochs + 1):
        train(epoch, network, optimizer, criterion, train_loader)
        print('*' * 80)
        test(epoch, network, criterion, test_loader)
