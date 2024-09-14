import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _mnist(name):
    with open(f"downloads/{name}", "rb") as f:
        return torch.Tensor(list(f.read()))

def mnist(device=None):
  return _mnist("X_train.gunzip")[0x10:].reshape(-1,1,28,28).to(device), \
            _mnist("Y_train.gunzip")[8:].to(device), \
            _mnist("X_test.gunzip")[0x10:].reshape(-1,1,28,28).to(device), \
            _mnist("Y_test.gunzip")[8:].to(device)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist(device="cuda")
    Y_train, Y_test = Y_train.long(), Y_test.long()

    model = nn.Sequential(
                nn.Conv2d(1, 32, 5), nn.ReLU(),
                nn.Conv2d(32, 32, 5), nn.ReLU(),
                # nn.BatchNorm2d(32), 
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(32, 64, 3), nn.ReLU(),
                nn.Conv2d(64, 64, 3), nn.ReLU(),
                # nn.BatchNorm2d(64), 
                nn.MaxPool2d((2, 2)),
                nn.Flatten(1), nn.Linear(576, 10)).cuda()
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    test_acc = float("nan")
    for i in range(70):
        opt.zero_grad()
        samples = torch.randint(0, X_train.shape[0], (512,))
        loss = loss_fn(model(X_train[samples]), Y_train[samples])
        loss.backward()
        opt.step()
        if i%10 == 9: test_acc = (torch.argmax(model(X_test), dim=1) == Y_test).float().mean() * 100
        print(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")
