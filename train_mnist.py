import torch
import torch.nn as nn
import torch.optim as optim


def _mnist(name):
    with open(f"downloads/{name}", "rb") as f:
        return torch.Tensor(list(f.read()))

def mnist(device=None):
  return _mnist("X_train.gunzip")[0x10:].reshape(-1,1,28,28).to(device), \
            _mnist("Y_train.gunzip")[8:].to(device), \
            _mnist("X_test.gunzip")[0x10:].reshape(-1,1,28,28).to(device), \
            _mnist("Y_test.gunzip")[8:].to(device)

def model_params(model) -> bytes:
    import struct

    params = b""
    for k, v in model.named_parameters():
        print(k, v.shape)
        params += b"".join([struct.pack("f", v) for v in v.flatten().tolist()])
        print(len(params))
    return params

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    X_train, Y_train, X_test, Y_test = mnist(device)
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
                nn.Flatten(1), nn.Linear(576, 10)).to(device)

    opt = optim.AdamW(model.parameters())


    # assumes batch input
    def cross_entropy_loss(probs, y):
        B, _ = probs.shape
        batch_idxs = torch.arange(B)
        losses = -torch.log(probs[batch_idxs, y[batch_idxs]])
        return torch.sum(losses) / B
    
    with open("untrained_params.bin", "wb") as f:
        f.write(model_params(model))

    test_acc = float("nan")
    for i in range(70):
        samples = torch.randint(0, X_train.shape[0], (4,))
        loss = cross_entropy_loss(torch.nn.functional.softmax(model(X_train[samples]), dim=1), Y_train[samples])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i%10 == 9: test_acc = (torch.argmax(model(X_test), dim=1) == Y_test).float().mean() * 100
        print(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

    with open("params.bin", "wb") as f:
        f.write(model_params(model))
