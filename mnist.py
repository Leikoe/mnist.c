import hashlib, urllib, pathlib, tempfile, gzip, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Union

# modified snippet from https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py#L266
def fetch(url:str, name:Optional[Union[pathlib.Path, str]]=None, subdir:Optional[str]=None, gunzip:bool=False) -> pathlib.Path:
    if url.startswith(("/", ".")):
        return pathlib.Path(url)
    if name is not None and (isinstance(name, pathlib.Path) or '/' in name):
        fp = pathlib.Path(name)
    else:
        fp = pathlib.Path("./") / "downloads" / (subdir or "") / \
            ((name or hashlib.md5(url.encode('utf-8')).hexdigest()) + (".gunzip" if gunzip else ""))
    if not fp.is_file():
        with urllib.request.urlopen(url, timeout=10) as r:
            assert r.status == 200
            length = int(r.headers.get('content-length', 0)) if not gunzip else None
            (path := fp.parent).mkdir(parents=True, exist_ok=True)
            readfile = gzip.GzipFile(fileobj=r) if gunzip else r
            with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
                while chunk := readfile.read(16384):
                    print("got", f.write(chunk), "B")
                f.close()
            if length and (file_size:=os.stat(f.name).st_size) < length:
                raise RuntimeError(f"fetch size incomplete, {file_size} < {length}")
            pathlib.Path(f.name).rename(fp)
    return fp

def _mnist(file, name):
    with open(fetch("https://storage.googleapis.com/cvdf-datasets/mnist/"+file, name, gunzip=True), "rb") as in_file:
        return torch.Tensor(list(in_file.read()))
def mnist(device=None):
  return _mnist("train-images-idx3-ubyte.gz", "train_x")[0x10:].reshape(-1,1,28,28).to(device), \
            _mnist("train-labels-idx1-ubyte.gz", "train_y")[8:].to(device), \
            _mnist("t10k-images-idx3-ubyte.gz", "test_x")[0x10:].reshape(-1,1,28,28).to(device), \
            _mnist("t10k-labels-idx1-ubyte.gz", "test_y")[8:].to(device)


X_train, Y_train, X_test, Y_test = mnist(device="cuda")
Y_train, Y_test = Y_train.to(torch.long), Y_test.to(torch.long)

# import matplotlib.pyplot as plt
# print(Y_train[0])
# plt.imshow(X_train[0][0])
# plt.show()

model = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(),
            nn.Conv2d(32, 32, 5), nn.ReLU(),
            nn.BatchNorm2d(32), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.BatchNorm2d(64), nn.MaxPool2d((2, 2)),
            nn.Flatten(1), nn.Linear(576, 10))
model.to("cuda")
opt = optim.Adam(model.parameters())

test_acc = float("nan")
for i in range(70):
    opt.zero_grad()
    samples = torch.randint(0, X_train.shape[0], (512,)).cuda()
    # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
    loss = F.nll_loss(model(X_train[samples]), Y_train[samples])
    loss.backward()
    opt.step()
    if i%10 == 9: test_acc = (torch.argmax(model(X_test), dim=1) == Y_test).to(torch.float).mean() * 100
    print(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")
