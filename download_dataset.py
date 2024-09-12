import hashlib, pathlib, tempfile, gzip, os
from typing import Optional, Union
import urllib.request

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
    fetch("https://storage.googleapis.com/cvdf-datasets/mnist/"+file, name, gunzip=True)

def mnist(device=None):
  return _mnist("train-images-idx3-ubyte.gz", "X_train"), \
            _mnist("train-labels-idx1-ubyte.gz", "Y_train"), \
            _mnist("t10k-images-idx3-ubyte.gz", "X_test"), \
            _mnist("t10k-labels-idx1-ubyte.gz", "Y_test")

if __name__ == "__main__":
    mnist()