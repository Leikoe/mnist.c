# Mnist 99% from scratch in c (and hopefully cuda too)

> Note: This code is heavily inspired by llm.c's, a lot of code was copied from there and modified.


# How to use

```shell
python download_dataset.py  # downloads the mnist dataset to downloads/
clang train_mnist.c -o train_mnist  # run the traning
```

optionally compile with openmp support
```shell
# this is for apple silicon macs, "brew install libomp" if you don't already have it
clang -DOMP -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp train_mnist.c -o train_mnist
```
