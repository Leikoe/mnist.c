# heavily modified from llm.c's Makefile: https://github.com/karpathy/llm.c/blob/master/Makefile
CC ?= clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
LDLIBS = -lm

OUTPUT_FILE = -o $@
TARGETS = train_mnist

all: $(TARGETS)

train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

clean:
	rm -f $(TARGETS)