# Makefile


RM=rm -f

CURL=curl
GZIP=gzip


LIBS=-lm

DATADIR=./data
MNIST_FILES= \
	$(DATADIR)/train-images-idx3-ubyte \
	$(DATADIR)/train-labels-idx1-ubyte \
	$(DATADIR)/t10k-images-idx3-ubyte \
	$(DATADIR)/t10k-labels-idx1-ubyte

all: test_serial

clean:
	-$(RM) ./mnist *.o

get_mnist:
	-mkdir ./data
	sudo dnf install pip -y
	sudo dnf install unzip -y
	pip3 install gdown
	gdown https://drive.google.com/uc?id=11ZiNnV3YtpZ7d9afHZg0rtDRrmhha-1E
	unzip MNIST_ORG.zip
	rm MNIST_ORG.zip
	mv data/t10k-images.idx3-ubyte data/t10k-images-idx3-ubyte
	mv data/t10k-labels.idx1-ubyte data/t10k-labels-idx1-ubyte
	mv data/train-images.idx3-ubyte data/train-images-idx3-ubyte
	mv data/train-labels.idx1-ubyte data/train-labels-idx1-ubyte


test_serial: ./cnn $(MNIST_FILES)
	./cnn $(MNIST_FILES)
./cnn: cnn.c
	gcc -o $@ $^ $(LIBS)

test_mpi: ./cnnmpi $(MNIST_FILES)
	mpirun -np 8 ./cnnmpi $(MNIST_FILES)
./cnnmpi: cnnmpi.c
	mpicc -o $@ $^ $(LIBS)

test_cuda: ./CUDAcnn $(MNIST_FILES)
	./CUDAcnn $(MNIST_FILES)
./CUDAcnn: CUDAcnn.cu
	nvcc -o $@ $^ $(LIBS)