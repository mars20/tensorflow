# Tensorflow Lite for RISCV

This is an experimental port of Tensorflow Lite for RISCV architecture
Grab a copy of tensorflow sources

```
$ git clone git@github.com:mars20/tensorflow.git
$ cd tensorflow/
$ git fetch origin riscv-tf:riscv-tf
$ git checkpoint riscv-tf

```
Set path to source files:

```
$ export TFLITE_RISCV_PATH=tensorflow/tensorflow/contrib/lite/experimental/riscv
$ export TFLITE_PATH=tensorflow/tensorflow/contrib/lite

```

#### Prerequiste
Install the dependencies by running the `download_dependencies.sh` script in tensorflow/contrib/lite/experimental/riscv/tools/make 

#### Build minimal RISC-V target 

```shell
$ export TFLITE_RISCV_PATH=tensorflow/tensorflow/contrib/lite/experimental/riscv
$ make -f $TFLITE_RISCV_PATH/tools/make/Makefile TARGET=riscv64
```

#### Execute a model

```shell
$ spike pk TFLITE_RISCV_PATH/tools/make/gen/riscv64_riscv64/bin/minimal /path/to/model.tflite
```
