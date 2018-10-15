# Tensorflow Lite for RISCV

This is an experimental port of Tensorflow Lite for RISCV architecture

#### Prerequiste
Install the dependencies by running the `download_dependencies.sh` script in tools/make

#### Build minimal 

```shell
$ export TFLITE_RISCV_PATH=tensorflow/tensorflow/contrib/lite/experimental/riscv
$ make -f $TFLITE_RISCV_PATH/tools/make/Makefile TARGET=riscv64
```

#### Execute a model

```shell
$ spike pk TFLITE_RISCV_PATH/tools/make/gen/riscv64_riscv64/bin/minimal /path/to/model.tflite
```
