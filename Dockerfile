FROM nvcr.io/nvidia/pytorch:21.03-py3
MAINTAINER Tom <tmbdev@gmail.com>
ENV DEBIAN_FRONTEND noninteractive
ENV PATH /usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:.
RUN apt-get -qq update
RUN apt-get -qqy install apt-utils
RUN apt-get -qqy install git curl unzip build-essential gcc g++ gdb
ENV PATH=.:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin
RUN conda install numpy
RUN conda install scipy
RUN conda install matplotlib
RUN conda install pip
RUN pip install -U git+git://github.com/tmbdev/webdataset.git#egg=webdataset
RUN pip install -U typer
RUN apt-get install net-tools
ENV PATH=.:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin:/sbin:/usr/sbin
# RUN conda install -c conda-forge pytorch-lightning
RUN pip install pytorch_lightning
RUN sed -i '3,34d' /usr/local/bin/nvidia_entrypoint.sh
