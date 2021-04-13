# A small demonstration of using WebDataset with ImageNet and PyTorch Lightning

This is a small repo illustrating how to use WebDataset on ImageNet.
using the PyTorch Lightning framework.

First, create the virtualenv:

```Bash
$ ./run venv  # make virtualenv
```

Next, you need to shard the ImageNet data:

```Bash
$ ln -s /some/imagenet/directory data
$ mkdir shards
$ ./run makeshards  # create shards
```

Run the training script:

```Bash
$ ./run train  # run the training jobs using PyTorch lightning
```

Of course, for local data, there is no need to go through this trouble. However,
you can now easily train remotely, for example by putting the data on a webserver:

```Bash
$ rsync -av shards webserver:/var/www/html/shards
$ ./run train --bucket http://webserver/shards
```

The [AIStore server](http://github.com/nvidia/aistore) is a high performance
server that works very well with WebDataset.
