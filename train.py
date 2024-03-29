# -*- Python -*-

#
# Copyright (c) 2017-2023 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

# A simple example of using WebDataset high performance distributed storage
# for ImageNet training.  This uses the PyTorch Lightning framework.

# Loosely based on
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/imagenet.py

import functools
import os.path
import pprint
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.utils import data
import torchvision
import webdataset as wds
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchvision import transforms
import simple_cluster

torchvision
pp = pprint.PrettyPrinter(indent=4, depth=2).pprint


def identity(x):
    return x


class ImagenetData(pl.LightningDataModule):
    def __init__(self, shards=None, valshards=None, batch_size=64, workers=4, bucket=None, **kw):
        super().__init__(self)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.training_urls = os.path.join(bucket, shards)
        print("training_urls = ", self.training_urls)
        self.val_urls = os.path.join(bucket, valshards)
        print("val_urls = ", self.val_urls)
        self.batch_size = batch_size
        self.num_workers = workers
        print("batch_size", self.batch_size, "num_workers", self.num_workers)

    def make_transform(self, mode="train"):
        if mode == "train":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        elif mode == "val":
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )

    def make_loader(self, urls, mode="train"):

        if isinstance(urls, str) and urls.startswith("fake:"):
            xs = torch.randn((self.batch_size, 3, 224, 224))
            ys = torch.zeros(self.batch_size, dtype=torch.int64)
            return wds.MockDataset((xs, ys), 10000)

        if mode == "train":
            dataset_size = 1281167
            shuffle = 5000
        elif mode == "val":
            dataset_size = 5000
            shuffle = 0

        transform = self.make_transform(mode=mode)

        dataset = (
            wds.WebDataset(urls)
            .shuffle(shuffle)
            .decode("pil")
            .to_tuple("jpg;png;jpeg cls")
            .map_tuple(transform, identity)
            .batched(self.batch_size, partial=False)
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )

        loader.length = dataset_size // self.batch_size

        if mode == "train":
            # ensure same number of batches in all clients
            loader = loader.ddp_equalize(dataset_size // self.batch_size)
            # print("# loader length", len(loader))

        return loader

    def train_dataloader(self):
        return self.make_loader(self.training_urls, mode="train")

    def val_dataloader(self):
        return self.make_loader(self.val_urls, mode="val")

    @staticmethod
    def add_loader_specific_args(parser):
        parser.add_argument("-b", "--batch-size", type=int, default=128)
        parser.add_argument("--workers", type=int, default=6)
        parser.add_argument("--bucket", default="./shards")
        parser.add_argument("--shards", default="imagenet-train-{000000..001281}.tar")
        parser.add_argument("--valshards", default="imagenet-val-{000000..000006}.tar")
        return parser


class ImageClassifier(pl.LightningModule):
    def __init__(self, learning_rate=0.1, momentum=0.9, weight_decay=1e-4, model="resnet18", **kw):
        super().__init__()
        self.save_hyperparameters()
        self.model = eval(f"torchvision.models.{model}")()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("val_loss", loss_val, on_step=True, on_epoch=True)
        self.log("val_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log("val_acc5", acc5, on_step=True, on_epoch=True)

    @staticmethod
    def schedule(epoch):
        return 0.1 ** (epoch // 30)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, ImageClassifier.schedule)
        return [optimizer], [scheduler]

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        outputs = self.validation_epoch_end(*args, **kwargs)

        def substitute_val_keys(out):
            return {k.replace("val", "test"): v for k, v in out.items()}

        outputs = {
            "test_loss": outputs["val_loss"],
            "progress_bar": substitute_val_keys(outputs["progress_bar"]),
            "log": substitute_val_keys(outputs["log"]),
        }
        return outputs

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", type=float, default=0.1)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--model", type=str, default="resnet18")
        return parser


def main(args):
    if args.verbose:
        pp(vars(args))
    if args.accelerator in ["ddp"]:
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))
    data = ImagenetData(**vars(args))
    model = ImageClassifier(**vars(args))
    plugs = []
    if args.accelerator == "ddp":
        plugs.append(simple_cluster.SimpleCluster())
    trainer = pl.Trainer.from_argparse_args(args, plugins=plugs)
    if args.evaluate:
        trainer.test(model, data)
    else:
        trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ncclfix", action="store_false")
    parser.add_argument("--nccldebug", action="store_true")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ImagenetData.add_loader_specific_args(parser)
    parser = ImageClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.ncclfix:
        simple_cluster.auto_configure_nccl()
    if args.nccldebug:
        simple_cluster.debug_nccl()
    main(args)
