# -*- Python -*-

# A simple example of using WebDataset for ImageNet training.
# This uses the PyTorch Lightning framework.

import os.path
import torch
import torchvision
from torchvision import transforms
from torch.nn import functional as F
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import plugins
import webdataset as wds


def identity(x):
    return x


class ImagenetData(pl.LightningDataModule):

    def __init__(self, training_urls=None, val_urls=None, batch_size=64, num_workers=4, bucket=None):
        super().__init__(self)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.training_urls = os.path.join(bucket, training_urls)
        print("training_urls = ", self.training_urls)
        self.val_urls = os.path.join(bucket, val_urls)
        print("val_urls = ", self.val_urls)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.world_size = 0
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        print("batch_size", self.batch_size, "num_workers", self.num_workers, "world_size", self.world_size)

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
            dataset, batch_size=None, shuffle=False, num_workers=self.num_workers,
        )
        loader.length = dataset_size // self.batch_size

        if self.world_size > 0:
            number_of_batches = dataset_size // (self.batch_size * self.world_size)
            print("# batches per node = ", number_of_batches)
            loader = loader.repeat(2).slice(number_of_batches)
            loader.length = number_of_batches

        return loader

    def train_dataloader(self):
        return self.make_loader(self.training_urls, mode="train")

    def val_dataloader(self):
        return self.make_loader(self.val_urls, mode="val")


class ImageClassifier(pl.LightningModule):

    def __init__(self, learning_rate=0.1):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.learning_rate = learning_rate

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        err = (output.argmax(1) != y).sum() / float(len(y))
        return dict(loss=loss, err=err)

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main(hparams):
    model = ImageClassifier()
    plugin = plugins.DDPPlugin(find_unused_parameters=False)
    trainer = pl.Trainer.from_argparse_args(args, plugins=plugin)
    data = ImagenetData(batch_size=args.batch_size, training_urls=args.shards, val_urls=args.valshards, bucket=args.bucket)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("--bucket", default="./shards")
    parser.add_argument("--shards", default="imagenet-train-{000000..000489}.tar")
    parser.add_argument("--valshards", default="imagenet-val-{000000..000022}.tar")
    parser = pl.Trainer.add_argparse_args(
        parser.add_argument_group(title="pl.Trainer args")
    )
    args = parser.parse_args()
    main(args)
