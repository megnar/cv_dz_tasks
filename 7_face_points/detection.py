import cv2

import albumentations as A
from torchvision import datasets, models
from torch import nn
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


import pandas as pd
import os

import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


target_size = (224, 224)

def compute_mean_and_std(images):
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std

def normalize_image(image, mean, std):
    normalized_image = (image - mean) / std
    return normalized_image

def resize_image(image, target_size):
    resized_image = cv2.resize(image, target_size)
    return resized_image


def to_tensor(image):
    image_tensor = torch.tensor(image.transpose((2, 0, 1))).float()
    return image_tensor

## main func
def preprocess_data(image, keypoint):

    resized_keypoints_np = None
    image_np = image.permute(1, 2, 0).numpy()
    resized_images_np = cv2.resize(image_np, target_size)
    resized_keypoints_np = None

    resized_keypoints_np = np.copy(keypoint[1:])
    resized_keypoints_np = resized_keypoints_np.astype(np.float32)
    resized_keypoints_np[::2] = resized_keypoints_np[::2] / image.shape[2] * target_size[0]
    resized_keypoints_np[1::2] = resized_keypoints_np[1::2] / image.shape[1] * target_size[1]

    mean, std = compute_mean_and_std(resized_images_np)
    normalized_images = normalize_image(resized_images_np, mean, std)

    images_tensor = to_tensor(normalized_images)
    # if not is_test:
    #   resized_keypoints_np = torch.tensor(resized_keypoints_np).float()
    resized_keypoints_np = torch.tensor(resized_keypoints_np).float()
    if resized_keypoints_np is not None:
      return resized_images_np, resized_keypoints_np
    return resized_images_np, np.ones((1))


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = np.array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res


class MyCustomDataset(Dataset):

    def __init__(
            self,
            mode,
            data_dir,
            gt_csv,
            fraction: float = 0.95,
            transform=None,
            fast_train=False
    ):
        ## list of tuples: (img_path, label)
        self._items = []
        self.mode = mode

        ## will use it later for augmentations
        self._transform = transform

        ## we can't store all the images in memory at the same time,
        ## because sometimes we have to work with very large datasets
        ## so we will only store data paths
        ## (also this is convenient for further augmentations)
        list_of_images = sorted(os.listdir(data_dir))
        list_of_labels = gt_csv
        result_list = []
        for i in range(len(list_of_images)):
            res_label = np.array([str(list_of_images[i])])
            res_label = np.concatenate((res_label, np.array(list_of_labels[list_of_images[i]])))
            result_list.append((os.path.join(data_dir, list_of_images[i]), res_label))

        if mode == "train":
            if not fast_train:
                self._items = result_list[: int(fraction * len(list_of_images))]
            else:
                self._items = result_list[: int((1 - fraction) * len(list_of_images))]

        elif mode == "val":
            self._items = result_list[int(fraction * len(list_of_images)):]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, label = self._items[index]

        ## read image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype(np.float32)
        image = image / 256
        image = torch.from_numpy(image).permute(2, 0, 1)

        image, label = preprocess_data(image=image, keypoint=label)
        label = np.clip(label, 0.1, 223.9)

        if self._transform:
            keypoints = np.zeros((14, 2))
            keypoints[:, 0] = label[::2]
            keypoints[:, 1] = label[1::2]
            transformed = self._transform(image=image, keypoints=keypoints)
            image = (transformed['image'])
            keypoints = (transformed['keypoints'])

            for i in range(len(keypoints)):
                label[2 * i] = keypoints[i][0]
                label[2 * i + 1] = keypoints[i][1]

        return image.transpose((2, 0, 1)), label


transform = A.Compose(
    [A.ColorJitter(p=0.3),
    A.Rotate(limit=30, p=1),
    A.GaussianBlur(p=0.3),
    ],
    keypoint_params = A.KeypointParams(format='xy')
)


class MyTrainingModule(pl.LightningModule):
    def __init__(self):
        """Define computations here."""

        super().__init__()
        self.validation_step_outputs = []

        self.conv1 = nn.Conv2d(3, 64, 3, padding = 'same')
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(64, 128, 3, padding = 'same')
        self.norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(128, 256, 3, padding = 'same')
        self.norm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(256, 256, 3, padding = 'same')
        self.norm4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p=0.3)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.fc1 = nn.Linear(256 * 14 * 14, 128)
        self.relu5 = nn.ReLU()
        self.norm_fc1 = nn.BatchNorm1d(128)
        self.dropout_fc1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(128, 28)
        # resnet
        # self.resnet = models.resnet18(pretrained=False)
        # self.resnet.fc = nn.Linear(512, 28)
        self.loss = nn.MSELoss()


    def forward(self, x):
        """Define forward pass."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu5(x)
        x = self.norm_fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
       # x = self.resnet(x)

        return x

    def training_step(self, batch, batch_idx):
        """The full training loop"""
        x, y = batch
        y_logit = self(x)
        loss = self.loss(y_logit, y)

        train_loss = torch.mean(((y_logit.detach() - y))**2)

        return {'loss': loss, 'train_loss' : train_loss}

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=5,
            verbose=True,
        )
        lr_dict = {
            # The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
        }

        return [optimizer], [lr_dict]

    def validation_step(self, batch, batch_idx):
        """The full val loop"""
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)

        val_loss = torch.mean(((y_logit.detach() - y)) ** 2) / 5
        self.validation_step_outputs.append(val_loss)

        return {'loss': loss, 'val_loss': val_loss}

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss", epoch_average, on_epoch=True, on_step=False)
        self.validation_step_outputs.clear()

def train_detector(train_gt, train_img_dir, fast_train=True):

    transform = A.Compose(
        [A.Rotate(limit=35, p=1),
          A.GaussianBlur(p=0.3),
          A.ToGray(p=0.05),
          A.ColorJitter(p=0.1)
          ],
        keypoint_params=A.KeypointParams(format='xy')
    )
    ds_train = MyCustomDataset(mode="train", data_dir=train_img_dir, gt_csv=train_gt, fast_train=fast_train)
    ds_val = MyCustomDataset(mode="val", data_dir=train_img_dir, gt_csv=train_gt)
    dl_train = DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=os.cpu_count())
    dl_val = DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=os.cpu_count())

    trainer1 = pl.Trainer(
        max_epochs=70,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False
    )

    trainer2 = pl.Trainer(
        max_epochs=1,
        num_sanity_val_steps=0,
        logger = False,
        enable_checkpointing=False
    )

    training_module = MyTrainingModule()

    if fast_train:
        trainer2.fit(training_module, dl_train, dl_val)

    else:
        trainer1.fit(training_module, dl_train, dl_val)
    return training_module

def detect(model_filename, test_img_dir):

    model = MyTrainingModule.load_from_checkpoint(model_filename, map_location=torch.device('cpu'))
    model.eval()
    model.to("cpu")
    image_dirs = os.listdir(test_img_dir)
    ans = dict()
    with torch.no_grad():
      for image_name in image_dirs:
        img_path = os.path.join(test_img_dir, image_name)
        test_img = Image.open(img_path).convert("RGB")
        test_img = np.array(test_img).astype(np.float32)
        test_img = test_img / 256
        img_shape = test_img.shape
        image = torch.from_numpy(test_img).permute(2, 0, 1)

        image, label = preprocess_data(image = image, keypoint = np.zeros(28))
        label = np.clip(label, 0.1, 223.9)

        image  = image.transpose((2,0,1))
        inputs = torch.from_numpy(image)

        inputs = inputs.to("cpu")[None, :]
        res = model(inputs).detach()

        res = res[0].numpy()
        res[0::2] = res[0::2] * img_shape[0] / 224
        res[1::2] = res[1::2] * img_shape[1] / 224
        ans[image_name] = res

    return ans
