import os
from PIL import Image
import torch
from torchvision.transforms import transforms
from utils.vocab import tokenize
import torch.utils.data as data
from pprint import pprint
import orjson as json


class IQDataset(data.Dataset):
    def __init__(self, dataset_path, train_or_val="train") -> None:
        with open(dataset_path, "r") as f:
            self.dataset = json.loads(f.read())["data"]

                
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224,
                                        scale=(1.00, 1.2),
                                        ratio=(0.75, 1.3333333333333333)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

        self.train_or_val = "train" if train_or_val=="train" else "val"
        self.image_dir = "/data/lama/mscoco/images/{}2014".format(self.train_or_val)

    def __getitem__(self, index):
        object = self.dataset[index]
        path = "COCO_%s2014_%012d.jpg" % (self.train_or_val, object["image_id"])
        image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
        image = self.transform(image)
        object["image"] = image
        return object

    def __len__(self):
        return len(self.dataset)
