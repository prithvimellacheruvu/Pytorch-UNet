import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == ".bin":
        gridOG = np.zeros((1, 61, 61, 25))
        gridStartEnd = np.zeros((1, 61, 61, 25))
        with open(filename, mode="rb") as infile:
            for i in range(61):
                for j in range(61):
                    for k in range(25):
                        tempRow = infile.read(2)
                        for l in range(2):
                            if l == 0:
                                gridOG[0][i][j][k] = tempRow[l]
                            if l == 1:
                                gridStartEnd[0][i][j][k] = tempRow[l]
        OG = torch.as_tensor(gridOG.copy()).float().contiguous()
        StartEnd = torch.as_tensor(gridStartEnd.copy()).float().contiguous()
        return torch.cat((OG, StartEnd), dim=0), gridOG.squeeze(0)
    else:
        raise ValueError("File not a .bin")


def load_mask(filename):
    ext = splitext(filename)[1]
    if ext == ".bin":
        grid = np.zeros((1, 61, 61, 25))
        with open(filename, mode="rb") as infile:
            for i in range(61):
                for j in range(61):
                    for k in range(25):
                        tempRow = infile.read(1)
                        for l in range(1):
                            grid[0][i][j][k] = tempRow[l]
        return grid
    else:
        raise ValueError("File not a .bin")


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + ".*"))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 3:
        return np.unique(mask)
    else:
        raise ValueError(f"Loaded masks should have 3 dimensions, found {mask.ndim}")


class BasicDataset(Dataset):
    def __init__(
        self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ""
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.idsImg = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        self.idsMask = [
            splitext(file)[0]
            for file in listdir(mask_dir)
            if isfile(join(mask_dir, file)) and not file.startswith(".")
        ]
        # correspondance of input img & mask to train on
        self.idsImg = sorted(self.idsImg)
        self.idsMask = sorted(self.idsMask)

        if not self.idsImg:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )
        if not self.idsMask:
            raise RuntimeError(
                f"No input file found in {mask_dir}, make sure you put your images there"
            )

        logging.info(f"Creating dataset with {len(self.idsImg)} examples")
        # logging.info('Scanning mask files to determine unique values')
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))

        # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        self.mask_values = list([0, 1])
        logging.info(f"Unique mask values: {self.mask_values}")

    def __len__(self):
        return len(self.idsImg)

    def __getitem__(self, idx):
        nameImg = self.idsImg[idx]
        nameMask = self.idsMask[idx]
        mask_file = list(self.mask_dir.glob(nameMask + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(nameImg + ".*"))

        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {name}: {img_file}"
        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        mask = load_mask(mask_file[0])
        img, gridOG = load_image(img_file[0])

        # %%%%%%%%%%%% Concatenate the inputMap & startGoal positions map along dim=1 %%%%%%%%%%%%%%

        # assert img.size == mask.size, \
        #     f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        return {"image": img, "mask": torch.as_tensor(mask.copy()).long().contiguous()}


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix="_mask")
