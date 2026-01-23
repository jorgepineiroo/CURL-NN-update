import logging
import torch
import os
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_adobe_5k_data(img_ids_filepath, data_dirpath):
    """ Loads the image data into a Python dictionary
    :param img_ids_filepath: path to the .txt file containing image IDs
    :param data_dirpath: path to the dataset folder containing input/ and output/ subfolders
    :returns: Python dictionary containing image pairs
    :rtype: Dictionary
    """
    data_dict = dict()

    with open(img_ids_filepath) as f:
        '''
        Load the image ids into a list data structure
        '''
        image_ids = f.readlines()
        # Remove whitespace characters like `\n` at the end of each line
        image_ids_list = [x.strip() for x in image_ids if x.strip()]

    idx = 0
    img_id_to_idx_dict = {}

    for root, dirs, files in os.walk(data_dirpath):
        for file in files:
            # Extract image ID from filename (e.g., "image0001" from "image0001-original.png")
            img_id = file.split("-")[0]

            # Check if this image ID is in our list
            is_id_in_list = img_id in image_ids_list

            if is_id_in_list:
                if img_id not in img_id_to_idx_dict.keys():
                    img_id_to_idx_dict[img_id] = idx
                    data_dict[idx] = {}
                    data_dict[idx]['input_img'] = None
                    data_dict[idx]['output_img'] = None
                    idx_tmp = idx
                    idx += 1
                else:
                    idx_tmp = img_id_to_idx_dict[img_id]

                if "input" in root.lower():
                    data_dict[idx_tmp]['input_img'] = os.path.join(root, file)

                elif "output" in root.lower():
                    data_dict[idx_tmp]['output_img'] = os.path.join(root, file)

    # Validate that all pairs have both input and output
    valid_data_dict = {}
    valid_idx = 0
    for idx, imgs in data_dict.items():
        if imgs.get('input_img') and imgs.get('output_img'):
            valid_data_dict[valid_idx] = (imgs['input_img'], imgs['output_img'])
            valid_idx += 1
        else:
            logging.warning(f"Incomplete pair at index {idx}: {imgs}")

    logging.info(f"Loaded {len(valid_data_dict)} valid image pairs from {data_dirpath}")
    return valid_data_dict


class CustomDataset(Dataset):
    def __init__(self, data_dir, target_size, split='train', random_resize=False, random_crop=False, test_plot=False):
        """
        Initialize the dataset.
        
        :param data_dir: Path to the dataset folder containing input/, output/, and images_*.txt files
        :param target_size: Tuple (width, height) for cropping
        :param split: Which split to use - 'train', 'valid', or 'test'
        :param random_resize: Whether to apply random resizing
        :param random_crop: Whether to apply random cropping
        :param test_plot: Whether this is for testing/visualization
        """
        # Map split names to txt filenames
        split_to_filename = {
            'train': 'images_train.txt',
            'valid': 'images_valid.txt',
            'validation': 'images_valid.txt',
            'test': 'images_test.txt'
        }
        
        if split not in split_to_filename:
            raise ValueError(f"Invalid split '{split}'. Must be one of: {list(split_to_filename.keys())}")
        
        txt_filename = split_to_filename[split]
        
        # Check if .txt file is in data_dir or parent directory
        img_ids_filepath = os.path.join(data_dir, txt_filename)
        if not os.path.exists(img_ids_filepath):
            # Try parent directory
            img_ids_filepath = os.path.join(os.path.dirname(data_dir), txt_filename)
        
        if not os.path.exists(img_ids_filepath):
            raise FileNotFoundError(f"Could not find {txt_filename} in {data_dir} or its parent directory")
        
        logging.info(f"Using image IDs from: {img_ids_filepath}")
        logging.info(f"Loading images from: {data_dir}")
        
        self.pairs = load_adobe_5k_data(img_ids_filepath, data_dir)
        self.num_samples = len(self.pairs)
        self.target_size = target_size
        self.random_crop = random_crop
        self.random_resize = random_resize
        self.test_plot = test_plot
        
        # Only create crop transform if target_size is valid
        if target_size[0] is not None and target_size[1] is not None:
            self.crop_transform = torch.nn.Sequential(
                transforms.RandomCrop(size=target_size),
                transforms.RandomHorizontalFlip(p=0.5)
            )
        else:
            self.crop_transform = None

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def normalise_image(img):
        """Normalises image data to be a float between 0 and 1
        """
        img = img.astype('float32') / 255
        return img

    def random_resize_compute(self, input_img, target_img, input_dir):
        if self.target_size[0] is None or self.target_size[1] is None:
            return input_img, target_img
            
        if input_img.shape[0] < self.target_size[0] or input_img.shape[1] < self.target_size[1]:
            logging.info(f'####### WARNING #######')
            logging.info(f'size issue check {input_dir}')
            return input_img, target_img

        h_res = random.randint(self.target_size[1], input_img.shape[1])
        w_res = int(input_img.shape[0] * h_res / input_img.shape[1])

        input_img = cv2.resize(input_img, (h_res, w_res))
        target_img = cv2.resize(target_img, (h_res, w_res))
        return input_img, target_img

    def random_crop_compute(self, img, target):
        if self.crop_transform is None:
            return img, target
            
        img = img[None, ...]
        target = target[None, ...]
        img_batch = torch.concat([img, target], dim=0)
        img_batch = self.crop_transform(img_batch)
        return torch.squeeze(img_batch[0]), torch.squeeze(img_batch[1])

    def __getitem__(self, idx):
        input_dir, target_dir = self.pairs[idx]

        input_img = cv2.cvtColor(cv2.imread(input_dir), cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(cv2.imread(target_dir), cv2.COLOR_BGR2RGB)

        # make sure the resolutions to be identical
        target_img = cv2.resize(target_img, (input_img.shape[1], input_img.shape[0]))

        if self.random_resize:
            input_img, target_img = self.random_resize_compute(input_img, target_img, input_dir)

        input_img = self.normalise_image(input_img)
        target_img = self.normalise_image(target_img)

        input_img = torch.tensor(input_img)
        target_img = torch.tensor(target_img)

        input_img = input_img.permute(2, 0, 1)
        target_img = target_img.permute(2, 0, 1)

        if self.random_crop and self.target_size[0] is not None and self.target_size[1] is not None:
            if input_img.shape[1] >= self.target_size[0] and input_img.shape[2] >= self.target_size[1]:
                input_img, target_img = self.random_crop_compute(input_img, target_img)

        if self.test_plot:
            return input_img.permute(1, 2, 0), target_img.permute(1, 2, 0)
        else:
            return input_img, target_img, input_dir


if __name__ == '__main__':
    # Example usage with your dataset structure
    data_path = 'path/to/curl_custom_dataset'

    batch = 8
    training_data = CustomDataset(data_dir=data_path, target_size=(1000, 1000),
                                  random_resize=True, random_crop=True, test_plot=True)
    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=False, num_workers=10)

    for i, j in train_dataloader:
        print(f"Input shape: {i.shape}, Target shape: {j.shape}")
