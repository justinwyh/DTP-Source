import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class LowLightDataSet(Dataset):
    def __init__(self, data_dir):
        self.dataset_input_dir = os.path.join(data_dir, 'low')
        self.dataset_gt_dir = os.path.join(data_dir, 'high')
        self.list_of_files = self.listAllInputImageFiles(self.dataset_input_dir)
        self.HighResTransform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        self.LowResTransform = transforms.Compose(
           [
            transforms.Resize((256,256)),
            transforms.ToTensor()
           ]
        )
        self.greyScaleTransform = transforms.Compose(
            [transforms.Grayscale(1),
             transforms.ToTensor()
             ]
       )

    def __getitem__(self, index):
        image_name = self.list_of_files[index]
        hr_input_image_path = os.path.join(self.dataset_input_dir,image_name)
        hr_gt_image_path = os.path.join(self.dataset_gt_dir,image_name)
        with Image.open(hr_input_image_path) as img:
            tmp_image = img.convert('RGB')
            input_image_hr = self.HighResTransform(tmp_image)
            input_image_lr = self.LowResTransform(tmp_image)
        with Image.open(hr_gt_image_path) as img2:
            tmp_image = img2.convert('RGB')
            gt_image_hr = self.HighResTransform(tmp_image)
            grey_image = self.greyScaleTransform(tmp_image)
        return input_image_hr, input_image_lr, gt_image_hr, grey_image

    def __len__(self):
        return len(self.list_of_files)

    @staticmethod
    def listAllInputImageFiles(data_dir):
        list = os.listdir(data_dir)
        files = []
        for l in list:
            fullpath = os.path.join(data_dir, l)
            # FOR DEBUG: print(fullpath)
            if os.path.isfile(fullpath):
                files.append(l)
        #files = sorted(files,key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        return files