from torchvision import transforms
import torchvision.transforms.functional as TF


def my_transforms(self, image):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale = 360
    input_shape = 224

    images = []

    image = TF.resize(image, scale)
    image = TF.resized_crop(image, i=80, j=144, h=224, w=224, size=224)
    images.append(image)
    images.append(TF.hflip(image))
    images.append(TF.vflip(image))
    images.append(TF.rotate(image, 90))
    images.append(TF.rotate(image, 270))
    return images
    # data_transforms = transforms.Compose([
    #     transforms.Resize(scale),
    #     transforms.RandomResizedCrop(input_shape),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(degrees=90),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)])

def transform(self, X):
    new_X = []
    for img in X:
        img_tfm = self.my_transform(img)
        for i in img_tfm:
            new_X.append(i)
    return new_X
