import torch
import torchvision.transforms as transforms
import random

def customFiveCrop(image):
    img_zoomout = transforms.Resize(28, interpolation=2)(image)
    pad_image = transforms.Pad(2, fill=0, padding_mode='constant')(img_zoomout)
    five_crop_images = transforms.FiveCrop(28)(pad_image)

    random_index = random.randint(0, 4)

    if random_index < 4:
        crop_image = five_crop_images[random_index]
        resized_image = transforms.Resize(32, interpolation=2)(crop_image)
    else:
        resized_image = image

    return resized_image

def fiveCrop(image):
    #img_zoomout = transforms.Resize(28, interpolation=2)(image)
    #pad_image = transforms.Pad(2, fill=0, padding_mode='constant')(img_zoomout)
    five_crop_images = transforms.FiveCrop(28)(image)
    final_images = []
    
    for crop_image in five_crop_images:
        resized_crop = transforms.Resize(32, interpolation=2)(crop_image) 
        random_affine = transforms.RandomAffine(25, translate=(0.1, 0.1), shear=20)
        image_affine = transforms.RandomApply([random_affine])(resized_crop)
        image_h_flip = transforms.RandomHorizontalFlip()(image_affine)

        final_images.append(image_h_flip)

    del five_crop_images

    tensors_images = torch.stack([transforms.ToTensor()(crop) for crop in final_images])
    normalize_tensors = torch.stack([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(t) for t in tensors_images])

    return normalize_tensors

def CropTensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def CropNormalize(tensors):
    return torch.stack([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(t) for t in tensors])

class AugmentationSettings:

    def __init__(self, affine_degress_rotation=0, affine_shear=None):
        
        self.__crop = transforms.Lambda(customFiveCrop)
        self.__fullcrop = transforms.Lambda(fiveCrop)

        self.randomAffine = transforms.RandomAffine(affine_degress_rotation, translate=(0.1, 0.1), shear=affine_shear)
                
        self.randomHorizontalFlip = transforms.RandomHorizontalFlip()

        self.contrast = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)

        padding = (6,6)
        self.padding = transforms.Pad(padding, fill=0, padding_mode='constant')        

        size = (24, 24)
        self.zoomout = transforms.Resize(size)

        size = (32, 32)
        self.zoomin = transforms.Resize(size)
    
    def generateTransformCompose(self, transform_dict, fiveCrop=False):
        
        transform_compose_list = []

        if fiveCrop == False:

            for key in transform_dict.keys():
                
                value = transform_dict.get(key)

                if value == True:
                    transform_compose_list.append(key)

            transform_compose_list.append(transforms.ToTensor())
            transform_compose_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        else:    
            transform_compose_list.append(self.__fullcrop)
        
        print("Augmentation list: ", transform_compose_list)

        return transforms.Compose(transform_compose_list)



