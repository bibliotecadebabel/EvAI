import torch
import torchvision.transforms as transforms
import random
import utilities.CutOut as cutout_utility

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
        resized_crop = transforms.Pad(2, fill=0, padding_mode='constant')(crop_image)
        random_affine = transforms.RandomAffine(25, translate=(0.1, 0.1), shear=20)
        image_affine = transforms.RandomApply([random_affine])(resized_crop)
        image_h_flip = transforms.RandomHorizontalFlip()(image_affine)

        final_images.append(image_h_flip)

    del five_crop_images

    tensors_images = torch.stack([transforms.ToTensor()(crop) for crop in final_images])
    normalize_tensors = torch.stack([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(t) for t in tensors_images])

    return normalize_tensors

def customCutout(image):
    
    p = 0.5
    rand_value = random.random()
    if p < rand_value:
        return image
    else:
        image_tensor = transforms.ToTensor()(image)
        tensor_cutout = cutout_utility.doCutOut(img=image_tensor, n_holes=1, size=16)
        image_cutout = transforms.ToPILImage()(tensor_cutout)
        
        return image_cutout

def customRandomCutout(image):
    
    p = 0.5
    rand_value = random.random()

    if p < rand_value:
        return image
    else:
        image_tensor = transforms.ToTensor()(image)
        size_cutout = random.randint(1, 16)
        tensor_cutout = cutout_utility.doCutOut(img=image_tensor, n_holes=1, size=size_cutout)
        image_cutout = transforms.ToPILImage()(tensor_cutout)

        return image_cutout

def customRandomCrop(image):
    
    random_crop = transforms.RandomCrop(28, fill=0, padding_mode='constant')
    image_crop = transforms.RandomApply([random_crop])(image)
    
    if image_crop.size[0] < 32:
        image_pad = transforms.Pad(2, fill=0, padding_mode='constant')(image_crop)
    else:
        image_pad = image
    
    return image_pad

def customRotation(image):

    random_rotation = transforms.RandomRotation(25)
    image_rotation = transforms.RandomApply([random_rotation])(image)

    return image_rotation

def customShear(image):

    random_shear = transforms.RandomAffine(0, shear=20)
    image_shear = transforms.RandomApply([random_shear])(image)

    return image_shear


class AugmentationSettings:

    def __init__(self):
        
        self.__crop = transforms.Lambda(customFiveCrop)
        self.__fullcrop = transforms.Lambda(fiveCrop)
        
        self.customRandomCrop = transforms.Lambda(customRandomCrop)
        self.translate = transforms.RandomAffine(0, translate=(0.1, 0.1))                
        self.randomHorizontalFlip = transforms.RandomHorizontalFlip()
        self.randomRotation = transforms.Lambda(customRotation)
        self.randomShear = transforms.Lambda(customShear)
        self.cutout = transforms.Lambda(customCutout)
        self.randomCutout = transforms.Lambda(customRandomCutout)
        
    def generateTransformCompose(self, transform_dict, customCrop=False):
        
        transform_compose_list = []

        if customCrop == False:

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



