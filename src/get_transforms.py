from utils import transforms as Trans




# ---------------------------------------------------------------------------------------------------------------
class DriveSegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [Trans.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(Trans.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(Trans.RandomVerticalFlip(vflip_prob))
        trans.extend([
            Trans.RandomCrop(crop_size),
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])
        self.transforms = Trans.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DriveSegmentationPresetEval:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Trans.Compose([
            Trans.Resize([crop_size, crop_size]),
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def drive_get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # 此数据集的每张图片都是565*584形状的图片。
    base_size = 544
    # 裁剪尺寸为480*480，也就是说Unet输入的图片大小固定为480*480
    crop_size = 544

    if train:
        return DriveSegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return DriveSegmentationPresetEval(crop_size, mean=mean, std=std)


# ---------------------------------------------------------------------------------------------------------------
class ChaseDB1SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # min_size = int(0.5 * base_size)
        # max_size = int(1.2 * base_size)

        # trans = [Trans.RandomResize(min_size, max_size)]
        trans = [Trans.Resize([crop_size, crop_size])]
        if hflip_prob > 0:
            trans.append(Trans.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(Trans.RandomVerticalFlip(vflip_prob))
        trans.extend([
            Trans.CenterCrop(crop_size),
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])
        self.transforms = Trans.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)



class ChaseDB1SegmentationResizePresetEval:
    def __init__(self, base_size, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Trans.Compose([
            Trans.Resize([crop_size, crop_size]),
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class ChaseDB1SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Trans.Compose([
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)



def chase_db_get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # 此数据集图片都是999*960的大小，故要裁切成960*960输入网络
    base_size = 544
    crop_size = 544

    if train:
        return ChaseDB1SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        # return ChaseDB1SegmentationPresetEval(mean=mean, std=std)
        return ChaseDB1SegmentationResizePresetEval(base_size, crop_size, mean=mean, std=std)





class RITESegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [Trans.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(Trans.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(Trans.RandomVerticalFlip(vflip_prob))
        trans.extend([
            Trans.CenterCrop(crop_size),
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])
        self.transforms = Trans.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class RITESegmentationPresetEval:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Trans.Compose([
            Trans.Resize([crop_size, crop_size]),
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def rite_get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 512
    crop_size = 512

    if train:
        return RITESegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return RITESegmentationPresetEval(crop_size=crop_size, mean=mean, std=std)


if __name__ == "__main__":
    present_train = DriveSegmentationPresetTrain(base_size=565, crop_size=480)
