import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


######################################################################################
# PRE-PROCESSING
######################################################################################
def pre_process(X_train, X_test, prep):
    # print(f'pre_process X_train.shape : {X_train.shape}, prep: {prep}')
    # image dataset
    if len(X_train.shape) > 2:

        if 'canny' in prep:
            # A value close to 1.0 means that only the edge image is visible.
            # A value close to 0.0 means that only the original image is visible.
            # If a tuple (a, b), a random value from the range a <= x <= b will be sampled per image.
            aug = iaa.Canny(alpha=(0.75, 1.0))
            X_train = aug(images=X_train)

        if 'clahe' in prep:
            # CLAHE on all channels of the images
            aug = iaa.AllChannelsCLAHE()
            X_train = aug(images=X_train)

        if 'blur' in prep:
            # Blur each image with a gaussian kernel with a sigma of 3.0
            aug = iaa.GaussianBlur(sigma=(0.0, 3.0))
            X_train = aug(images=X_train)

        if 'augment' in prep:
            # append augmented dataset to the existing dataset
            # ia.seed(1)

            seq = iaa.Sequential([
                iaa.Fliplr(0.5), # horizontal flips
                iaa.Crop(percent=(0, 0.1)), # random crops
                # # Small gaussian blur with random sigma between 0 and 0.5.
                # # But we only blur about 50% of all images.
                # iaa.Sometimes(
                #     0.5,
                #     iaa.GaussianBlur(sigma=(0, 0.5))
                # ),
                # # Strengthen or weaken the contrast in each image.
                # iaa.LinearContrast((0.75, 1.5)),
                # # Add gaussian noise.
                # # For 50% of all images, we sample the noise once per pixel.
                # # For the other 50% of all images, we sample the noise per pixel AND
                # # channel. This can change the color (not only brightness) of the
                # # pixels.
                # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                # # Make some images brighter and some darker.
                # # In 20% of all cases, we sample the multiplier once per channel,
                # # which can end up changing the color of the images.
                # iaa.Multiply((0.8, 1.2), per_channel=0.2),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-8, 8)
                )
            ], random_order=True) # apply augmenters in random order

            augmented_images = seq(images=X_train)
            X_train = np.concatenate((X_train, augmented_images))

        if 'gray' in prep:
            X_train = np.dot(X_train[..., :3], [0.2126 , 0.7152 , 0.0722 ])
            X_test = np.dot(X_test[..., :3], [0.2126 , 0.7152 , 0.0722 ])

        # print('flatten images')
        # Flatten the images
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # X_train = np.reshape(X_train,(X_train.shape[0],np.prod(X_train.shape[1:])))
        # X_test = np.reshape(X_test ,(X_test.shape[0],np.prod(X_test.shape[1:])))
        # print('done flattening images. x.shape, X_test.shape : ', x.shape, X_test.shape)

    # [0,1] Normalization
    if 'norm' in prep:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if 'std' in prep:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # print("pre_process : X_train.shape X_train.shape : ", X_train.shape, X_test.shape)

    return X_train, X_test