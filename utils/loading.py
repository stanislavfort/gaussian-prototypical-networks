#library for Omniglot data loading and preprocessing

import matplotlib.pyplot as plt
import numpy as np

from visualize import visualize

def loadList(path, list_name, limit = None, verbose = False):

    class_info = []
    class_ids = []
    images = []

    class_id_now = -1
    class_name_now = ""

    p = list_name
    list_path = path + p

    f = open(list_path, "r")
    lines = f.readlines()

    if limit is not None:
        lines = lines[:limit]

    for line in lines: #goes through each character example
        image_path = line[:-1]
        subset_name, alphabet_name, character_name, image_name = image_path.split("/")

        class_info.append((subset_name,alphabet_name,character_name))

        if alphabet_name+character_name != class_name_now: #new class encoutered
            class_id_now += 1
            class_name_now = alphabet_name+character_name

        class_ids.append(class_id_now)

        img_filename = path + image_path

        if verbose:
            print(img_filename)

        im = plt.imread(img_filename, format='png')

        im = np.array(im)

        #print(np.mean(im))
        #print(im.shape)

        #plt.imshow(im)
        #plt.show()

        images.append(im)

    images = np.array(images)
    class_ids = np.array(class_ids)

    #print(images.shape)

    return class_ids, images, class_info


def loadOmniglot(path = "../data/", train = 1, limit = None):

    lists = [
    "images_background_list.txt",
    "images_background_small1_list.txt",
    "images_background_small2_list.txt",
    "images_evaluation_list.txt"
    ]

    labels_train, images_train, info_train = loadList(path, lists[train], verbose = False, limit = limit)
    labels_val, images_val, info_val = loadList(path, lists[3], verbose = False, limit = limit)

    return (labels_train, images_train, info_train, labels_val, images_val, info_val)


#for testing, if ran directly
if __name__ == "__main__":

    labels_train, images_train, info_train, labels_val, images_val, info_val = loadOmniglot(train = 0, limit = None)

    ids = range(images_train.shape[0])
    ids = [i*20 for i in range(int(images_train.shape[0]/20))] #just single copy of a character

    visualize(
        images_train[ids,:,:],
        output = "characters_train_small.png",
        width = 25
        )

    ids = range(images_val.shape[0])
    ids = [i*20 for i in range(int(images_val.shape[0]/20))] #just single copy of a character

    visualize(
        images_val[ids,:,:],
        output = "characters_val_small.png",
        width = 25
        )
