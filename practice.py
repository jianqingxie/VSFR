from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json
import random



def painting(image_id, size, path):
    pylab.rcParams['figure.figsize'] = (size, size)

    annFile = path
    coco = COCO(annFile)
    img = coco.loadImgs(image_id)[0]
    I = io.imread(img['flickr_url'])

    # load and display caption annotations
    annIds = coco.getAnnIds(imgIds=img['id']);
    anns = coco.loadAnns(annIds)
    print(anns)
    coco.showAnns(anns)
    plt.imshow(I)
    plt.axis('off')
    plt.show()


def paints(x, y):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if x is None or y is None:
        x = [0,25,36,49,64,81,100,121,144,169]
        y = [131.5,132.1,132.5,132.7,133.1,132.9,133.0,133.2,132.8,132.9]

    ax1.plot(x, y, 'c*-', label='CIDEr score', linewidth=2)
    ax1.legend()
    ax1.set_yticks([131,131.5,132,132.5,133,133.5])
    plt.ylim(131,134)
    plt.show()

if __name__ == '__main__':
    output_path = 'test_online/results/captions_val2014_Baseline_results.json'
    val_annotation = '../m2_annotations/captions_val2014.json'

    image_caption = json.load(open(output_path, 'r'))
    captions = dict()

    for cap in image_caption:
        captions[cap['image_id']] = cap['caption']

    for image_id in random.sample(captions.keys(), 1):
        print(image_id)
        painting(image_id, 8.0, val_annotation)
        print("********************************************")
        print(captions[image_id])







