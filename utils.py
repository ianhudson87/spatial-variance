import matplotlib.pyplot as plt
import json

def imshow(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

def get_options(options_file = "options.json"):
    with open(options_file) as json_file:
        data = json.load(json_file)
        return data