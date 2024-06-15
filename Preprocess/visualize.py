import os
import matplotlib.pyplot as plt
import cv2

def visualize_images(dir_path, n_images):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    dpath = dir_path
    count = 0

    for expression in os.listdir(dpath):
        # get list of images in a given expression class
        expression_path = os.path.join(dpath, expression)
        images = os.listdir(expression_path)

        # plot the images
        for j in range(n_images):
            img_path = os.path.join(expression_path, images[j])
            img = cv2.imread(img_path)

            axs[count // 2][count % 2].imshow(img)
            count += 1

            if count >= 4:
                break  # Break if we have shown the desired number of images

        if count >= 4:
            break  # Break if we have shown the desired number of images

    # Set titles outside the inner loop
    for i in range(2):
        for j in range(2):
            axs[i][j].title.set_text(os.listdir(dpath)[i * 2 + j])

    fig.tight_layout()
    plt.show(block=True)

dir_path = 'C:\\Users\\TIRUMALA PHANENDRA\\PycharmProjects\\DL Project\\dataset_aug'
n_images = 4
visualize_images(dir_path, n_images)
