import os
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import operator
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d import axes3d
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE


def show_image(image, comparison_image=None, title="", figsize=(20,12)):
    plt.figure(figsize=figsize)

    if not comparison_image is None:
        imout = np.hstack((comparison_image, image))
    else:
        imout = image.copy()

    if len(image.shape) == 3:
        b, g, r = cv2.split(imout)
        imout = cv2.merge([r, g, b])
    else:
        imout = image.copy()

    plt.imshow(imout)
    plt.title(title)
    plt.show()


def smooth_image(img):
    smoothed = cv2.bilateralFilter(img, 15, 50, 50)

    show_image(smoothed, title="smoothed")
    return smoothed


def enh_hist(img):
    tile = 25
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(tile, tile))
    clahe_img = []
    for c in cv2.split(img):
        clahe_img.append(clahe.apply(c))
    clahe_img = cv2.merge(clahe_img)

    show_image(clahe_img, title="hist enhanced")

    return clahe_img


def median_blur(img):
    median_img = cv2.medianBlur(img, 7)
    show_image(median_img, title="Median blur")

    return median_img


def cluster_colors(img):
    img_proc = img.reshape((-1, 3))
    img_proc = np.float32(img_proc)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    k = 35
    ret, label, center = cv2.kmeans(img_proc, k, None, criteria, 10, flags)

    center = np.uint8(center)
    res = center[label.flatten()]
    digitized_img = res.reshape((img.shape))
    # cv2.imwrite('digitized.png', digitized_img)

    show_image(digitized_img, title="clustered")
    return digitized_img


def calculate_color_palette(image):
    color_palette = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            b,g,r = image[x][y]
            color_palette.append((b,g,r))
    color_palette = set(color_palette)
    return np.array(sorted(color_palette, key=operator.itemgetter(0,1,2)))


def plot_color_palette(reduced_color_palette):
    palette_image = np.full((15, 0, 3), (0, 0, 0))

    for i, col in enumerate(reduced_color_palette):
        palette_image = np.hstack((palette_image, np.full((15, 1, 3), col)))

    palette_image = np.uint8(palette_image)
    show_image(palette_image, title="color palette")


def outline_image(img, color_palette):
    morphed_img = np.full(img.shape, (255, 255, 255))

    for col in color_palette:
        mask = cv2.inRange(img, np.array(col), np.array(col))

        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #     morphed_img = cv2.drawContours(morphed_img, contours, -1, [int(i) for i in col ], 1)
        morphed_img = cv2.drawContours(morphed_img, contours, -1, (125, 125, 125), 1)

    morphed_img = np.uint8(morphed_img)
    show_image(morphed_img, title="morphed")
    return morphed_img


def scatter_color_map(palette):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    b, r, g = cv2.split(np.array([palette]))

    xs = r[0]
    ys = g[0]
    zs = b[0]
    color_hex = [rgb2hex(rgb) for rgb in np.array(palette)/255.]
    ax.scatter(xs, ys, zs, c=color_hex)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def ocv_kmeans_cluster_colors(img, n=32):
    t0 = time.time()
    img_proc = img.reshape((-1, 3))
    img_proc = np.float32(img_proc)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    ret, label, center = cv2.kmeans(img_proc, n, None, criteria, 10, flags)

    center = np.uint8(center)
    res = center[label.flatten()]
    digitized_img = res.reshape((img.shape))
    t1 = time.time()

    print("Execution time: {} s".format(t1-t0))
    show_image(digitized_img, title="digitized")

    return digitized_img


def kmeans_cluster_colors(color_palette, n=32):
    t0 = time.time()
    Y = KMeans(n_clusters=n, random_state=42).fit_predict(color_palette)
    t1 = time.time()

    print("Execution time: {} s".format(t1 - t0))
    Y = np.array(Y) / 255.
    color_hex = [rgb2hex(rgb) for rgb in Y]

    print("Plotting...")
    plt.scatter(Y[:, 0], Y[:, 1], c=color_hex, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.show()


def create_number_paint_image(input_image):
    smoothed = smooth_image(input_image)
    enhanced = enh_hist(smoothed)
    median1 = median_blur(enhanced)

    orig_color_palette = calculate_color_palette(median1)
    digitized = kmeans_cluster_colors(orig_color_palette)

    # digitzed = ocv_kmeans_cluster_colors(median1)

    # scatter_color_map(orig_color_palette)

    # median2 = median_blur(digitized)
    # reduced_color_palette = calculate_color_palette(median2)

# ----------------
# img_src = "D:\\Testdata\\paris_street.png"
# img_src = "D:\\Testdata\\hokusai_wave.png"
# img_src = "D:\\Testdata\\vangogh_almondflower.png"
img_src = "images/anastasia-taioglou-CTivHyiTbFw-unsplash.jpg"

# plt.figure(figsize=(16,9))
# plt_img = plt.imread(img_src)
# print(plt_img.shape)
# plt.imshow(plt_img)
# plt.show()

if not os.path.exists(img_src):
    print("File not found: {}".format(img_src))
    exit()

img = cv2.imread(img_src)
img = cv2.resize(img, None, None, 0.25, 0.25)
create_number_paint_image(img)
