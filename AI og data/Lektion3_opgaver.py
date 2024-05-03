import cv2 as cv
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA



img = cv.imread('/Users/jens-jakobskotingerslev/Desktop/AI og data/fruits. copy.jpeg')
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#               OPAGVE 1 

def salt_pepper_noise(image, amount=0.01):
    noisy_image = np.copy(image)
    # Number of pixels to noise
    num_noise = np.ceil(amount * image.size / 2).astype(int)
    
    # Salt noise (white)
    coords = [np.random.randint(0, i, num_noise) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # Pepper noise (black)
    coords = [np.random.randint(0, i, num_noise) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

def mean_filter(image, kernel_size=(3,3)):

    return cv.blur(image, kernel_size)

def median_filter(image, kernel_size=3):

    return cv.medianBlur(image, kernel_size)

# Apply salt and pepper noise
noisy_image = salt_pepper_noise(gray_image, amount=0.04)

# Apply mean filtering
filtered_image_mean = mean_filter(noisy_image)
filtered_image_median = median_filter(noisy_image)

#cv.imshow('Original', img)
#cv.imshow('Grayscale Image', gray_image)
cv.imshow('Noisy Image', noisy_image)
cv.imshow('Filtered Image_mean', filtered_image_mean)
cv.imshow('Filtered Image_median', filtered_image_median)
cv.waitKey(0)
cv.destroyAllWindows()

#               OPGAVE 2 

def add_gaussian_noise(image, mean=0, sigma=15):
    # Generate Gaussian noise
    gaussian = np.random.normal(mean, sigma, image.shape)

    # Add the Gaussian noise to the image
    noisy_image = cv.add(image.astype(np.float32), gaussian.astype(np.float32))

    # Clip the values to stay within valid image range and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def median_filter(image, kernel_size=3):

    return cv.medianBlur(image, kernel_size)

def mean_filter(image, kernel_size=(3,3)):

    return cv.blur(image, kernel_size)

# Apply Gaussian noise
noisy_image = add_gaussian_noise(gray_image)
median_filter_img = median_filter(noisy_image)
mean_filter_img = mean_filter(noisy_image)

# Display the original and noisy images
#cv.imshow('Original Image', gray_image)
cv.imshow('Gaussain noisy Image', noisy_image)
cv.imshow('Gaussian median_filter',median_filter_img)
cv.imshow('Gaussian mean_filter',mean_filter_img)
cv.waitKey(0)
cv.destroyAllWindows()

#               Opgave 3

X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, train_size=1_000, test_size=100
)

rng = np.random.RandomState(0)
noise = rng.normal(scale=0.25, size=X_test.shape)
X_test_noisy = X_test + noise

noise = rng.normal(scale=0.25, size=X_train.shape)
X_train_noisy = X_train + noise

def plot_digits(X, title):
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=24)
    #plt.show()
plot_digits(X_test, "Uncorrupted test images")
plot_digits(
    X_test_noisy, f"Noisy test images\nMSE: {np.mean((X_test - X_test_noisy) ** 2):.2f}"
)

pca = PCA(n_components=100, random_state=42)
kernel_pca = KernelPCA(
    n_components=400,
    kernel="rbf",
    gamma=1e-3,
    fit_inverse_transform=True,
    alpha=5e-3,
    random_state=42,
)

pca.fit(X_train_noisy)
_ = kernel_pca.fit(X_train_noisy)

X_reconstructed_kernel_pca = kernel_pca.inverse_transform(
    kernel_pca.transform(X_test_noisy)
)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test_noisy))

plot_digits(
    X_reconstructed_pca,
    f"PCA reconstruction\nMSE: {np.mean((X_test - X_reconstructed_pca) ** 2):.2f}",
    
)

plot_digits(
    X_reconstructed_kernel_pca,
    (
        "Kernel PCA reconstruction\n"
        f"MSE: {np.mean((X_test - X_reconstructed_kernel_pca) ** 2):.2f}"
    ),
    plt.show()
)
