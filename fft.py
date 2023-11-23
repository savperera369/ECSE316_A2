import argparse
import cv2
import numpy as np
from fourier_transform import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--mode', type=int, default=1)
# parser.add_argument('-i', '--image', type=str, default="moonlanding.png")

# args = parser.parse_args()

if __name__ == "__main__":

    image = cv2.imread("moonlanding.png")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgList = img.tolist()
    numRows = len(imgList[0])

    n=0
    while 2**n <= numRows:
        n = n + 1

    paddingPixels = (2**n) - numRows
    
    modImage = []

    for rowTwoD in imgList:
        for i in range(paddingPixels):
            rowTwoD.append([0,0,0])
        
        modImage.append(rowTwoD)
    
    modFftImage = []
    modBuiltInDft = []

    for row in modImage:
        ft_row = fft_twoD(row)
        modFftImage.append(ft_row)
        dft_result = np.fft.fft2(row)
        modBuiltInDft.append(dft_result.tolist())

    fftImage = np.array(modFftImage)
    fftBuiltInImage = np.array(modBuiltInDft)

    absFFtImage = np.abs(fftImage)
    absBuiltInDft = np.abs(fftBuiltInImage)

    print(np.allclose(absFFtImage, absBuiltInDft, rtol=1e-5, atol=1e-8))

    # Create a 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first image without LogNorm
    im1 = axes[0].imshow(img, cmap='viridis')
    axes[0].set_title('Image 1')

    # Plot the second image with LogNorm for each RGB channel
    for i in range(3):  # Loop over RGB channels
        im2 = axes[1].imshow(absFFtImage[:,:,i], cmap='viridis', alpha=0.3)  # Use alpha for overlapping channels
    axes[1].set_title('Image 2')

    # Add colorbars
    cb1 = plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
    cb2 = plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

    # Adjust layout to prevent clipping of colorbars
    plt.tight_layout()

    # Show the plot
    plt.show()
