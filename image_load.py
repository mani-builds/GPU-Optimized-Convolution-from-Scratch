from PIL import Image
import numpy as np

def image_load():
    jpg = Image.open('dog.jpg')
    array = np.asarray(jpg)
    print(f"Fromat {jpg.format}, size: {jpg.size}, mode: {jpg.mode}")
    print("Shape: ", array.shape)
    height, width = jpg.size
    n_channels = array.shape[2]
    print("Reshaped: ",n_channels, height, width)
    array = array.reshape(n_channels, height, width)
    print("Image array", array)

    array_reshaped = array.reshape(-1)

    print("Array reshaped for saving : ", array_reshaped)

    with open('dog_array.txt', 'w') as file:
        np.savetxt(file, array_reshaped, delimiter=',', fmt='%.1f')

    with open('dog_array_shape.txt', 'w') as fname:
        np.savetxt(fname, np.array([n_channels, height, width]), delimiter=',', fmt='%.1f')

    #kernel
    identity_4d = np.zeros((3, 3, 3, 3))
    for i in range(3):
        identity_4d[i, i, 1, 1] = 1.0 # Channel i looks only at Channel i input

    with open('kernel_array_shape.txt', 'w') as file:
        np.savetxt(file, np.array((identity_4d.shape[0], identity_4d.shape[1], identity_4d.shape[2], identity_4d.shape[3])), delimiter=',', fmt='%.1f')

    identity_4d_reshaped = identity_4d.reshape(-1)
    with open('kernel_array.txt', 'w') as file:
        np.savetxt(file, identity_4d_reshaped, delimiter=',', fmt='%.1f')

if __name__ == '__main__':
    image_load()
