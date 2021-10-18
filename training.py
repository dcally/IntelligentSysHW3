import numpy as np
import matplotlib.pyplot as plt

def train():
    # data = np.random.random((10, 10))
    # plt.subplot(2, 2, 1)
    # plt.imshow(data, interpolation='nearest', cmap="rainbow")
    # plt.title('HeatMap Using Matplotlib Library')
    # plt.show()

    # image_size = 28  # width and length
    # no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
    # image_pixels = image_size * image_size
    # data_path = "../"
    # train_data = np.loadtxt(data_path + "MNISTnumImages5000_balanced.txt", delimiter=",")
    # # test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")
    #
    # print(train_data)

    # with open("MNISTnumImages5000_balanced.txt", "r") as file:
    #     for line in file:
    #         line.split("\n")[1]
    #
    # print(file)

    image_size = 28  # width and length
    no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = "data/mnist/"
    train_data = np.loadtxt("MNISTnumImages5000_balanced.txt")
    # test_data = np.loadtxt(data_path + "mnist_test.csv",
    #                        delimiter=",")
    # test_data[:10]
    print(train_data.shape)
    for i in range(10):
        img = train_data[i].reshape((28, 28))
        plt.imshow(img, cmap="Greys")
        plt.show()
