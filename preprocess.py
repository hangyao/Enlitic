import numpy as np
import pickle
import matplotlib.pyplot as plt

def shuffleData(dataset, n=10000):
    '''
    Shuffle Data
    '''
    numData = dataset.shape[0]
    index = np.random.permutation(numData)
    train_x = dataset[index][0:n, :]
    valid_x = dataset[index][n:2*n, :]
    return train_x, valid_x

def loadCIFAR(file):
    '''
    Load CIFAR100 Data
    '''
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data[b'data']

def loadMNIST(file):
    '''
    Load MNIST Data
    '''
    with open(file, 'rb') as fo:
        train, _, test = pickle.load(fo, encoding='latin1')
    return train[0], test[0]

def saveData(file, train_M, valid_M, test_M, train_C, valid_C, test_C):
    '''
    Save Data
    '''
    f = open(file, 'wb')
    save = {
      'train_MNIST': train_M,
      'valid_MNIST': valid_M,
      'test_MNIST': test_M,
      'train_CIFAR': train_C,
      'valid_CIFAR': valid_C,
      'test_CIFAR': test_C
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print('Preprocessed data saved in', file)

def visualize(data, ch=3, num=20):
    '''
    Visualization
    '''
    data = data[0:num**2, :]
    imsz = int(np.sqrt(data.shape[1] / ch))
    if ch == 3:
        img = data.reshape(num**2, ch, imsz**2).transpose(1, 0, 2)
        img = img.reshape(ch, num, num, imsz, imsz).swapaxes(2, 3)
        img = img.reshape(ch, num*imsz, num*imsz).transpose(1, 2, 0)
        cmap = None
    elif ch == 1:
        img = data.reshape(num, num, imsz, imsz).swapaxes(1, 2)
        img = img.reshape(num*imsz, num*imsz)
        cmap = 'gray'
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.show()

def main():
    '''
    Preprocessing
    '''
    file_MNIST = 'mnist.pkl'
    files_CIFAR = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    testfile_CIFAR = 'test_batch'
    saveFile = 'preprocessed_data.pkl'

    train_MNIST, test_MNIST = loadMNIST(file_MNIST)

    data1 = loadCIFAR(files_CIFAR[0])
    data2 = loadCIFAR(files_CIFAR[1])
    data3 = loadCIFAR(files_CIFAR[2])
    data4 = loadCIFAR(files_CIFAR[3])
    data5 = loadCIFAR(files_CIFAR[4])
    train_CIFAR = np.concatenate([data1, data2, data3, data4, data5])
    test_CIFAR = loadCIFAR(testfile_CIFAR)

    train_M, valid_M = shuffleData(train_MNIST)
    train_C, valid_C = shuffleData(train_CIFAR)

    visualize(train_M, 1)
    visualize(train_C, 3)

    saveData(saveFile, train_M, valid_M, test_MNIST, train_C, valid_C, test_CIFAR)

if __name__ == "__main__":
   main()
