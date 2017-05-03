import numpy as np
import pickle
import time

def meanLogProb(xa, xb, sigma):
    '''
    The Mean of the Log Probability
    '''
    k, d = xa.shape
    m = xb.shape[0]
    sigma_m = 0.
    for i in np.arange(m):
        sigma_d = np.sum(-(xb[i,:] - xa)**2, axis=1)
        log_p = np.log(1/k) + logsumexp(sigma_d / (2*sigma**2) - d/2 * np.log(2*np.pi*sigma**2))
        sigma_m += log_p
    return sigma_m/m

def logsumexp(arr):
    '''
    Log Sum Exp
    '''
    a_max = np.amax(arr)
    s = np.sum(np.exp(arr - a_max))
    return a_max + np.log(s)

def loadData(file):
    '''
    Load Preprocessed Data
    '''
    with open(file, 'rb') as f:
        data_pkl = pickle.load(f)
        train_MNIST = data_pkl['train_MNIST']
        valid_MNIST = data_pkl['valid_MNIST']
        test_MNIST = data_pkl['test_MNIST']
        train_CIFAR = data_pkl['train_CIFAR']
        valid_CIFAR = data_pkl['valid_CIFAR']
        test_CIFAR = data_pkl['test_CIFAR']
        del data_pkl
    return train_MNIST, valid_MNIST, test_MNIST, train_CIFAR, valid_CIFAR, test_CIFAR

def main():
    '''
    Compute the Mean of the Log-Probability
    '''
    filename = 'preprocessed_data.pkl'
    sigma = [.05, .08, .1, .2, .5, 1., 1.5, 2.]
    num_GS = 200

    train_MNIST, valid_MNIST, test_MNIST, train_CIFAR, valid_CIFAR, test_CIFAR = loadData(filename)
    train_CIFAR = train_CIFAR / 255
    valid_CIFAR = valid_CIFAR / 255
    test_CIFAR = test_CIFAR / 255

    print('sigma \tD_MNIST_valid \tD_CIFAR_valid')

    sigma_opt = None
    res = float()

    for s in sigma:
        res_M = meanLogProb(train_MNIST[0:num_GS, :], valid_MNIST[0:num_GS, :], s)
        res_C = meanLogProb(train_CIFAR[0:num_GS, :], valid_CIFAR[0:num_GS, :], s)
        if res_M > res:
            res = res_M
            sigma_opt = s
        print('{:>5.2f} {:>15.3f} {:>15.3f}'.format(s, res_M, res_C))

    print('\nThe optimal value of sigma is', sigma_opt)

    print('\nComputing the Mean of the Log-Probability for MNIST test data...')
    start = time.time()
    res_M = meanLogProb(train_MNIST, test_MNIST, sigma_opt)
    end = time.time()
    time_M = end - start

    print('\nComputing the Mean of the Log-Probability for CIFAR test data...')
    start = time.time()
    res_C = meanLogProb(train_CIFAR, test_CIFAR, sigma_opt)
    end = time.time()
    time_C = end - start

    print('\nD_MNIST_test \tD_CIFAR_test')
    print('{:>12.3f} {:>15.3f}'.format(res_M, res_C))
    print('{:>10.3f} s {:>13.3f} s'.format(time_M, time_C))

if __name__ == "__main__":
   main()
