import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, matrix_normal
from scipy.misc import logsumexp
import pickle
from skimage.util import view_as_windows as viewW
import time

PLOT_GRAPHS = False
DEBUG = False

def DEBUG_MSG(str):
    if(DEBUG):
        print(str)

def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(patches[:,i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                        window[0] * window[1]).T[:, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    DEBUG_MSG("start denoise...")
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    DEBUG_MSG("split the image into columns and denoise the columns:")
    noisy_patches = im2col(Y, patch_size)
    DEBUG_MSG("call to denoise func")
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    DEBUG_MSG("reshape the denoised columns into a picture:")
    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    DEBUG_MSG("make the image noisy:")
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
        noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    DEBUG_MSG("denoise the image:")
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    DEBUG_MSG("calculate the MSE for each noise range:")
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    """
    def __init__(self, cov, mix):
        self.cov = cov
        self.mix = mix


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """
    def __init__(self, P, vars, mix):
        self.P = P
        self.vars = vars
        self.mix = mix


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """
    mvn = multivariate_normal(model.mean, model.cov)
    return np.sum(mvn.logpdf(X.T))


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """
    dim, n = np.shape(X)
    k = len(model.mix)
    mean = np.zeros(dim)
    mix_p = model.mix
    ll = np.zeros(n)

    for y in range(k):
        mod = multivariate_normal(mean, model.cov[y])
        ll += mod.pdf(X.T) * mix_p[y]
    ll = np.sum(np.log(ll))
    return ll


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """
    S = np.dot(model.P.T, X)
    d, k = np.shape(model.vars)
    mean = 0
    ll = 0
    for i in range(d):
        cov = model.vars[i]
        mix = model.mix[i]
        x = S[i]
        row_ll = 0
        for y in range(k):
            mod = multivariate_normal(mean, cov[y])
            row_ll += mod.pdf(x) * mix[y]
        ll += np.sum(np.log(row_ll))

    return ll


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    dim, n = np.shape(X)
    mean = X.sum(1) / n
    cov = np.zeros((dim, dim))
    for j in range(n):
        q = X[:, j] - mean
        cov += np.outer(q, q.T)
    cov /= n
    return MVN_Model(mean, cov)


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.
    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """
    #****** initialization **********
    d, n = np.shape(X) #n- number of samples.
    mean = np.zeros(d)
    # find the cov matrix of the data:
    cov = np.zeros((d, d))
    mu_hat = np.sum(X, axis=1) / n
    for i in range(n):
        temp = X[:, i] - mu_hat
        cov += np.outer(temp, temp.T)
    cov /= n
    cov_inv = np.linalg.pinv(cov)
    # r values for the k different cov matrices
    r = (np.arange(k) + (k/2))/10
    DEBUG_MSG("init r:" + str(r))
    mix_p = r / 2
    mix_p /= np.sum(mix_p)
    DEBUG_MSG("init mix values:" + str(mix_p))
    c = np.zeros((k, n)) # c contains the probabilities for each patch
    ll = [] # loglikelihood array
    # ****** EM algorithm **********
    j = 0
    epsilon = 0
    while(True):
        # build the c matrix (log scale)
        for y in range(k):
            pi_log = np.log(mix_p[y])
            mod = multivariate_normal(mean, cov * r[y], allow_singular=True)
            c[y] = mod.logpdf(X.T) + pi_log

        ll.append(np.sum(logsumexp(c, axis=0)))
        DEBUG_MSG("ll = " + str(ll[-1]))
        c = normalize_log_likelihoods(c)
        c = np.exp(c) # return to regular scale
        # update the r and the mix prob values:
        temp = np.diag(np.dot(X.T, np.dot(cov_inv, X)))
        for y in range(k):
            r[y] = np.nan_to_num(np.sum(c[y] * temp) / (d * np.sum(c[y])))
            mix_p[y] = np.sum(c[y]) / n
        DEBUG_MSG("r: " + str(r))
        DEBUG_MSG("p: " + str(mix_p))

        if (j > 0): # Conditions of convergence 0.1%
            epsilon = np.abs(0.001 * ll[j-1])
            if (ll[j] - ll[j - 1] < epsilon):
                break
        j += 1
    # ****** build the final model and plot the LL**********
    if(PLOT_GRAPHS):
        plt.plot(ll)
        plt.ylabel('loglikelihood')
        plt.xlabel('iterations')
        plt.grid(True)
        plt.show()
    cov_mat = np.outer(r, cov).reshape((k, d, d))
    return GSM_Model(cov_mat, mix_p)


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """
    # calculate the cov
    d, n = np.shape(X)
    mu = np.sum(X, axis=1) / n
    cov = np.zeros((d, d))
    for j in range(n):
        q = X[:, j] - mu
        cov += np.outer(q.T, q)
    cov /= n
    # 1. find P :
    w, p = np.linalg.eig(cov)
    # 2. transform to S domain :
    S = np.dot(p.T, X)
    # 3. EM for mvn version :
    vars = np.zeros((d, k))
    mix_prob = np.zeros((d, k))
    for i in range(d):
        x_row = S[i]
        # EM in one dim:
        cov_x, mix = Learn_1D_GSM(x_row, k)
        vars[i] = cov_x
        mix_prob[i] = mix

    if(PLOT_GRAPHS):
        plt.grid(True)
        plt.show()
    return ICA_Model(p, vars, mix_prob)


def Learn_1D_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X, 1 dimension array using EM.
    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.
    :param X: a N data array, where N is the number of samples.
    :param k: The number of components of the GSM model.
    :return: tuple, cov array and mix probabilities.
    """
    n = len(X)
    # EM in one dim:
    cov = np.cov(X)
    mu = 0
    r_arr = (np.random.rand(k) + 1)
    mix_p = np.ones(k) / k
    c = np.zeros((k, n))
    epsilon = 0
    ll = []
    i = 0
    while (True):
        DEBUG_MSG("cov: " + str(r_arr))
        for y in range(k):
            c[y] = multivariate_normal.logpdf(X, mean=mu, cov=cov * r_arr[y]) + np.log(mix_p[y])

        ll.append(np.sum(logsumexp(c, axis=0)))
        DEBUG_MSG("ll :" + str(ll[-1]))
        c = normalize_log_likelihoods(c)
        c_sum = np.exp(logsumexp(c, axis=1))
        mix_p = c_sum / n
        DEBUG_MSG("mix: " +str(mix_p))
        for y in range(k):
            r_arr[y] = np.exp(logsumexp(c[y], b=(np.square(X) / cov))) / c_sum[y]

        if (i > 0):  # Conditions of convergence 0.1%
            epsilon = np.abs(0.001 * ll[i-1])
            if ((ll[i] - ll[i - 1]) < epsilon):
                break
        i += 1
    if(PLOT_GRAPHS):
        plt.plot(ll)
        plt.ylabel('loglikelihood')
        plt.xlabel('iterations')

    return r_arr*cov, mix_p


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    dim, n = np.shape(Y)
    sigma_i = np.linalg.pinv(mvn_model.cov)
    # Weiner filter :
    q_1 = np.linalg.pinv(sigma_i + np.eye(dim)/np.square(noise_std))
    q_2 = np.dot(sigma_i, mvn_model.mean)
    q_2 = np.repeat(q_2, n, axis=0).reshape((dim, n))
    q_2 += (Y / np.square(noise_std))

    X_hat = np.dot(q_1, q_2)
    return X_hat


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.
    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.
    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    d, n = np.shape(Y)
    mean = np.zeros(d)
    cov = gsm_model.cov
    k, d, d = np.shape(cov)
    c = np.zeros((k, n))
    x_hat = np.zeros((d, n))
    # calculate matrix of the data probabilities
    for y in range(k):
        pi_log = np.log(gsm_model.mix[y])
        mod = multivariate_normal(mean, cov[y] + (np.eye(64) * noise_std), allow_singular=True)
        c[y] = mod.logpdf(Y.T) + pi_log

    c = normalize_log_likelihoods(c)
    c = np.exp(c)
    # apply mvn on each patch
    for y in range(k):
        mv = MVN_Model(mean, cov[y])
        x_hat += (MVN_Denoise(Y, mv, noise_std) * c[y])
    return x_hat


def GSM_1D_Denoise(Y, cov, mix, noise_std):
    """
    Denoise Y (1 dimension samples), assuming a GSM model and gaussian white noise.
    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.
    :param Y: a N data array, where N is the dimension, and M is the number of noisy samples.
    :param cov: The GSM cov values.
    :param mix: The GSM mix probabilities.
    :param noise_std: The standard deviation of the noise.
    :return: a N array of denoised samples.
    """
    n = len(Y)
    k = len(cov)
    mean = np.zeros(k)
    c = np.zeros((k, n))
    x_hat = np.zeros(n)
    # calculate matrix of the data probabilities
    for y in range(k):
        pi_log = np.log(mix[y])
        mod = multivariate_normal(mean[y], cov[y] + np.square(noise_std), allow_singular=True)
        c[y] = mod.logpdf(Y.T) + pi_log

    c = normalize_log_likelihoods(c)
    c = np.exp(c)
    # apply mvn on each index
    for y in range(k):
        q_1 = (1/cov[y]) + (1/np.square(noise_std))
        q_2 = (1/np.square(noise_std)) * Y
        x_hat += (c[y] * (1/q_1) * q_2)
    return x_hat


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.
    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.
    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    d, n = np.shape(Y)
    S = np.dot(ica_model.P.T, Y)
    for i in range(d):
        S[i] = GSM_1D_Denoise(S[i], ica_model.vars[i], ica_model.mix[i], noise_std)
    return np.dot(ica_model.P, S)


def MVN_test():
    '''
    This test run MVN learning model over 10,000 images patches, learn the model and test it on
    1600x1200 image for 4 times over 4 different noised levels(Gaussian noise). The test include calculation of
    log-likelihood values, MSE and running time.
    '''
    print("Testing MVN model")
    # Load train and test data
    patch_size = (8, 8)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    with open('test_images.pickle', 'rb') as f:
        test_pictures = pickle.load(f)

    num_p = 10000 #number of patches
    X = sample_patches(train_pictures, psize=patch_size, n=num_p)
    X_test = sample_patches(test_pictures, psize=patch_size, n=num_p)
    test_pictures = grayscale_and_standardize(test_pictures)
    mvn_model = learn_MVN(X)
    ll_train = MVN_log_likelihood(X, mvn_model)
    ll_test = MVN_log_likelihood(X_test, mvn_model)
    print("log-likelihood train : ", ll_train)
    print("log-likelihood test  : ", ll_test)
    image = test_pictures[0]
    start = time.time()
    test_denoising(image, mvn_model, MVN_Denoise)
    end = time.time()
    print("Denoising time for one image of 1600x1200 pixels : ", int((end-start)/4), "sec")


def GSM_test(k):
    '''
    This test run GSM learning model over 10,000 images patches, learn the model and test it on
    1600x1200 image for 4 times over 4 different noised levels(Gaussian noise). The test include calculation of
    log-likelihood values, MSE and running time.
    :param k: The number of components of the GSM model.
    '''
    print("Testing GSM model")
    # Load train and test data
    patch_size = (8, 8)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    with open('test_images.pickle', 'rb') as f:
        test_pictures = pickle.load(f)

    num_p = 10000 #number of patches
    X = sample_patches(train_pictures, psize=patch_size, n=num_p)
    X_test = sample_patches(test_pictures, psize=patch_size, n=num_p)
    test_pictures = grayscale_and_standardize(test_pictures)
    gsm_model = learn_GSM(X, k)
    ll_train = GSM_log_likelihood(X, gsm_model)
    ll_test = GSM_log_likelihood(X_test, gsm_model)
    print("log-likelihood train : ", ll_train)
    print("log-likelihood test  : ", ll_test)
    image = test_pictures[0]
    start = time.time()
    test_denoising(image, gsm_model, GSM_Denoise)
    end = time.time()
    print("Denoising time for one image of 1600x1200 pixels : ", int((end-start)/4), "sec")


def ICA_test(k):
    '''
    This test run ICA learning model over 10,000 images patches, learn the model and test it on
    1600x1200 image for 4 times over 4 different noised levels(Gaussian noise). The test include calculation of
    log-likelihood values, MSE and running time.
    :param k: The number of components of the ICA model.
    '''
    print("Testing ICA model")
    # Load train and test data
    patch_size = (8, 8)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    with open('test_images.pickle', 'rb') as f:
        test_pictures = pickle.load(f)

    num_p = 10000 #number of patches
    X = sample_patches(train_pictures, psize=patch_size, n=num_p)
    X_test = sample_patches(test_pictures, psize=patch_size, n=num_p)
    test_pictures = grayscale_and_standardize(test_pictures)
    ica_model = learn_ICA(X, 10)
    ll_train = ICA_log_likelihood(X, ica_model)
    ll_test = ICA_log_likelihood(X_test, ica_model)
    print("log-likelihood train : ", ll_train)
    print("log-likelihood test  : ", ll_test)
    image = test_pictures[0]
    start = time.time()
    test_denoising(image, ica_model, ICA_Denoise)
    end = time.time()
    print("Denoising time for one image of 1600x1200 pixels : ", int((end - start) / 4), "sec")


if __name__ == '__main__':
    # 3 test for each model including learning and denoising.
    MVN_test()
    GSM_test(4)
    ICA_test(4)








