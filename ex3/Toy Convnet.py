from scipy import signal
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

# define the functions we would like to predict:
num_of_functions = 3
size = 4
W = 4 * (np.random.random((size, size)) - 0.5)
y_funcs = {
    0: lambda x: np.sum(np.dot(x, W), axis=1),
    1: lambda x: np.max(x, axis=1),
    2: lambda x: np.log(np.sum(np.exp(np.dot(x, W)), axis=1))
}


def learn_linear(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a linear model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (w, training_loss, test_loss):
            w: the weights of the linear model
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    w = {func_id: np.zeros(size) for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):
        print("function number {}".format(func_id))
        for _ in range(iterations):

            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx,:], Y[func_id]['train'][idx]
            x_t, y_t = X['test'], Y[func_id]['test']

            # calculate the loss and derivatives:
            p = np.dot(x, w[func_id])
            p_t = np.dot(x_t, w[func_id])

            loss = np.dot((p-y), (p-y)) + (lamb/2)*(np.linalg.norm(w[func_id]) ** 2)
            iteration_test_loss = np.dot((p_t-y_t), (p_t-y_t)) + (lamb/2)*(np.linalg.norm(w[func_id]) ** 2)
            # print("test loss : {}".format(iteration_test_loss))

            dl_dw = np.dot(2*(p-y), x) + lamb * (w[func_id])

            # update the model and record the loss:
            w[func_id] -= learning_rate * dl_dw
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    return w, training_loss, test_loss


def forward(cnn_model, x):
    """
    Given the CNN model, fill up a dictionary with the forward pass values.
    :param cnn_model: the model
    :param x: the input of the CNN
    :return: a dictionary with the forward pass values
    """
    fwd = {}
    fwd['x'] = x
    fwd['o1'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, cnn_model['w1'], mode='same'))
    fwd['o2'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, cnn_model['w2'], mode='same'))

    m1 =     np.max(fwd['o1'][:, :2], axis=1)
    m1_idx = np.argmax(fwd['o1'][:, :2], axis=1)
    m2 =     np.max(fwd['o1'][:, 2:], axis=1)
    m2_idx = np.argmax(fwd['o1'][:, 2:], axis=1)
    m3 =     np.max(fwd['o2'][:, :2], axis=1)
    m3_idx = np.argmax(fwd['o2'][:, :2], axis=1)
    m4 =     np.max(fwd['o2'][:, 2:], axis=1)
    m4_idx = np.argmax(fwd['o2'][:, 2:], axis=1)

    fwd['m'] = np.vstack((np.vstack((m1, m2)), np.vstack((m3, m4)))).T
    fwd['m_argmax'] = np.vstack((np.vstack((m1_idx, m2_idx)), np.vstack((m3_idx, m4_idx))))
    fwd['p'] = np.dot(fwd['m'], cnn_model['u'])

    return fwd


def backprop(model, y, fwd, batch_size, lamb):
    """
    given the forward pass values and the labels, calculate the derivatives
    using the back propagation algorithm.
    :param model: the model
    :param y: the labels
    :param fwd: the forward pass values
    :param batch_size: the batch size
    :return: a tuple of (dl_dw1, dl_dw2, dl_du)
            dl_dw1: the derivative of the w1 vector
            dl_dw2: the derivative of the w2 vector
            dl_du: the derivative of the u vector
    """
    # do/dw1 & do/dw2
    do_dw = np.zeros((batch_size, 4, 3))
    do_dw[:, 1:, 0] = fwd['x'][:, :-1]
    do_dw[:, :, 1] = fwd['x']
    do_dw[:, :-1, 2] = fwd['x'][:, 1:]

    # dr/do1
    dr_do1 = np.zeros((batch_size, 4, 4))
    temp = (fwd['o1'] > 0).astype(int)
    dr_do1[:, 0, 0] = temp[:, 0]
    dr_do1[:, 1, 1] = temp[:, 1]
    dr_do1[:, 2, 2] = temp[:, 2]
    dr_do1[:, 3, 3] = temp[:, 3]

    # dr/do2
    dr_do2 = np.zeros((batch_size, 4, 4))
    temp = (fwd['o2'] > 0).astype(int)
    dr_do2[:, 0, 0] = temp[:, 0]
    dr_do2[:, 1, 1] = temp[:, 1]
    dr_do2[:, 2, 2] = temp[:, 2]
    dr_do2[:, 3, 3] = temp[:, 3]

    # dm/dr1
    dm_dr1 = np.zeros((batch_size, 2, 4))
    dm_dr1[:, 0, 0] = (fwd['m_argmax'][0] == 0).astype(int)
    dm_dr1[:, 0, 1] = (fwd['m_argmax'][0] == 1).astype(int)
    dm_dr1[:, 1, 2] = (fwd['m_argmax'][1] == 0).astype(int)
    dm_dr1[:, 1, 3] = (fwd['m_argmax'][1] == 1).astype(int)

    # dm/dr2
    dm_dr2 = np.zeros((batch_size, 2, 4))
    dm_dr2[:, 0, 0] = (fwd['m_argmax'][2] == 0).astype(int)
    dm_dr2[:, 0, 1] = (fwd['m_argmax'][2] == 1).astype(int)
    dm_dr2[:, 1, 2] = (fwd['m_argmax'][3] == 0).astype(int)
    dm_dr2[:, 1, 3] = (fwd['m_argmax'][3] == 1).astype(int)

    # dp/dm1
    dp_dm1 = model['u'][:2].T

    # dp/dm2
    dp_dm2 = model['u'][2:].T

    # dl/dp
    dl_dp = 2 * (fwd['p'] - y)

    dl_dw1 = []
    dl_dw2 = []
    for i in range(batch_size):
        t1 = np.dot(dr_do1[i], do_dw[i])
        t2 = np.dot(dm_dr1[i], t1)
        t3 = np.dot(dp_dm1, t2)
        t4 = dl_dp[i] * t3
        dl_dw1.append(t4)

        t1 = np.dot(dr_do2[i], do_dw[i])
        t2 = np.dot(dm_dr2[i], t1)
        t3 = np.dot(dp_dm2, t2)
        t4 = dl_dp[i] * t3
        dl_dw2.append(t4)

    dl_dw1 = (np.array(dl_dw1).T + (lamb * model['w1'])).T
    dl_dw2 = (np.array(dl_dw2).T + (lamb * model['w2'])).T
    dl_du = np.multiply(fwd['m'].T, dl_dp)

    dl_dw1 = np.sum(dl_dw1, axis=0).reshape((3, 1))
    dl_dw2 = np.sum(dl_dw2, axis=0).reshape((3, 1))
    dl_du = np.sum(dl_du, axis=1)

    return (dl_dw1, dl_dw2, dl_du)


def learn_cnn(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a cnn model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (models, training_loss, test_loss):
            models: a model for every function (a dictionary for the parameters)
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    models = {func_id: {} for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):

        # initialize the model:
        models[func_id]['w1'] = np.reshape(4 * np.random.random(3) - 0.5, (3, 1))
        models[func_id]['w2'] = np.reshape(4 * np.random.random(3) - 0.5, (3, 1))
        models[func_id]['u'] = 4 * np.random.random(4) - 0.5

        # train the network:
        for _ in range(iterations):

            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx,:], Y[func_id]['train'][idx]

            # calculate the loss and derivatives using back propagation:
            fwd = forward(models[func_id], x)
            loss = np.dot((fwd['p']-y), (fwd['p']-y)) \
                   + (lamb/2)*((np.linalg.norm(models[func_id]['w1']) ** 2)
                               + (np.linalg.norm(models[func_id]['w2']) ** 2))

            dl_dw1, dl_dw2, dl_du = backprop(models[func_id], y, fwd, batch_size, lamb)

            # record the test loss before updating the model:
            test_fwd = forward(models[func_id], X['test'])
            iteration_test_loss = np.dot((test_fwd['p']-Y[func_id]['test']), (test_fwd['p']-Y[func_id]['test'])) \
                   + (lamb/2)*((np.linalg.norm(models[func_id]['w1']) ** 2)
                               + (np.linalg.norm(models[func_id]['w2']) ** 2))

            # update the model using the derivatives and record the loss:

            models[func_id]['w1'] -= learning_rate * dl_dw1
            models[func_id]['w2'] -= learning_rate * dl_dw2
            models[func_id]['u'] -= learning_rate * dl_du
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)
            print("test loss : {}".format(iteration_test_loss))

    return models, training_loss, test_loss


def grapichalComparision(data):

    for i in range(len(data)):
        plt.plot(data[i], label='y{}'.format(i))

    plt.grid('on')
    plt.title('Test loss vs. iterations')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # generate the training and test data, adding some noise:
    X = dict(train=5 * (np.random.random((1000, size)) - .5),
             test=5 * (np.random.random((200, size)) - .5))
    Y = {i: {
        'train': y_funcs[i](X['train']) * (
        1 + np.random.randn(X['train'].shape[0]) * .01),
        'test': y_funcs[i](X['test']) * (
        1 + np.random.randn(X['test'].shape[0]) * .01)}
         for i in range(len(y_funcs))}

    # ****************************************************************
    # ********************* LINEAR MODEL *****************************
    # w, training_loss, test_loss = learn_linear(X, Y, 50, 0.2, 200, 0.001)
    # grapichalComparision(test_loss)

    # ****************************************************************
    # ********************* TOY CNN MODEL *****************************
    # models, training_loss, test_loss = learn_cnn(X, Y, 50, 0.5, 200, 0.00001)
    # grapichalComparision(test_loss)

