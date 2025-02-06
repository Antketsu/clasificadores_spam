import codecs
import utils
import numpy as np
import logistic_reg
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import nn
import torch
import torch.nn
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def parse_dir(dir, num_files, vocab):
    X = np.empty((0, len(vocab) + 1), dtype=np.float32)
    for i in range(1, num_files + 1):
        email_contents = codecs.open('{0}/{1:04d}.txt'.format(dir, i), 'r', encoding='utf-8', errors='ignore').read()
        email = utils.email2TokenList(email_contents)
        row = np.zeros(len(vocab) + 1, dtype=np.float32)
        for word in email:
            if word in vocab:
                row[vocab[word] - 1] = 1
        X = np.vstack((X,row))
    X[:,-1] = 1 if dir == 'data_spam/spam' else 0
    return X

def build_partition(spam, easy_ham, hard_ham, ini_spam, fin_spam, ini_easy, fin_easy, ini_hard, fin_hard):
    np.random.seed(10)
    part = np.vstack((spam[ini_spam:fin_spam,:], hard_ham[ini_hard:fin_hard,:]))
    part = np.vstack((part,easy_ham[ini_easy:fin_easy,:]))
    np.random.shuffle(part)
    return part[:,:-1], part[:,-1]
        
def try_logistic(X_train, X_test, y_train, y_test):
   """
   X_train: (m,n)
   y_train: (m,)
   X_test: (a,b)
   y_test: (a,)
   """
   print("Testing logistic Regression")
   w, b = logistic_reg.train(X_train, y_train, np.zeros(X_train.shape[1]), 0, 0.9, 1000, 0.1)
   p = logistic_reg.predict(X_test, w,b,)
   print(f"Accuracy: {np.mean(p == y_test)}")
   print("-------------------------------------------") 

def try_nn(X_train, y_train, hidden_layer_size, lambda_):
    INIT_EPSILON = 0.12
    INPUT_LAYER_SIZE = 1899
    OUTPUT_LAYER_SIZE = 2
    theta1_in = np.random.random((hidden_layer_size,INPUT_LAYER_SIZE + 1))*(2*INIT_EPSILON) - INIT_EPSILON
    theta2_in = np.random.random((OUTPUT_LAYER_SIZE,hidden_layer_size + 1))*(2*INIT_EPSILON) - INIT_EPSILON
    theta1, theta2, J = nn.gradient_descent(X_train, y_train,theta1_in, theta2_in, 0.95, 1000, lambda_)
    return theta1, theta2

def nn_study(X_train, X_test, X_cv, y_train, y_test, y_cv):
    print("Testing P5 nn")
    m = X_train.shape[0]
    y_train_one_hot = np.zeros((m, 2), dtype = int)
    y_train_one_hot[np.arange(m), y_train.astype(int)] = 1
    lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
    hidden_sizes = [5, 10, 15, 20, 25]
    best_thetas = None
    best_acu, best_hs, best_lambda = (0,0,0)
    for hs in hidden_sizes:
        print(f"Trying hidden size {hs}")
        for lambda_ in lambdas:
            theta1, theta2 = try_nn(X_train, y_train_one_hot, hs, lambda_)
            acu = accuracy_score(y_cv, nn.predict(theta1, theta2, X_cv))
            if acu > best_acu:
                best_thetas = (theta1, theta2)
                best_acu = acu
                best_hs, best_lambda = (hs, lambda_)
    print(f"Choosen parameters are hidden layer size = {best_hs} and lambda = {best_lambda}")
    print(f"CV accuracy = {accuracy_score(y_cv, nn.predict(best_thetas[0], best_thetas[1], X_cv))}")
    print(f"Test accuracy = {accuracy_score(y_test, nn.predict(best_thetas[0], best_thetas[1], X_test))}")
    print("-------------------------------------------") 


def try_svm(X_train, y_train, C, sigma):
    svm = SVC(kernel='rbf', C = C, gamma= 1 / (2 * sigma**2))
    svm.fit(X_train, y_train.ravel())
    return svm


def svm_study(X_train, X_test, X_cv, y_train, y_test, y_cv):
    print("Testing svm")
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    best_svm = None
    best_C, best_sigma = (0,0)
    best_acu = 0
    for C in values:
        print(f"Trying C {C}")
        for sigma in values:
            svm = try_svm(X_train, y_train, C, sigma)
            acu = accuracy_score(y_cv, svm.predict(X_cv))
            if acu > best_acu:
                best_svm, best_C, best_sigma = (svm, C, sigma)
                best_acu = acu
    print(f"Choosen parameters are C = {best_C} and sigma = {best_sigma}")
    print(f"CV accuracy = {accuracy_score(y_cv, best_svm.predict(X_cv))}")
    print(f"Test accuracy = {accuracy_score(y_test, best_svm.predict(X_test))}")
    print("-------------------------------------------") 
        

def try_torch(X_train_, y_train_, hidden_layer_size, regu_param):
    torch.manual_seed(1)
    INPUT_LAYER_SIZE = X_train_.shape[1]
    OUTPUT_LAYER_SIZE = 2
    model = torch.nn.Sequential(
        torch.nn.Linear(INPUT_LAYER_SIZE, hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer_size, OUTPUT_LAYER_SIZE)
    )

    learning_rate = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=regu_param)
    num_epochs = 1000
    for epoch in range(num_epochs):
        pred = model(X_train_)
        loss = loss_fn(pred, y_train_.long())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model

def torch_study(X_train, X_test, X_cv, y_train, y_test, y_cv):
    print("Testing pytorch nn")
    X_train_ =  torch.tensor(X_train, dtype = torch.float32)
    y_train_ =  torch.tensor(y_train, dtype = torch.float32)
    X_test_ =  torch.tensor(X_test, dtype = torch.float32)
    y_test_ =  torch.tensor(y_test, dtype = torch.float32)
    X_cv_ = torch.tensor(X_cv, dtype = torch.float32)
    y_cv_ = torch.tensor(y_cv, dtype = torch.float32)
    lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
    hidden_sizes = [5, 10, 15, 20, 25]
    max_accuracy = 0
    best_model = None
    best_lambda, best_hs = (0,0) 
    for hs in hidden_sizes:
        print(f"Trying hidden size {hs}")
        for lambda_ in lambdas:
            model = try_torch(X_train_, y_train_, hs, lambda_)
            acu = accuracy_score(y_cv_, torch.argmax(model(X_cv_),dim=1)) 
            if acu > max_accuracy:
                best_model, best_hs, best_lambda = (model, hs, lambda_)
                max_accuracy = acu 

    print(f"Choosen parameters are hidden layer size = {best_hs} and lambda = {best_lambda}")
    print(f"CV accuracy = {accuracy_score(y_cv_, torch.argmax(best_model(X_cv_),dim=1))}")
    print(f"Test accuracy = {accuracy_score(y_test_, torch.argmax(best_model(X_test_),dim=1))}")
    print("-------------------------------------------") 


def main():
    vocab = utils.getVocabDict()
    easy_ham_size, spam_size, hard_ham_size = (2551, 500, 250)
    easy_ham = parse_dir('data_spam/easy_ham', easy_ham_size, vocab)
    spam = parse_dir('data_spam/spam', spam_size, vocab)
    hard_ham = parse_dir('data_spam/hard_ham', hard_ham_size, vocab)

    print('Data loaded')

    X_train, y_train = build_partition(spam, easy_ham, hard_ham, 0, 300, 0,2151, 0,200) 
    X_test, y_test = build_partition(spam, easy_ham, hard_ham, 300,400, 2151,2351, 200,225) 
    X_cv, y_cv = build_partition(spam, easy_ham, hard_ham, 400,500, 2351,2551, 225,250) 
    
    print('Partition done')
    try_logistic(X_train, X_test, y_train, y_test)
    nn_study(X_train, X_test, X_cv, y_train, y_test, y_cv)
    torch_study(X_train, X_test, X_cv, y_train, y_test, y_cv)
    svm_study(X_train, X_test, X_cv, y_train, y_test, y_cv)


main()