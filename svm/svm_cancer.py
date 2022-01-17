import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def execute():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    svc = SVC()
    svc.fit(X_train, y_train)

    print("Accuracy of training set: {:.2f}".format(svc.score(X_train, y_train)))
    print("Accuracy of training set: {:.2f}".format(svc.score(X_test, y_test)))

    fig = plt.figure()
    plt.boxplot(X_train,manage_ticks=False)
    plt.yscale("symlog")
    plt.xlabel("Feature index")
    plt.ylabel("Feature magnitude")
    fig.savefig("svm/svm_cancer.png")
