import mglearn.plots
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC


def execute():
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    X_new = np.hstack([X, X[:, 1:] ** 2])

    linear_svm_3d = LinearSVC().fit(X_new, y)
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

    fig = plt.figure()
    ax = Axes3D(fig, elev=-152, azim=-26)
    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

    mask = y == 0
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
    ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_zlabel("Feature1 ** 2")
    fig.savefig('svm/svm.png')
