import numpy as np


def _prepare_inputs(A, B):
    """将输入整理为二维数组，并检查格式。"""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # 单个样本也变成一行，例如[1, 2]变成[[1, 2]]
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("输入必须是单个样本或二维样本表。")

    if A.size == 0 or B.size == 0:
        raise ValueError("输入数据不能为空。")

    if A.shape[1] != B.shape[1]:
        raise ValueError("两组数据的特征数量必须一致。")

    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError("输入不能包含缺失值或无穷大。")

    return A, B


def euclidean_distance(A, B):
    """计算A中每个样本到B中每个样本的欧氏距离。"""
    A, B = _prepare_inputs(A, B)

    differences = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(differences ** 2, axis=2))


def manhattan_distance(A, B):
    """计算A中每个样本到B中每个样本的曼哈顿距离。"""
    A, B = _prepare_inputs(A, B)

    differences = A[:, None, :] - B[None, :, :]
    return np.sum(np.abs(differences), axis=2)

def majority_vote(labels):
    """多数投票。labels必须已经按距离从近到远排列。"""
    labels = np.asarray(labels)

    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("投票标签必须是一维非空数组。")

    # 统计每个类别的票数
    classes, counts = np.unique(labels, return_counts=True)

    # 找出票数最多的类别，可能不止一个
    tied_classes = classes[counts == counts.max()]

    # 从最近的邻居开始检查，解决平票
    for label in labels:
        if label in tied_classes:
            return label


class KNNClassifier:
    def __init__(self, k=3, metric="euclidean"):
        if (
            not isinstance(k, (int, np.integer))
            or isinstance(k, bool)
            or k < 1
        ):
            raise ValueError("k必须是正整数。")

        if metric not in ("euclidean", "manhattan"):
            raise ValueError("距离类型只能是euclidean或manhattan。")

        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """保存训练样本及其类别。"""
        X, _ = _prepare_inputs(X, X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("类别标签必须是一维数组。")

        if len(X) != len(y):
            raise ValueError("样本数量与标签数量不一致。")

        if self.k > len(X):
            raise ValueError("k不能大于训练样本数。")

        self.X_train = X.copy()
        self.y_train = y.copy()

        return self

    def kneighbors(self, X):
        """返回每个待测样本的近邻距离和训练集行号。"""
        if self.X_train is None:
            raise ValueError("请先调用fit提供训练数据。")

        if self.metric == "euclidean":
            distances = euclidean_distance(X, self.X_train)
        else:
            distances = manhattan_distance(X, self.X_train)

        # 对每行距离排序，取最近的k个训练样本
        # 距离相同时保留训练样本原来的顺序
        indices = np.argsort(
            distances, axis=1, kind="stable"
        )[:, :self.k]

        neighbor_distances = np.take_along_axis(
            distances, indices, axis=1
        )

        return neighbor_distances, indices

    def predict(self, X):
        """找近邻，再通过投票预测类别。"""
        _, indices = self.kneighbors(X)

        predictions = []

        for neighbor_indices in indices:
            neighbor_labels = self.y_train[neighbor_indices]
            prediction = majority_vote(neighbor_labels)
            predictions.append(prediction)

        return np.asarray(predictions)

def euclidean_distance_loop(A, B):
    """用Python循环计算每一对样本的欧氏距离。"""
    A, B = _prepare_inputs(A, B)

    result = np.zeros((len(A), len(B)))

    for i in range(len(A)):
        for j in range(len(B)):
            squared_sum = 0.0

            for feature in range(A.shape[1]):
                difference = A[i, feature] - B[j, feature]
                squared_sum += difference ** 2

            result[i, j] = np.sqrt(squared_sum)

    return result

class KNNRegressor(KNNClassifier):
    """复用分类器的距离计算与近邻搜索，改用均值预测。"""

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)

        if y.ndim != 1 or not np.isfinite(y).all():
            raise ValueError("回归目标必须是一维数组，且不能包含缺失或无穷大。")

        return super().fit(X, y)

    def predict(self, X):
        _, indices = self.kneighbors(X)

        # 每行是一个待测样本对应的k个邻居目标值
        neighbor_values = self.y_train[indices]

        # 沿着每一行求平均
        return neighbor_values.mean(axis=1)