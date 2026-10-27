import numpy as np
from knn import euclidean_distance, manhattan_distance
from knn import majority_vote, KNNClassifier
from knn import KNNRegressor

def test_classifier():
    # 测试投票：多数票获胜
    assert majority_vote([0, 1, 1]) == 1

    # 测试平票：标签已按距离排序，较近的类别获胜
    assert majority_vote([1, 0]) == 1

    # 测试不能让最近邻压过多数票
    assert majority_vote([0, 1, 1]) == 1

    X_train = [
        [0, 0],
        [0, 2],
        [4, 4],
        [5, 5]
    ]
    y_train = [0, 0, 1, 1]

    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    # 测试近邻索引
    distances, indices = model.kneighbors([[0, 0]])
    np.testing.assert_array_equal(indices, [[0, 1, 2]])

    # 测试近邻距离
    np.testing.assert_allclose(
        distances, [[0, 2, np.sqrt(32)]]
    )

    # 测试多个样本的预测
    predictions = model.predict([
        [0.1, 0.1],
        [4.9, 4.9]
    ])
    np.testing.assert_array_equal(predictions, [0, 1])

    # 测试曼哈顿距离模式也能预测
    model_m = KNNClassifier(k=3, metric="manhattan")
    model_m.fit(X_train, y_train)
    np.testing.assert_array_equal(
        model_m.predict([[0.1, 0.1], [4.9, 4.9]]),
        [0, 1]
    )

    # 测试k超过训练样本数时会报错
    try:
        KNNClassifier(k=5).fit(X_train, y_train)
    except ValueError:
        pass
    else:
        raise AssertionError("k过大时没有报错！")

    # 测试没有fit就预测时会报错
    try:
        KNNClassifier().predict([[0, 0]])
    except ValueError:
        pass
    else:
        raise AssertionError("未fit时没有报错！")

    print("分类器测试通过：投票、近邻、预测和异常处理均正常")

def test_regressor():
    model = KNNRegressor(k=2)
    model.fit([[0], [2], [4]], [0, 20, 40])

    # 输入1，最近的是0和2，目标均值是10
    # 输入3，最近的是2和4，目标均值是30
    prediction = model.predict([[1], [3]])

    np.testing.assert_allclose(prediction, [10, 30])
    print("回归测试通过：近邻均值预测正确")

def main():
    # 测试1：单个样本的欧氏距离
    result = euclidean_distance([1, 2], [4, 6])
    np.testing.assert_allclose(result, [[5]])
    print("测试1通过：欧氏距离为5")

    # 测试2：单个样本的曼哈顿距离
    result = manhattan_distance([1, 2], [4, 6])
    np.testing.assert_allclose(result, [[7]])
    print("测试2通过：曼哈顿距离为7")

    # 测试3：相同样本的距离为0
    result = euclidean_distance([2, 3], [2, 3])
    np.testing.assert_allclose(result, [[0]])
    print("测试3通过：相同样本距离为0")

    # 测试4：多样本欧氏距离
    A = [[0, 0], [3, 4]]
    B = [[0, 0], [6, 8]]

    result = euclidean_distance(A, B)
    np.testing.assert_allclose(result, [[0, 10], [5, 5]])
    print("测试4通过：多样本欧氏距离正确")

    # 测试5：多样本曼哈顿距离
    result = manhattan_distance(A, B)
    np.testing.assert_allclose(result, [[0, 14], [7, 7]])
    print("测试5通过：多样本曼哈顿距离正确")

    # 测试6：一个样本与多个样本计算距离
    result = euclidean_distance([0, 0], B)
    np.testing.assert_allclose(result, [[0, 10]])
    print("测试6通过：单样本到多样本距离正确")

    # 测试7：两种函数都应拒绝不合法输入
    invalid_pairs = [
        ([], [1, 2]),
        ([1, 2], [1, 2, 3]),
        ([float("nan"), 2], [1, 2])
    ]

    for function in [euclidean_distance, manhattan_distance]:
        for A, B in invalid_pairs:
            try:
                function(A, B)
            except ValueError:
                pass
            else:
                raise AssertionError("不合法输入没有触发报错！")

    print("测试7通过：空数据、维度不一致、缺失值均被拒绝")
    print("\n全部距离测试通过！")
    test_classifier()
    test_regressor()

if __name__ == "__main__":
    main()