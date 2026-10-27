from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from timeit import repeat
from knn import euclidean_distance, euclidean_distance_loop
from knn import KNNRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from knn import KNNClassifier

# 固定随机种子，后续划分数据时也统一使用42
SEED = 42
np.random.seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULT_DIR = PROJECT_DIR / "results"

def select_k(X_train, y_train):
    candidates = [1, 3, 5, 7, 9, 11, 13, 15]

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    records = []

    for k in candidates:
        fold_scores = []

        for fit_indices, val_indices in splitter.split(
            X_train, y_train
        ):
            X_fit = X_train[fit_indices]
            X_val = X_train[val_indices]
            y_fit = y_train[fit_indices]
            y_val = y_train[val_indices]

            # 每一折都只根据该折的训练部分计算标准化参数
            mean = X_fit.mean(axis=0)
            std = X_fit.std(axis=0)
            std = np.where(std == 0, 1.0, std)

            X_fit_scaled = (X_fit - mean) / std
            X_val_scaled = (X_val - mean) / std

            model = KNNClassifier(k=k, metric="euclidean")
            model.fit(X_fit_scaled, y_fit)
            prediction = model.predict(X_val_scaled)

            score = np.mean(prediction == y_val)
            fold_scores.append(score)

        records.append({
            "k": k,
            "mean_accuracy": np.mean(fold_scores),
            "std_accuracy": np.std(fold_scores)
        })

    scores = pd.DataFrame(records)

    # 选平均准确率最高的k；完全相同时取候选列表中较小的k
    best_row = scores["mean_accuracy"].idxmax()
    best_k = int(scores.loc[best_row, "k"])

    print("\n5折交叉验证结果：")
    print(scores.to_string(index=False))
    print("选择的k值：", best_k)

    scores.to_csv(
        RESULT_DIR / "k_cv_scores.csv",
        index=False,
        encoding="utf-8"
    )

    # 折线为平均准确率，误差线为5折准确率的标准差
    fig, ax = plt.subplots()
    ax.errorbar(
        scores["k"],
        scores["mean_accuracy"],
        yerr=scores["std_accuracy"],
        marker="o",
        capsize=4,
        label="5-fold mean +/- std"
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("K selection on training data")
    ax.set_xticks(candidates)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "k_accuracy.png", dpi=150)
    plt.close(fig)

    return best_k


def evaluate_predictions(X_test, y_test, y_pred):
    class_names = ["setosa", "versicolor", "virginica"]
    records = []

    # 1. 自己计算每个类别的评价指标
    for label, name in enumerate(class_names):
        # TP：真实是这一类，也预测成这一类
        tp = np.sum((y_test == label) & (y_pred == label))

        # FP：真实不是这一类，却预测成这一类
        fp = np.sum((y_test != label) & (y_pred == label))

        # FN：真实是这一类，却预测成其他类
        fn = np.sum((y_test == label) & (y_pred != label))

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        records.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(np.sum(y_test == label))
        })

    report = pd.DataFrame(records)

    print("\n手写KNN分类报告：")
    print(report.round(4).to_string(index=False))
    print(
        "宏平均F1：",
        round(report["f1"].mean(), 4)
    )

    report.to_csv(
        RESULT_DIR / "classification_report.csv",
        index=False,
        encoding="utf-8"
    )

    # 2. 画测试集散点图，颜色表示真实类别
    fig, ax = plt.subplots()

    for label, name in enumerate(class_names):
        selected = y_test == label

        ax.scatter(
            X_test[selected, 2],
            X_test[selected, 3],
            label=name,
            alpha=0.7
        )

    # 3. 用红色空心圆圈出预测错误的样本
    wrong = y_pred != y_test

    if wrong.any():
        ax.scatter(
            X_test[wrong, 2],
            X_test[wrong, 3],
            s=160,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="Misclassified"
        )

    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_title(
        f"Test predictions: {int(wrong.sum())} errors"
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        RESULT_DIR / "misclassified_samples.png",
        dpi=150
    )
    plt.close(fig)

    print("误分类样本数：", int(wrong.sum()))

def benchmark_distance(A, B):
    # 先检查结果相同，再比较速度
    loop_result = euclidean_distance_loop(A, B)
    vector_result = euclidean_distance(A, B)

    np.testing.assert_allclose(
        loop_result,
        vector_result,
        rtol=1e-10,
        atol=1e-12
    )

    records = []

    methods = [
        ("Python loops", euclidean_distance_loop),
        ("NumPy vectorized", euclidean_distance)
    ]

    for name, function in methods:
        # 每组连续运行20次，共测5组
        times = repeat(
            lambda: function(A, B),
            repeat=5,
            number=20
        )

        # 取5组耗时的中位数，再换算为单次毫秒数
        milliseconds = np.median(times) / 20 * 1000

        records.append({
            "method": name,
            "query_samples": len(A),
            "reference_samples": len(B),
            "features": A.shape[1],
            "milliseconds_per_call": milliseconds
        })

    table = pd.DataFrame(records)

    print("\n欧氏距离性能比较：")
    print("两种实现的计算结果一致。")
    print(table.to_string(index=False))

    loop_ms = table.loc[0, "milliseconds_per_call"]
    vector_ms = table.loc[1, "milliseconds_per_call"]

    print(f"循环版耗时 / 向量化版耗时：{loop_ms / vector_ms:.2f}")

    table.to_csv(
        RESULT_DIR / "distance_benchmark.csv",
        index=False,
        encoding="utf-8"
    )

def run_regression_demo():
    # 生成80组数据：y大致等于2x，但带有随机误差
    rng = np.random.default_rng(SEED)
    X = rng.uniform(0, 10, size=(80, 1))
    y = 2 * X[:, 0] + rng.normal(0, 1, size=80)

    # 回归目标是连续数值，这里不使用stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED
    )

    # 标准化参数仍然只从训练集计算
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    model = KNNRegressor(k=5)
    model.fit((X_train - mean) / std, y_train)
    prediction = model.predict((X_test - mean) / std)

    # 回归不用分类准确率，计算数值误差
    mae = np.mean(np.abs(prediction - y_test))
    mse = np.mean((prediction - y_test) ** 2)

    print("\nKNN回归实验，固定k=5：")
    print("训练样本数：", len(y_train))
    print("测试样本数：", len(y_test))
    print(f"平均绝对误差MAE：{mae:.4f}")
    print(f"均方误差MSE：{mse:.4f}")

    pd.DataFrame({
        "x": X_test[:, 0],
        "actual": y_test,
        "prediction": prediction
    }).to_csv(
        RESULT_DIR / "regression_predictions.csv",
        index=False,
        encoding="utf-8"
    )

    pd.DataFrame([{
        "k": 5,
        "MAE": mae,
        "MSE": mse
    }]).to_csv(
        RESULT_DIR / "regression_metrics.csv",
        index=False,
        encoding="utf-8"
    )

    # 在按顺序排列的x上预测，画出回归曲线
    grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    grid_prediction = model.predict((grid - mean) / std)

    fig, ax = plt.subplots()

    ax.scatter(
        X_train[:, 0], y_train,
        alpha=0.6, label="Training data"
    )
    ax.scatter(
        X_test[:, 0], y_test,
        marker="x", color="red", label="Test data"
    )
    ax.plot(
        grid[:, 0], grid_prediction,
        color="black", label="KNN prediction (k=5)"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("KNN regression demo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "knn_regression.png", dpi=150)
    plt.close(fig)

def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    csv_file = DATA_DIR / "iris.csv"

    # 第一次运行时，生成CSV；以后直接读取
    if not csv_file.exists():
        iris = load_iris(as_frame=True)
        iris.frame.to_csv(
            csv_file, index=False, encoding="utf-8"
        )

    df = pd.read_csv(csv_file)

    # 前4列是特征，target列是类别标签
    feature_columns = df.columns.drop("target")
    X = df[feature_columns].to_numpy()
    y = df["target"].to_numpy()

    print("前5行：")
    print(df.head())

    print("\n样本数：", X.shape[0])
    print("特征数：", X.shape[1])

    print("\n各类别的样本数量：")
    print(df["target"].value_counts().sort_index())

    print("\n各列缺失值数量：")
    print(df.isna().sum())

        # 1. 划分训练集和测试集，同时保留原始行号
    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=0.3,
        random_state=SEED,
        stratify=y
    )

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    print("\n训练集样本数：", len(y_train))
    print("测试集样本数：", len(y_test))

    # 2. 只使用训练集，计算每个特征的均值与标准差
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # 某个特征全部相同时，避免除以0
    std = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    benchmark_distance(X_test_scaled, X_train_scaled)

    # 检查：标准化后的训练集，各列均值应接近0
    np.testing.assert_allclose(
        X_train_scaled.mean(axis=0),
        np.zeros(X_train.shape[1]),
        atol=1e-12
    )

        # 3. 仅用原始训练集进行交叉验证，选择k
    best_k = select_k(X_train, y_train)

    # 选定k后，使用全部训练集作为参考，评估测试集
    model = KNNClassifier(k=best_k, metric="euclidean")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    evaluate_predictions(X_test, y_test, y_pred)

    # 4. 自己计算准确率
    correct = np.sum(y_pred == y_test)
    accuracy = correct / len(y_test)

    print("\n预测正确数量：", correct)
    print(f"测试集准确率：{accuracy:.2%}")

    # 使用相同的k、标准化数据、距离和投票方式
    sklearn_model = KNeighborsClassifier(
        n_neighbors=best_k,
        metric="euclidean",
        weights="uniform",
        algorithm="brute"
    )

    sklearn_model.fit(X_train_scaled, y_train)
    sklearn_pred = sklearn_model.predict(X_test_scaled)

    sklearn_accuracy = np.mean(sklearn_pred == y_test)
    difference_count = np.sum(y_pred != sklearn_pred)

    print("\nsklearn对照实验：")
    print(f"手写KNN准确率：{accuracy:.2%}")
    print(f"sklearn准确率：{sklearn_accuracy:.2%}")
    print("两个模型预测不同的样本数：", difference_count)

    # 保存两种模型的总体结果
    summary = pd.DataFrame({
        "model": ["handwritten", "sklearn"],
        "k": [best_k, best_k],
        "test_accuracy": [accuracy, sklearn_accuracy]
    })

    summary.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8"
    )

    # 逐个样本比较，保留原始数据行号，方便检查
    comparison = pd.DataFrame({
        "original_row": test_indices,
        "actual": y_test,
        "handwritten": y_pred,
        "sklearn": sklearn_pred,
        "same_prediction": y_pred == sklearn_pred
    })

    comparison.to_csv(
        RESULT_DIR / "prediction_comparison.csv",
        index=False,
        encoding="utf-8"
    )

    if difference_count > 0:
        print("\n预测不同的样本：")
        print(comparison.loc[~comparison["same_prediction"]])
    else:
        print("两个模型在本次测试集上的预测完全一致。")

    # 5. 自己统计混淆矩阵
    # Iris的类别固定为0、1、2
    matrix = np.zeros((3, 3), dtype=int)

    for actual, predicted in zip(y_test, y_pred):
        matrix[int(actual), int(predicted)] += 1

    confusion_table = pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1", "actual_2"],
        columns=["predicted_0", "predicted_1", "predicted_2"]
    )

    print("\n混淆矩阵：行是真实类别，列是预测类别")
    print(confusion_table)

    confusion_table.to_csv(
        RESULT_DIR / "confusion_matrix.csv",
        encoding="utf-8"
    )

    # 6. 保存每条测试数据的真实答案与预测答案
    prediction_table = df.iloc[test_indices].copy()
    prediction_table["prediction"] = y_pred
    prediction_table["correct"] = y_pred == y_test

    prediction_table.to_csv(
        RESULT_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8"
    )

    # 7. 显示第一个测试样本的邻居，观察预测依据
    distances, indices = model.kneighbors(
        X_test_scaled[:1]
    )

    print("\n第一个测试样本的预测依据：")
    print("邻居在训练集中的行号：", indices[0])
    print("对应距离：", distances[0])
    print("邻居类别：", y_train[indices[0]])
    print("真实类别：", y_test[0])
    print("预测类别：", y_pred[0])

    # 用花瓣长度和花瓣宽度画散点图
    class_names = ["setosa", "versicolor", "virginica"]

    for label, name in enumerate(class_names):
        selected = y == label

        plt.scatter(
            X[selected, 2],
            X[selected, 3],
            label=name,
            alpha=0.7
        )

    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.title("Iris dataset")
    plt.legend()
    plt.tight_layout()

    image_file = RESULT_DIR / "iris_scatter.png"
    plt.savefig(image_file, dpi=150)
    plt.close()

    print("\n散点图已保存：", image_file)

    run_regression_demo()

if __name__ == "__main__":
    main()