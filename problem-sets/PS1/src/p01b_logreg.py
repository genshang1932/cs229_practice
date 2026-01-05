import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train logistic regression
    model = LogisticRegression()
    theta = model.fit(x_train, y_train) # 训练模型，得到theta参数
    train_acc = np.mean((model.predict(x_train) >= 0.5) == y_train) # 计算训练集准确率
    print("Train accuracy: {:.4f}".format(train_acc))
    print("Learned theta:", theta)
    # Plot data and decision boundary
    util.plot(x_train, y_train, theta, 'pred/p01b_1.png')
    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    eval_acc = np.mean((y_pred >= 0.5) == y_eval) # 计算验证集准确率
    print("Eval accuracy: {:.4f}".format(eval_acc))
    util.plot(x_eval, y_eval, theta, 'pred/p01b_1_eval.png') #由于本课程没有使用os，所以直接写路径
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    print("Successfully saved predictions")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        print("Fitting logistic regression model...")

        self.theta = np.zeros(x.shape[1]) # initialize theta, 从0开始迭代
        m = x.shape[0] # 样本数量
        iteration = 0 # 记录迭代次数
        # 牛顿迭代
        while True:
            iteration += 1
            theta_old = np.copy(self.theta) # 保存旧的theta值

            h_x = 1 / (1 + np.exp(-x.dot(self.theta))) # 计算h(x)=g(X dot theta)

            gradient_J_theta = (1/m) * x.T.dot(h_x - y) # 计算梯度

            H = (1/m) * (x.T * h_x * (1 - h_x)).dot(x)  # 计算Hessian矩阵

            self.theta -= np.linalg.inv(H).dot(gradient_J_theta) # 更新theta值： - H^-1 dot gradient

            if np.linalg.norm(self.theta - theta_old, ord=1) < 1e-5: # 检查收敛条件,根据题目要求是1e-5
                break
                
        print(f"Converged in {iteration} iterations.")
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))  # 预测函数h(x)=g(X dot theta)
        # *** END CODE HERE ***


# Test logistic regression
if __name__ == '__main__':
    main(
        train_path="/Users/bamgen1932/Downloads/cs229-2018-autumn-main/problem-sets/PS1/data/ds1_train.csv",
        eval_path="/Users/bamgen1932/Downloads/cs229-2018-autumn-main/problem-sets/PS1/data/ds1_valid.csv",
        pred_path='pred/p01b.txt'
    )

