import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
x = np.random.randn(20)
y = np.random.randn(20)

# 绘制散点图
plt.scatter(x, y)
plt.title("Scatter Plot of Random Data")  # 添加标题
plt.xlabel("X-axis")  # 添加x轴标签
plt.ylabel("Y-axis")  # 添加y轴标签
plt.show()  # 显示图形