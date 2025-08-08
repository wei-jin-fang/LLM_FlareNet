import matplotlib.pyplot as plt

# 创建一个图形和坐标轴
fig, ax = plt.subplots()

# 绘制罗马数字 II，调整y坐标来将两个 I 连在一起
ax.text(0.5, 0.5, 'I', fontsize=50, ha='center', va='center', fontname='serif')
ax.text(0.5, 0.6, 'I', fontsize=50, ha='center', va='center', fontname='serif')

# 设置坐标轴为空
ax.set_axis_off()

# 显示图形
plt.show()
