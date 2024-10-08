###有限状态机(正则语言)-->下推自动机(上下文无关语言)-->图灵机(递归可枚举语言)-->不可判定

#如果一个问题可以由多项式时间解决, 那么这个问题就被认为是计算可行的------图灵可计算性

#算法性能分析的步骤

# 1、确定算法性能的数学模型
#  数学模型的确定主要是由经验性和理论性分析法相辅相成, 共同确定的
#  :经验性分析法
#    直接在目标系统上运行算法, 并用时钟时间来测量算法的运行时间; 
#    足够多的测试样本可以可视化的呈现出算法性能随参数或输入规模的变化规律(但这个方法需要测试的样本太多)
#    我们可以按照2的倍率调整输入规模, 算法性能通常会按照某个恒定的幂指数增加; 
#    基本的数学模型有以下4种: T(n) = an + b  线性增加, T(n) = alog(n) + b, T(n) = an^k, T(n) = ae^(kn); 
#    我们可以根据图像走势分辨出是哪个数学模型这个就是倍率实验
#    若数据符合幂指模型, 然后我们就可以通过对数变换将幂指模型线性化, 然后使用线性回归拟合出方程的斜率和截距, 以及从截距中推导出常量α
#  :理论性分析法
#    因为程序的总运行时间主要和两点有关: 执行每条命令的耗时, 以及执行每条语句的频率; 
#    所以我们可以通过分析算法的控制结构, 特别是涉及循环、递归和其他控制结构的分析来预测算法性能;
#    单条语句的耗时乘以执行频率, 再把总的耗时加起来, 这样我们也可以得到算法性能的数学模型
# 2、模型简化
#  我们得到的数学模型里面通常包含多项表达式, 而当数据规模很大时, 模型中的最高项会主导整个函数的行为, 通常会忽略掉此要想得到首项近似公式, 
#  对首项近似公式继续简化, 去掉常数因子就得到了增长数量级, 用于描述当问题数量级趋向无穷大时, 算法性能如何增长
#  特别说明, 对数时间复杂度没有底数, 由于换底公式可在任意底数之间转换, 不同底数的对数只是相差一个常数因子, 在大O表示法中, 常数因子不重要, 所以省略
# 3、算法的成本模型(对比增长数量级, 更细致和精确)
#  成本模型定义了哪些操作或资源应该被计算; 
#   比如在快速排序中, 成本模型分析主要包括 元素的比较次数, 数据的移动次数, 而在通常情况下数据移动是个更耗时的操作, 因此优化快速排序的关键时减少数据移动次数
#  如何选择合适的成本模型:
#   计算密集型任务: 主要考虑cpu周期数, 算数运行次数, 函数调用; 
#   I/O密集型任务: 磁盘访问, 缓存命中率, 网络延迟;