###动态规划核心
# Bellman方程:(这里使用Latex代码表示)
#  V(s) = \max_{a \in A(s)} \left\{ R(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V(s') \right\}
# - \( V(s) \)：在状态 \( s \) 下的价值函数。
# - \( \max_{a \in A(s)} \)：在状态 \( s \) 下，从可行的动作集合 \( A(s) \) 中选择一个最大化未来回报的动作 \( a \)。
# - \( R(s, a) \)：在状态 \( s \) 下采取动作 \( a \) 后的即时回报。
# - \( \gamma \)：折现因子，表示未来回报相对于即时回报的重要性。
# - \( \sum_{s' \in S} \)：对所有可能的后继状态 \( s' \) 的求和。
# - \( P(s'|s, a) \)：从状态 \( s \) 采取动作 \( a \) 后转移到状态 \( s' \) 的概率。
# - \( V(s') \)：在下一个状态 \( s' \) 的价值函数。

#动态规划适用于不同类型的优化问题, 特别是具备重叠子问题和最优子结构特性的问题

##例: 求解斐波那契数

# 递归求解:
def fibonacci(N):
    """
    使用递归的方法求解第N个斐波那契数
    """
    if N <= 0:
        return 0
    elif N == 1:
        return 1
    else:
        return fibonacci(N - 1) + fibonacci(N - 2)
'''
这种方式时间复杂度为O(2^n), 是指数时间的时间复杂度
'''
#采用备忘录法
def fibonacci_memo(n, memo={}):
    """
    备忘录法求第N个斐波那契数(记忆化递归)
    """
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    else:
        result = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
        memo[n] = result # 存储子问题的解
        return result
'''
在递归代码中添加了一个数组来存储子问题的解, 这样我们就可以把时间复杂度从指数时间优化到线性时间O(n)

但是由于还是使用的递归方式, 依然会遇到栈溢出的问题(Fibonacci_memo(1000)), 且由于函数调用开销性能并不理想
'''
#表格法

#定义自下而上(表格方法)的函数
def fibonacci_bottom_up(N):
    # 用0初始化DP数组, 并设置F(0)和F(1)的基本情况
    dp = [0] * (N + 1)
    dp[1] = 1

    # 从F(2)开始, 自下而上构建解决方案
    for i in range(2, N + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    # 返回第N个斐波那契数
    return dp[N]

###使用动态规划解决原问题的步骤
# 拆解原问题
#   1、重叠子问题--->一般会使用一个表格来缓存中间结果(DP表)
#   2、最优子结构--->判断解决这些子问题能不能一步步构建出原问题的解
#   3、状态转移----->从原始状态开始如何一步步递推出最终的状态, 这个递推关系可以用状态转移方程来表示
##不同问题可以用不同状态转移方程表示:
# 斐波那契数dp[i] = dp[i - 1] + dp[i - 2]
# 背包问题dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i])
# 最长公共子序列dp[i][j] = \if A[i] = B[j]: dp[i - 1][j - 1] + 1  \else: max(dp[i - 1][j], dp[i][j - 1])
# Floyd-Warshall算法dp[i][j][k] = min(dp[i][j][k - 1], dp[i][k][k - 1] + dp[k][j][k - 1])

###使用以上步骤解决问题
##一、01背包问题(每个背包只能选一次或者不选)
"""
5kg的背包最多能装多少价值的物品(IPAD PRO 11: 重1kg 价值5k; CANON EOS 90D: 重2kg 价值8k; MICROSOFT XBOX SERIES X: 重3kg 价值7k)
"""
##二维DP表
def knapsack(W, weights, values, n):
    """
    解决0-1背包问题的动态规划函数
    :param W: 背包的总容量
    :param weights: 每个物品的重量列表
    :param values: 每个物品的价值列表
    :param n:物品的数量
    :return: 背包能装的最大价值
    """
    # 构造DP表, 行数为物品数量＋1, 列数为背包容量＋1
    K = [[0 for w in range(W + 1)] for i in range(n + 1)]

    # 进行填充DP表
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0 # 填充基础情况
            elif weights[i - 1] <= w:
                # 当前物品可以装入背包时考虑装入和不装入两种情况
                K[i][w] = max(values[i - 1] + K[i - 1][w - weights[i - 1]], K[i - 1][w])
            else:
                # 当物品不能装入背包时, 只能选择不装入
                K[i][w] = K[i - 1][w]

    # DP表的最后一个元素即为问题的最优解
    return K[n][W]
# 时间复杂度O(nW)空间复杂度O(nW)

##一维DP表
def knapsack_optimized(W, weights, values):
    """
    使用一维DP数组解决0-1背包问题
    :param W: 背包的总容量
    :param weights: 每个物品的重量列表
    :param values: 每个物品的价值列表
    :return: 背包能装的最大价值
    """
    n = len(weights)    # 获取物品的数量
    dp = [0] * (W + 1)  # 初始化一维DP数组, 大小为背包容量+1, 所有位置初始化为0

    for i in range(n):
        # 从背包的容量开始递减, 直到当前物品的重量. 这是为了确保每个物品只被考虑一次
        for w in range(W, weights[i] - 1, -1):

            # 对于每个容量w, 我们尝试放入物品i, 并更新dp[w]的值
            # dp[w]的新值时不放入物品i和放入物品i这两种选择中的最大值
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]])

    return dp[w]
# 时间复杂度O(nW)空间复杂度O(W)

##背包问题是一类组合优化问题, 他有很多变体
# 例如:
#  分割等和子集:
#    给定一个非负整数数组nums, 确定是否存在两个非空子集subset1,subset2, 使得两个自己之和相等
#  硬币找零:
#    给定一组硬币C = {c1,c2,...,cn}和一个总金额M, 找到组成金额M所需的最少硬币数量--->完全背包问题(与0-1背包的区别在于允许每种物品取任意件数)
#  分数背包:(使用贪心策略)
#    给定一组物品I = (w1,v1),(w2,v2),...,(wn,vn)和一个背包容量W, 最大化\sum(xi * vi), 其中\sum(xi * wi) ≤ W 且0 ≤ xi ≤ 1

##二、最长公共子序列
"""
给定两个序列, 找出两个序列的最长公共子序列, 子序列不要求连续, 但元素必须保持相对顺序
"""
def longestCommonSubsequence_with_path(X, Y):
    """
    计算两个序列X和Y的最长公共子序列, 并返回子序列
    :param X:序列X
    :param Y:序列Y
    :return: 最长公共子序列的长度和子序列
    """
    m, n = len(X), len(Y)

    # 初始化dp表的值
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 逐个填充dp表的值
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # 当前字符匹配
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # 当前字符不匹配, 取上方或左方的较大值
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # 回溯, 构造最长公共子序列
    i, j = m, n
    lcs = []
    while i > 0 and j > 0:
        # 当前字符匹配
        if X[i - 1] == Y[j - 1]:
            lcs.append(X[i - 1])
            i -= 1
            j -= 1
        # 否则, 移动到值较大的方向(上方或左方)
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    lcs = lcs[::-1] # 反转, 得到正确的序列

    return dp[m][n], ''.join(lcs)