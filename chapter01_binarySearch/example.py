def binary_search(arr, target):
    # 初始化左指针和右指针
    left = 0
    right = len(arr) - 1

    #当左指针小于或等于右指针时，执行循环
    while left <= right:
        #计算中间索引
        mid = (left + right) // 2

        #如果中间元素等于目标值，则返回中间索引
        if arr[mid] == target:
            return mid
        #如果中间元素小于目标值，调整左指针到mid + 1
        elif arr[mid] < target:
            left = mid + 1
        #如果中间元素大于目标值，调整右指针到mid - 1
        else:
            right = mid - 1

    #如果没有找到目标值，返回-1
    return -1

#测试代码
array = [2, 6, 34, 41, 46, 79]
print(binary_search(array, 6))
print(binary_search(array, 79))
print(binary_search(array, 42))

#////////////////////////////////

# 算法性能分析：
#   时间复杂度:\log_2{n}  对数时间

# 复杂度表示法：
#   O(n)  上界
#   \Omega(n)  下界
#   \theta(n)  确界
#   o(n)  松上界
#   \omega(n)  松下界