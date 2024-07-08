#最大公约数gcb(m, n)
#辗转相除法(更像减损法)

#///////////////////////////

#分治法(divide and conquer)包含于递归(recursion)
#分治法：原问题分解为一个或多个子问题，均递归到基线条件，最后合并子问题的解

#/////////////////////////////

#归并排序(分治法应用案例)
'''合并'''
def merge(left, right):
    """
    合并两个有序数组的函数，输入两个有序数组，返回一个新的有序数组
    """
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
'''分解'''
def merge_sort(arr):
    """
    归并排序函数，输入一个数组，返回排序后的新数组
    """
    #基线条件
    if len(arr) <= 1:
        return arr

    #将数组分为两部分，分别进行递归排序
    mid = len(arr) // 2
    left_arr = merge_sort(arr[:mid])
    right_arr = merge_sort(arr[mid:])

    #合并两个有序子数组
    return merge(left_arr, right_arr)

#算法性能分析：时间复杂度O(n\log_2{n})  空间复杂度O(n)

#///////////////////////////////////

#快速排序(in-place方式：用以节省内存空间)

#分区函数(挖坑法)
def partition(arr, low, high):
    #选择第一个元素作为基准
    pivot = arr[low]
    #挖坑初始位置
    left = low
    right = high

    while left < right:
        #从右边开始找小于基准的元素
        while left < right and arr[right] >= pivot:
            right -= 1
        #将小于基准的元素填入左边的坑
        arr[left] = arr[right]
        #从左边开始找大于基准的元素
        while left < right and arr[left] <= pivot:
            left += 1
        #将大于基准的元素填入右边的坑
        arr[right] = arr[left]
    #基准元素放入最后的坑
    arr[left] = pivot
    #返回基准元素的索引
    return left

def quicksort(arr, low, high):
    if low < high:
        #找到分割点，使用分区函数
        pivot_index = partition(arr, low, high)
        #递归对左侧部分进行快速排序
        quicksort(arr, low, pivot_index - 1)
        #递归对右侧部分进行快速排序
        quicksort(arr, pivot_index + 1, high)

#算法性能分析：时间复杂度O(n\log_2{n})  空间复杂度O(\log_2{n})

#测试代码
array = [2, 6, 79, 34, 68, 46]
n = len(array)
quicksort(array, 0, n - 1)
print("排序后的数组:", array)