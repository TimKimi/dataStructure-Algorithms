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
    return result
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
import random

def partition(arr, low, high):
    # 随机选择基准元素
    pivot_index = random.randint(low, high)
    pivot = arr[pivot_index]
    
    # 将基准元素交换到数组的第一个位置
    arr[low], arr[pivot_index] = arr[pivot_index], arr[low]
    
    # 挖坑初始位置
    left = low
    right = high

    while left < right:
        # 从右边开始找小于基准的元素
        while left < right and arr[right] >= pivot:
            right -= 1
        # 将小于基准的元素填入左边的坑
        arr[left] = arr[right]
        # 从左边开始找大于基准的元素
        while left < right and arr[left] <= pivot:
            left += 1
        # 将大于基准的元素填入右边的坑
        arr[right] = arr[left]
    
    # 基准元素放入最后的坑
    arr[left] = pivot
    # 返回基准元素的索引
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
import unittest

class TestSortingAlgorithms(unittest.TestCase):

    def test_merge(self):
        # 测试合并两个有序数组的函数
        self.assertEqual(merge([1, 3, 5], [2, 4, 6]), [1, 2, 3, 4, 5, 6])
        self.assertEqual(merge([1, 1, 1], [2, 2, 2]), [1, 1, 1, 2, 2, 2])
        self.assertEqual(merge([], []), [])
        self.assertEqual(merge([1], []), [1])
        self.assertEqual(merge([], [1]), [1])
        self.assertEqual(merge([1, 2, 3], [4, 5, 6]), [1, 2, 3, 4, 5, 6])

    def test_merge_sort(self):
        # 测试归并排序函数
        self.assertEqual(merge_sort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]), [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9])
        self.assertEqual(merge_sort([]), [])
        self.assertEqual(merge_sort([1]), [1])
        self.assertEqual(merge_sort([2, 1]), [1, 2])
        self.assertEqual(merge_sort([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])
        self.assertEqual(merge_sort([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])

    def test_partition(self):
    # 测试分区函数
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        pivot_index = partition(arr, 0, len(arr) - 1)
        print(f"Pivot index: {pivot_index}")
        print(f"Array after partition: {arr}")
        assert all(arr[i] <= arr[pivot_index] for i in range(pivot_index))
        assert all(arr[i] >= arr[pivot_index] for i in range(pivot_index + 1, len(arr)))

    def test_quicksort(self):
        # 测试快速排序函数
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        quicksort(arr, 0, len(arr) - 1)
        self.assertEqual(arr, [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9])
        arr = []
        quicksort(arr, 0, len(arr) - 1)
        self.assertEqual(arr, [])
        arr = [1]
        quicksort(arr, 0, len(arr) - 1)
        self.assertEqual(arr, [1])
        arr = [2, 1]
        quicksort(arr, 0, len(arr) - 1)
        self.assertEqual(arr, [1, 2])
        arr = [5, 4, 3, 2, 1]
        quicksort(arr, 0, len(arr) - 1)
        self.assertEqual(arr, [1, 2, 3, 4, 5])
        arr = [1, 2, 3, 4, 5]
        quicksort(arr, 0, len(arr) - 1)
        self.assertEqual(arr, [1, 2, 3, 4, 5])

if __name__ == '__main__':
    unittest.main()