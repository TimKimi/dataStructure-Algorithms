"""
Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.
"""

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

#测试用例
import unittest

class TestSolution(unittest.TestCase):
    def setUp(self):
        # 在每个测试方法运行前执行，可以初始化需要的资源或对象
        self.solution = Solution()

    def test_search_found(self):
        nums = [1, 3, 5, 7, 9, 11, 13]
        target = 7
        expected_index = 3
        self.assertEqual(self.solution.search(nums, target), expected_index)

    def test_search_not_found(self):
        nums = [1, 3, 5, 7, 9, 11, 13]
        target = 4
        expected_index = -1
        self.assertEqual(self.solution.search(nums, target), expected_index)

    def test_search_empty_list(self):
        nums = []
        target = 5
        expected_index = -1
        self.assertEqual(self.solution.search(nums, target), expected_index)

    def test_search_single_element(self):
        nums = [10]
        target = 10
        expected_index = 0
        self.assertEqual(self.solution.search(nums, target), expected_index)

if __name__ == '__main__':
    unittest.main()
