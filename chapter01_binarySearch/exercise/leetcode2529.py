"""
Given an array nums sorted in non-decreasing order, return the maximum between the number of positive integers and the number of negative integers.

In other words, if the number of positive integers in nums is pos and the number of negative integers is neg, then return the maximum of pos and neg.
Note that 0 is neither positive nor negative.
"""

class Solution:
    def maximumCount(self, nums: list[int]) -> int:
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > 0:
                right = mid
            else:
                left = mid + 1
        temp = left
        for i in reversed(nums[:left]):
            if i == 0:
                temp -= 1
            else:
                break
        return max(temp,len(nums) - left)

#测试用例
import unittest
from typing import List

class TestMaximumCount(unittest.TestCase):

    def setUp(self):
        self.solution = Solution()

    def test_normal_case(self):
        nums = [1, 1, 2, 3, 3, 3, 4, 5, 5]
        self.assertEqual(self.solution.maximumCount(nums), 9)

    def test_all_positive(self):
        nums = [1, 2, 3, 4, 5]
        self.assertEqual(self.solution.maximumCount(nums), 5)

    def test_all_negative(self):
        nums = [-3, -2, -1]
        self.assertEqual(self.solution.maximumCount(nums), 3)

    def test_contains_zero(self):
        nums = [0, 0, 1, 1, 2, 3, 4]
        self.assertEqual(self.solution.maximumCount(nums), 5)

    def test_empty_array(self):
        nums = []
        self.assertEqual(self.solution.maximumCount(nums), 0)

if __name__ == "__main__":
    unittest.main()
