"""
You are given an array happiness of length n, and a positive integer k.

There are n children standing in a queue, where the ith child has happiness value happiness[i]. You want to select k children from these n children in k turns.

In each turn, when you select a child, the happiness value of all the children that have not been selected till now decreases by 1. Note that the happiness value cannot become negative and gets decremented only if it is positive.

Return the maximum sum of the happiness values of the selected children you can achieve by selecting k children.
"""

class Solution:
    def selection_sort(self, happiness):
        for i in range(len(happiness)):
            max_index = i
            for j in range(i+1, len(happiness)):
                if happiness[j] > happiness[max_index]:
                    max_index = j
            happiness[i], happiness[max_index] = happiness[max_index], happiness[i]  

    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        self.selection_sort(happiness)
        ans = 0
        for i, x in enumerate(happiness[:k]):
            if x <= i:
                break
            ans += x - i
        return ans
#这种算法会超出题解时间限制，但是放在这里是为了练习选择排序，因此列出此项答案，实际上使用python内置的的sort()(这是一种TimSort算法的实现方式)更为高效和稳定，可直接通过题解