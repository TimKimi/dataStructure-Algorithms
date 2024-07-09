# DRAM&数据结构
#   数组VS链表
#     数组：插入数据O(n)  读取数据O(1)
#     链表：插入数据O(1)  读取数据O(n)

#////////////////////////////////////

# 单向链表
class Node:
    def __init__(self, data): 
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)  
        if not self.head:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def prepend(self, data):
        new_node = Node(data)  
        new_node.next = self.head
        self.head = new_node

    def delete(self, key):
        cur = self.head
        if cur and cur.data == key:
            self.head = cur.next
            cur = None
            return
        prev = None
        while cur and cur.data != key:
            prev = cur
            cur = cur.next
        if cur is None:
            return
        prev.next = cur.next
        cur = None

    def print_list(self):
        cur = self.head
        while cur:
            print(cur.data)
            cur = cur.next

# 使用单向链表
sllist = SinglyLinkedList()
sllist.append(1)
sllist.append(2)
sllist.append(3)
sllist.append(4)

print("正向打印链表:")
sllist.print_list()

print("在头部添加元素0:")
sllist.prepend(0)
sllist.print_list()

print("删除元素2:")
sllist.delete(2)
sllist.print_list()


#/////////////////////////////////////

# 双向链表
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node
        new_node.prev = last_node

    def prepend(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        self.head.prev = new_node
        new_node.next = self.head
        self.head = new_node

    def delete(self, key):
        cur = self.head
        while cur:
            if cur.data == key:
                if cur == self.head:
                    if not cur.next:
                        self.head = None
                    else:
                        self.head = cur.next
                        self.head.prev = None
                else:
                    if cur.next:
                        nxt = cur.next
                        prv = cur.prev
                        prv.next = nxt
                        nxt.prev = prv
                    else:
                        prv = cur.prev
                        prv.next = None
                cur = None
                return
            cur = cur.next

    def print_list(self):
        if not self.head:
            print("链表为空")
            return
        cur = self.head
        while cur:
            print(cur.data)
            cur = cur.next

# 使用双向链表
dllist = DoublyLinkedList()
dllist.append(1)
dllist.append(2)
dllist.append(3)
dllist.append(4)

print("正向打印链表:")
dllist.print_list()

print("在头部添加元素0:")
dllist.prepend(0)
dllist.print_list()

print("删除元素2:")
dllist.delete(2)
dllist.print_list()


#////////////////////////////////////////

# 双向链表
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node
        new_node.prev = last_node

    def prepend(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        self.head.prev = new_node
        new_node.next = self.head
        self.head = new_node

    def delete(self, key):
        cur = self.head
        while cur:
            if cur.data == key:
                if cur == self.head:
                    if not cur.next:
                        self.head = None
                    else:
                        self.head = cur.next
                        self.head.prev = None
                else:
                    if cur.next:
                        nxt = cur.next
                        prv = cur.prev
                        prv.next = nxt
                        nxt.prev = prv
                    else:
                        prv = cur.prev
                        prv.next = None
                cur = None
                return
            cur = cur.next

    def print_list(self):
        if not self.head:
            print("链表为空")
            return
        cur = self.head
        while cur:
            print(cur.data)
            cur = cur.next

# 使用双向链表
dllist = DoublyLinkedList()
dllist.append(1)
dllist.append(2)
dllist.append(3)
dllist.append(4)

print("正向打印链表:")
dllist.print_list()

print("在头部添加元素0:")
dllist.prepend(0)
dllist.print_list()

print("删除元素2:")
dllist.delete(2)
dllist.print_list()

#/////////////////////////////

# 块状链表
class BlockNode:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = [None] * capacity
        self.next = None

class BlockLinkedList:
    def __init__(self, block_size):
        self.block_size = block_size
        self.head = None

    def append(self, data):
        if not self.head:
            self.head = BlockNode(self.block_size)
            self.head.data[0] = data
        else:
            cur = self.head
            while cur:
                empty_slot = self.find_empty_slot(cur)
                if empty_slot is not None:
                    cur.data[empty_slot] = data
                    return
                if not cur.next:
                    break
                cur = cur.next
            new_node = BlockNode(self.block_size)
            new_node.data[0] = data
            cur.next = new_node

    def find_empty_slot(self, node):
        for i in range(self.block_size):
            if node.data[i] is None:
                return i
        return None

    def print_list(self):
        cur = self.head
        while cur:
            for data in cur.data:
                if data is not None:
                    print(data, end=' ')
            cur = cur.next
        print()

# 使用块状链表
block_size = 3
block_list = BlockLinkedList(block_size)
block_list.append(1)
block_list.append(2)
block_list.append(3)
block_list.append(4)
block_list.append(5)

print("打印块状链表:")
block_list.print_list()


#/////////////////////////////////////////////////

#selectionSort

#数组结构实现
def selection_sort(arr):
    # 遍历数组的每一个元素
    for i in range(len(arr)):
        # 假设当前元素是最小的
        min_index = i
        # 找到剩余部分中最小的元素
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        # 交换最小元素和当前位置元素
        arr[i], arr[min_index] = arr[min_index], arr[i]

# 测试选择排序
array = [64, 25, 12, 22, 11]
print("未排序数组:", array)
selection_sort(array)
print("排序后数组:", array)
# 输出结果：
# 未排序数组: [64, 25, 12, 22, 11]
# 排序后数组: [11, 12, 22, 25, 64]

#链表结构实现
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            last = self.head
            while last.next:
                last = last.next
            last.next = new_node

    def display(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def selection_sort(self):
        if self.head is None:
            return

        current = self.head
        while current:
            min_node = current
            next_node = current.next
            while next_node:
                if next_node.data < min_node.data:
                    min_node = next_node
                next_node = next_node.next
            
            # Move min_node to current position
            if min_node != current:
                temp_data = current.data
                current.data = min_node.data
                min_node.data = temp_data
            current = current.next

# 创建一个单向链表并添加一些元素
ll = LinkedList()
ll.append(64)
ll.append(25)
ll.append(12)
ll.append(22)
ll.append(11)

# 显示未排序链表
print("未排序链表:")
ll.display()

# 进行选择排序
ll.selection_sort()

# 显示排序后链表
print("排序后链表:")
ll.display()

# 输出结果
# 未排序链表:
# 64 -> 25 -> 12 -> 22 -> 11 -> None
# 排序后链表:
# 11 -> 12 -> 22 -> 25 -> 64 -> None

#数组实现VS链表实现
# 数组实现不稳定，链表实现稳定

#排序算法稳定指相等的元素的相对位置不会发生变化

#算法性能分析：时间复杂度O(n^2) 空间复杂度O(1)