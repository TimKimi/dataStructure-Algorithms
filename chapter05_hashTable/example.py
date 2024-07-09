#哈希表(散列表)
#通过键值(keys)，经过哈希函数(散列函数Hash Function)映射到对应的数组(buckets)索引，实现快速访问数组元素

#哈希函数
# 相同键值映射到相同索引
# 尽量把键值映射到不同位置，越分散越好

#把整数打散的常用方法：除留余数法N % M
# 若把M设置为100，得到的数组索引分布并不均匀，且有不少相同的键值散列到了相同的索引上，这种现象称为“冲突”，由此这个散列函数就不是一个好的设计
# 若把M设置为一个素数，例如97，相对于前一种，分布更加均匀，冲突的情况明显减少，就可以认为这是一个好的散列函数
# 实际上，理想情况下，“冲突”也难以避免

#如何解决冲突
#1.开散列(opening hash)  也称为拉链法(separate chaning)
# 把冲突元素存储到散列表的主表以外
# 在数组索引后面拉出一条链表，当发生冲突时，把元素添加到链表上(可以根据元素插入顺序，也可以根据元素进行排序，也可以根据元素访问频率排序)

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

class ChainingHashTable:
    def __init__(self, size):
        """
        初始化散列表大小和数据
        """
        self.size = size
        self.hash_table = [None] * size
    
    def _hash(self, key):
        """
        散列函数将键转化为散列索引
        """
        return key % self.size

    def insert(self, key, value):
        """
        向散列表中插入键值对
        """
        index = self._hash(key)  #计算散列索引
        if self.hash_table[index] is None:
            #若索引位置为空，则直接插入新节点
            self.hash_table[index] = Node(key, value)
        else:
            #若索引位置不为空，则遍历链表找到合适位置插入新节点
            current = self.hash_table[index]
            while current:
                if current.key == key:
                    current.value = value  #更新现有键的值
                    return
                if current.next is None:
                    break
                current = current.next
            current.next = Node(key, value)
    
    def search(self, key):
        """
        在散列表中查找给定键的值
        """
        index = self._hash(key)  #计算散列索引
        current = self.hash_table[index]
        while current:
            if current.key == key:
                return current.value  #返回键对应的值
            current = current.next
        return None  #键不存在于散列表中

    def delete(self, key):
        """
        从散列表中删除给定键的值
        """
        index = self._hash(key)  #计算散列索引
        current = self.hash_table[index]
        prev = None
        while current:
            if current.key == key:
                if prev:
                    prev.next = current.next
                else:
                    self.hash_table[index] = current.next
                return
            prev = current
            current = current.next


#2.闭散列(closed hash)  也称为开放定址法(open addressing)
# 必须符合内存空间大于键值数量
# 当发生冲突时，重新进行散列或继续向下遍历直到找到空位
# 寻找空位可以线性探测(Linear Probing)也可以随机探测(Random Probing)
# !!!注意，线性探测容易发生元素堆积形成元素集群(长簇 long cluster)，当集群增多时，会严重影响算法性能

#使用线性探测法实现哈希表
class LinearProbeHashTable:
    def __init__(self, size):
        """
        初始化散列表大小和数据
        """
        self.size = size
        self.threshold = int(size * 0.7)  #装填因子阈值
        self.count = 0  #当前散列表中元素个数
        self.hash_table = [None] * size

    def _hash(self, key):
        """
        散列函数将键转化为散列索引
        """
        return key % self.size

    def _next_index(self, index):
        """
        获取下一个索引位置
        """
        return (index + 1) % self.size

    def _resize(self):
        """
        重新开辟地址空间，扩大散列表的大小并重新插入已有的键值对
        """
        self.size *= 2  #扩大散列表的大小为原来的两倍
        self.threshold = int(self.size * 0.5)  #更新装填因子阈值
        old_table = self.hash_table
        self.hash_table = [None] * self.size
        self.count = 0

        for item in old_table:
            if item is not None:
                key, value = item
                self.insert(key, value)  #重新插入已有键值对

    def insert(self, key, value):
        """
        向散列表中插入键值对，当装填因子达到阈值时，重新开辟地址空间
        """
        if self.count >= self.threshold:
            self._resize()  #装填因子超过阈值，重新开辟地址空间

        index = self._hash(key)  #计算初始索引
        while self.hash_table[index] is not None:
            index = self._next_index(index)  #冲突发生获取下一个索引位置
        self.hash_table[index] = (key, value)  #插入键值对
        self.count += 1

    def search(self, key):
        """
        在散列表中查找给定键的值
        """
        index = self._hash(key)  #计算初始索引
        while self.hash_table[index] is not None:
            if self.hash_table[index][0] == key:
                return self.hash_table[index][1]  #返回键对应的值
            index = self._next_index(index)  #冲突发生，获取下一个索引位置
        return None  #键不存在于散列表中


#////////////////////////////////////////////////////////////////////////////////

#性能分析
# 使用哈希表这种数据结构，无论使用哪种方法，其性能表现均与装填因子(Load factor)有关  \alpha = \frac{N}{M}
# N: the number of entries occupied in the hash table(键的数量)
# M: the number of buckets(分配的存储空间)
# 对线性探测法来说，我们认为装载因子低于0.7是一个合理的区间，当装载因子大于0.8时程序需要重复探测的次数急剧增加
# 对拉链法，他的装载因子可以超过1，但是不建议超过1(程序重复探测次数线性递增)

#各个语言已经实现了散列表
#Python：字典dict    java：HashMap、HashTable  等

#散列表应用
# 数据存储和检索(数据库主键索引、哈希分区等)、缓存(缓存数据存储、缓存命中检查、缓存更新和失效等)、字符串匹配和模式匹配(Karp-Rabin算法等)