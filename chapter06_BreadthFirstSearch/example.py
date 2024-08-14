#戈尼斯堡七桥问题--》欧拉--》图论
# 这里研究的是与图相关的路径问题

#图(Graph)
# 是否存在路径?  是否是最短路径?
# 图由节点和边组成(图可以模拟任何有链接的事物，事物看作点，联系看做边)

#图的代码实现
class Graph:
    def __init__(self):
        """
        初始化图对象, 使用一个空字典。
        字典中的每个键代表图中的一个顶点,
        对应的值是这个顶点的邻居列表.
        """
        self.graph_dict = {}
    
    def add_vertex(self, node):
        """
        如果顶点"node"不在字典中, 向字典添加键"node", 值为空列表.
        否则, 不需要进行任何操作.
        """
        if node not in self.graph_dict:
            self.graph_dict[node] = []

    def add_edge(self, node1, node2):
        """
        要在图中添加一条边,需要在每个顶点的邻居列表中添加另一个顶点.
        """
        self.graph_dict[node1].append(node2)
        self.graph_dict[node2].append(node1)

    def show_edges(self):
        """
        此方法返回一个元组列表, 每个元组代表图中的一条边.
        元组的两个元素是这条边连接的两个顶点
        """
        edges = []
        for node in self.graph_dict:
            for neighbour in self.graph_dict[node]:
                if {neighbour, node} not in edges:
                    edges.append({node, neighbour})
        return edges

#简易示例:

# 初始化图
G = Graph()
# 添加节点
G.add_vertex("猪八戒")
G.add_vertex("唐僧")
G.add_vertex("孙悟空")
G.add_vertex("牛魔王")
G.add_vertex("铁扇公主")
G.add_vertex("红孩儿")
G.add_vertex("小妖怪")
# 添加边
G.add_edge("孙悟空", "猪八戒")
G.add_edge("孙悟空", "唐僧")
G.add_edge("孙悟空", "牛魔王")
G.add_edge("孙悟空", "铁扇公主")
G.add_edge("铁扇公主", "红孩儿")
G.add_edge("牛魔王", "红孩儿")
G.add_edge("牛魔王", "铁扇公主")
G.add_edge("猪八戒", "唐僧")


###一个关于图的算法---用于解决上面的关于图的路径问题

#广度优先搜索(breadthFirstSearch)

'''
此处需要使用一个新的数据结构------队列(Queue)
队列只允许最前面的出队列(dequeue), 从队列的最后入队列(enqueue)
'''
# 队列的实现代码
class Queue:
    def __init__(self):
        """
        初始化一个空队列
        """
        self.items = []

    def isEmpty(self):
        """
        检查队列是否为空
        返回True, 则队列为空
        """
        return self.items == []

    def enqueue(self, item):
        """
        把一个元素添加到队尾
        """
        self.items.append(item)

    def dequeue(self):
        """
        从队首移除一个元素
        返回被移除的元素
        """
        if self.isEmpty():
            raise Exception("队列为空, 不能执行出队操作")
        return self.items.pop(0)

    def size(self):
        """
        返回队列中元素的数量
        """
        return len(self.items)

from collections import deque #这是一个Python自带的双端队列
# 其包含方法：append() appendleft() pop() popleft() extend({7, 8, 9}) extendleft({1, 2, 3}) rotate()这个是向右旋转, 参数为负值则向左旋转

'''
一点疑问解答:
  虽然此处pop(0)和popleft()可实现相同功能, 但pop(0)的时间复杂度为O(n), 而popleft()确实O(1)更为高效
'''

#BFS代码实现如下

class Graph:
    def __init__(self):
        """
        初始化图对象, 使用一个空字典。
        字典中的每个键代表图中的一个顶点,
        对应的值是这个顶点的邻居列表.
        """
        self.graph_dict = {}
    
    def add_vertex(self, node):
        """
        如果顶点"node"不在字典中, 向字典添加键"node", 值为空列表.
        否则, 不需要进行任何操作.
        """
        if node not in self.graph_dict:
            self.graph_dict[node] = []

    def add_edge(self, node1, node2):
        """
        要在图中添加一条边,需要在每个顶点的邻居列表中添加另一个顶点.
        """
        self.graph_dict[node1].append(node2)
        self.graph_dict[node2].append(node1)
    
    def BFS(self, start):
        """
        这个方法实现了 
        它以"start"为起点进行搜索.
        返回一个列表, 表示从"start"开始的广度优先搜索的顶点访问顺序.
        """
        visited = []
        queue = deque([start])

        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.append(vertex)
                neighbours = self.graph_dict[vertex]
                unvisited_neighbours = list(set(neighbours) - set(visited))
                queue.extend(unvisited_neighbours)
            
        return visited

    def bfs(graph, start, end):
        """
        使用广度优先搜索寻找从start到end的最短路径.
        返回一个列表, 表示最短路径中的节点
        """
        queue = deque([[start]])
        #创建一个空集合, 存储已经访问过的节点
        vidited = set()

        while queue:
            path = queue.popleft()
            #获取路径上最后一个节点
            vertex = path[-1]
            #如果该节点是目标节点，返回当前路径(即为最短路径)
            if vertex == end:
                return path
            
            elif vertex not in visited:
                #遍历该节点的所有邻居节点
                for neighbour in graph[vertex]:
                    #对于每个邻居节点，都将其加入到当前路径的尾部，形成一个新的路径
                    new_path = list(path)
                    new_path.append(neighbour)
                    #将新的路径添加到队列的右端
                    queue.append(new_path)
                #将当前节点标记为已访问
                visited.add(vertex)

###算法性能分析
#时间复杂度: O(Vertex + Edge)


###########################
#补充: 
# 上述图均为无向图(Undirected Graph), 关系是双向的. 
# 还有有向图(Directed Graph), 如果关系是双向的则需要两条边. 
# 还有一种图叫加权图(Weighted Graph), 它的每条边是加权的
# *****还有一种特殊的图, 称为树(Tree), 它无向连通且没有环路, 存在根节点*****这属于另一种数据结构了

# 图的存储方式：
# 1. 邻接表(Adjacency List)
#   节点用数组存储, 它们的邻接点放到链表中存储
#   优点: 节省空间(存储空间需求与边数量成正比, 适合表示稀疏图)  
#   缺点: 查找两个顶点是否相邻的速度较慢(使用字典而不使用链表属于一种特殊的邻接表, 字典实现了哈希表, 查找较链表方式更快)
# 2. 邻接矩阵(Adjacency Matrix)
#   是一种二维数组, 行列均为所有节点,如果两个节点间有联系则对应矩阵位置设为1或者为权重值, 否则为0
#   优点: 实现简单, 可快速判断两节点是否相邻
#   缺点: 如果图是稀疏的, 会浪费很多存储空间

#Python中并没有图这种存储方式, 但有第三方库提供, 常用的是NetworkX和python-igraph