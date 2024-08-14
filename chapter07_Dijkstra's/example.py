#应用最广泛的最短路径算法

##小球实验
# 桌子上有几个小球(红黄蓝紫)，每个球代表一座城市，连接小球的城市代表城市之间的路径，绳子的长短表示两个城市之间的距离，如果要找出红色小球到其他三个小球的最短路径,我们只需要提起红色小球就能找出其他小球到红色球的最短路径(两点之间直线段最短)

#迪杰斯特拉算法的具体过程
# 伪代码示例：
# DIJKSTRA(G, w, s)
#  INITIALIZE-SINGLE-SOURCE(G, s) //初始化图，将所有节点的距离设置为无穷大，只有起点的距离为0
#  //创建两个集合, 分别是S集合和Q集合, S集合用于存储已经确定最短路径的节点, Q集合即为未被确定的节点
#  S = {空集}
#  Q = G.V
#  while Q != {空集}
#    // 从Q集合中找出距离起点最短的节点, 加入到S集合中
#    u = EXTRACT-MIN(Q)
#    S = S u {u}
#    //边松弛
#    // 从上一个节点开始, 找出与其相邻的所有邻居节点到起始点的距离, 如果这个距离小于当前存储的距离那就更新它
#    for each vertex v 属于 G.Adj[u]
#      RELAX(u, v, w)

### 具体的算法代码实现

# 引入heapq库, 用于实现优先队列
import heapq

def dijkstra(graph, start, end):
    if start not in graph or end not in graph:
        return float('infinity'), []

    # 用无穷大初始化最短距离，并将前一个节点初始化为 None
    shortest_distances = {node: (float('infinity'), None) for node in graph}
    shortest_distances[start] = (0, None)
    
    # 最小堆优先队列
    heap = [(0, start)]

    while heap:
        # 弹出距离最小的节点
        current_distance, current_node = heapq.heappop(heap)
        
        # 如果当前路径比已知的最短路径长，跳过
        if current_distance > shortest_distances[current_node][0]:
            continue
        
        # 检查相邻节点
        for neighbour, distance in graph[current_node].items():
            new_distance = current_distance + distance
            
            # 如果找到更短的路径
            if new_distance < shortest_distances[neighbour][0]:
                shortest_distances[neighbour] = (new_distance, current_node)
                heapq.heappush(heap, (new_distance, neighbour))
    
    # 重建最短路径
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = shortest_distances[current_node][1]
    path = path[::-1]
    
    # 检查是否有路径，如果没有，返回无穷大和空路径
    if shortest_distances[end][0] == float('infinity'):
        return float('infinity'), []
    
    # 返回最短距离和路径
    return shortest_distances[end][0], path

#测试代码
import unittest

class TestDijkstra(unittest.TestCase):
    def setUp(self):
        # 在测试前初始化一个图
        self.graph = {
            'A': {'B': 1, 'C': 4},
            'B': {'A': 1, 'C': 2, 'D': 5},
            'C': {'A': 4, 'B': 2, 'D': 1},
            'D': {'B': 5, 'C': 1}
        }

    def test_shortest_path_A_to_D(self):
        distance, path = dijkstra(self.graph, 'A', 'D')
        self.assertEqual(distance, 4)
        self.assertEqual(path, ['A', 'B', 'C', 'D'])

    def test_shortest_path_A_to_C(self):
        distance, path = dijkstra(self.graph, 'A', 'C')
        self.assertEqual(distance, 3)
        self.assertEqual(path, ['A', 'B', 'C'])

    def test_shortest_path_B_to_D(self):
        distance, path = dijkstra(self.graph, 'B', 'D')
        self.assertEqual(distance, 3)
        self.assertEqual(path, ['B', 'C', 'D'])

    def test_same_start_end(self):
        distance, path = dijkstra(self.graph, 'A', 'A')
        self.assertEqual(distance, 0)
        self.assertEqual(path, ['A'])

    def test_no_path(self):
        # 图中不存在的节点之间的路径
        distance, path = dijkstra(self.graph, 'A', 'E')
        self.assertEqual(distance, float('infinity'))
        self.assertEqual(path, [])

if __name__ == '__main__':
    unittest.main()

###迪杰斯特拉算法的适用范围
# 有向、非负权重加权图(所有边的权重都是非负的)、单源最短路径(单源:一个点到另一个点\全源:一个点到其他所有点)

# 迪杰斯特拉算法不能解决负权重加权图的原因: 使用贪心策略, 只考虑局部最优解, 不考虑全局
# 要解决负权重加权图, 可以使用动态规划策略: 贝尔曼福特算法(Bellmon-Ford)每一步依赖于前一步的结果, 考虑了全局最优解
