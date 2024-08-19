#贪心策略
# 迪杰斯特拉算法、prim编码、哈夫曼编码

###贪心策略
# 1、在什么样的情况下使用贪心策略得到的结果就是全局最优解

# 1、在什么样的情况下使用贪心策略得到的结果不是全局最优解

# 3、如果使用贪心策略得到的不是最优解，那么得到的会是近似解吗

##资源调度问题
# 例如课程表排课，我需要尽可能将更多的课程安排在一间教室里，(课程不能冲突)
#  用贪心策略解决的话, 先将课程按结束时间排序, 再依次选取结束时间最早的(但开始时间晚于上一节课程的结束时间)那一节课

courses = [
    {'name': 'Algorithms',                              'start': 9,     'end': 11},
    {'name': 'Operating Systems',                       'start': 11,    'end': 13},
    {'name': 'Computer Architecture',                   'start': 8,     'end': 10},
    {'name': 'Software Engineering',                    'start': 10,    'end': 12},
    {'name': 'Database Systems',                        'start': 13,    'end': 15},
    {'name': 'Artificial Inteligence',                  'start': 15,    'end': 17},
    {'name': 'Networks',                                'start': 14,    'end': 16},
    {'name': 'Cryptography',                            'start': 16,    'end': 18},
    {'name': 'Machine Learning',                        'start': 17,    'end': 19},
    {'name': 'Data Science',                            'start': 12,    'end': 14},
]

def schedule(courses):
    # 按照结束时间对课程排序
    sorted_courses = sorted(courses, key=lambda x: x['end'])
    # 初始化结果集
    result = [sorted_courses[0]]
    for i in range(1, len(sorted_courses)):
        # 如果课程的开始时间不冲突, 就把他加入结果集
        if sorted_courses[i]['start'] >= result[-1]['end']:
            result.append(sorted_courses[i])
    return result

result = schedule(courses)
for courses in result:
    print(courses['name'], courses['start'], courses['end'])

# 1、贪心策略需要满足两个条件
# 贪心选择性质: 每次选出局部最优解，且选择之后不允许更改
# 最优子结构: 一个问题的最优解包含子问题的最优解

# 一些反例: 最长路径问题(不满足最优子结构), 背包问题(不适合贪心算法), 
# 其他正面例子: 集合覆盖问题(贪心算法求近似解)

###NP(非确定多项式时间)(Nondeterministic  Polynomial time)类问题
# 用于描述一类可以在多项式时间内验证答案正确性的问题，但找到解则可能需要非多项式时间
# 多项式时间P(可以由输入大小的某个固定次数的多项式来界定)
# 非多项式时间NP(类似指数时间复杂度, 随输入大小的增长而爆炸式增长)
## NP与P类问题能否互相转化属于十大千禧年难题之首, 亟待解决