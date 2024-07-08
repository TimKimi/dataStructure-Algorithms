#递归思想

#递归调用：自己调用自己的过程叫做递归
#避免无限递归，每次递归问题的规模都要缩小
#规模缩小到合适条件时，终止递归，此条件称为基线条件(base case)

#递归VS循环
# 所有递归都可以转化为循环
# 循环不一定可以转化为递归

#递归(recursion) ：可读性：优 性能：差  会使用更多内存空间
#循环(while loop)：可读性：差 性能：优 (注：也称为迭代)

#总结：优先保证代码可读性，碰到性能瓶颈再去优化

#///////////////////////////////

#斐波那契数示例
def fibonacci(n):
    #基线条件
    if n <= 1:
        return n
    else:
        #递归条件
        return fibonacci(n - 1) + fibonacci(n - 2)

#测试代码
n = 10
print("斐波那契数的第", n, "个数是：", fibonacci(n))

#//////////////////////////////////

#栈(stack)

#类比弹夹
#子弹可以压入与弹出，越先压入的子弹会越晚弹出

#如果一个程序调用一个函数，它会先把这个程序压入栈中，等函数返回结果后才会从栈中弹出    这里使用的栈叫做调用栈

#递归程序执行时，会不断地调用自身，把函数压入栈中，当最后一个函数也就是基线条件出现时，再逐渐清空栈空间

#递归优化：循环  尾递归

#尾递归：解决内存占用问题(递归函数调用时，只是把函数压入栈中，什么都没干，导致栈中函数层层依赖，只有全部函数入栈后才逐个出栈)
# 在每次递归迭代的时候，就进行一次运算，然后把运算结果作为参数带入下一个循环中，用以切断函数间的依赖关系

#///////////////////////////////////

#斐波那契数列的尾递归优化示例  !!!但是对于Python语言并不支持尾递归优化，需要第三方模块；JS语言在Chrome浏览器中V8引擎也不支持
def fibonacci_tail(n, a = 0, b = 1):
    if n == 0:
        return a
    elif n == 1:
        return b
    else:
        return fibonacci_tail(n - 1, b, b + a)

#测试代码
n = 10
print("斐波那契数的第", n, "个数是：", fibonacci(n))