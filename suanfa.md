算法刷题

​	对于不认识的内置模块总结：

​				1.float('inf')代表无穷大float('-inf')表示无穷小（固定格式）

​				2.abs() 求取绝对值

​				3.set()经常用于去重，比如里面qweer那么就会返回qwer，还有用于记录多次出现的数字

​				 4.% 2取模可以判断奇偶性，比如4%2 =0 还有5%2=1这两个判断例子

​					5.sign = -1 if x < 0 else 1可以用来存正负号

​					6.`map()` 只能对可迭代对象（如列表、元组、字符串等）但是整数不能迭代，

​					7.bool()用于判断是否true或者false一般都是判断里面是否为空

​					8.min(iterable, key=None)

​							`iterable`：需要查找最小值的可迭代对象（例如列表、元组等）

​					9.**`for i, char in enumerate(min_str)`**enumerate

索引和字符的元组 `(i, char)`

​					10.stack.pop()可以移除栈顶的元素，遵从先进后出

​					11.str字符串遍历直接通过len（）

​					12.str.find(substring, start, end)**`substring`** : 要查找的子字符串例如： text = "hello world"   index = text.find("world") 

​					13.<<**位左移运算符**不会修改原始变量的值，生成新的结果

​						<<=直接改变本来的值

			14.reversed 和 reverse 的区别特性	
			reverse()	reversed()
	是否修改原列表	是（原地修改）	否（返回一个新的迭代器）
	返回值	无（返回 None）	返回一个反转的迭代器使用场景	
	需要直接修改原列表时使用	需要生成一个新的反转列表时使用


[::-1]：适合用于直接获取反转后的序列，尤其是处理小序列时，性能较好

​				15.对于字典的使用：比如frep是一个字典{}，freq = {1: 3, 2: 2, 3: 1}，`freq.itemse()`获取所有的键值对的同时转换元组，freq.key获取键，就会返回[3, 2, 1]，freq.values获取值返回[1,2,3]

​				16.tuple(sorted(s))s是属于字符串，重新排序转换成元组，元组是固定的

			17.方法	行为	适用场景
answer = matrix	共享引用（修改互相影响）	绝对不要用
answer = matrix.copy()	浅拷贝（同 row[:]）	简单二维列表
row[:]	浅拷贝每一行	推荐用于二维列表answer = [row[:] for row in matrix]
copy.deepcopy()	递归深拷贝所有层级	嵌套结构（如三维列表）

​			18.''.join(path)字符串拼接

​			19.zip

			
			names = ["Alice", "Bob", "Charlie"]
			ages = [25, 30, 35]
	zipped = zip(names, ages)
	print(list(zipped))  # 输出: [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
​				用于将多个可迭代对象（如列表、元组等）的元素按位置配对，生成一个迭代器，如果输入的可迭代对象长度不同，`zip()` 会以最短的对象为准，忽略多余的元素。

		20.`Counter` 是一个字典子类，用于统计哈希对象（如字符串、列表中的元素）的出现次数。from collections import Counter

words = ["apple", "banana", "apple", "orange", "banana", "apple"]
word_counts = Counter(words)

print(word_counts)

##### 输出: Counter({'apple': 3, 'banana': 2, 'orange': 1})
