## 1.def analyze_scores(scores):
定义函数analyze_scores，引字典scores进入函数
## 2.if not scores:
  raise ValueError("成绩数据不能为空")
判断字典是否为空，若为空则提示成绩数据不能为空
## 3.sum(scores.values()) / len(scores)
sum求字典中成绩数值的总和，len求字典中人数
## 4.key=scores.get
表示按对应的值去比较
