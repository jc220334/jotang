def analyze_scores(scores):
    if not scores:
        raise ValueError("成绩数据不能为空")

    average = sum(scores.values()) / len(scores)
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]

    return average, best_name, best_score
scores = {
    "张三": 88,
    "李四": 92,
    "王五": 79,
    "赵六": 95
}
print("姓名\t成绩")
for name, score in scores.items():
    print(f"{name}\t{score}")
average, best_name, best_score = analyze_scores(scores)
print(f"\n平均成绩: {average:.2f}")
print(f"最高成绩: {best_name} - {best_score}")