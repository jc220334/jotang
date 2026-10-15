# jotang
## task0中hello_ml.py为字典numpy_matmul.py为矩阵乘法
## task2中catdog_classifier.py为猫狗分类，conv_kernel.py为卷积核练习
## task3中compute_graph.py为计算链 f(x,y)=(xy+x²)² 的手写反向传播（NumPy）与 autograd 对照；nn_backprop.py 为两层网络 X→Linear→ReLU→Linear→Sigmoid→BCE 的梯度手写与 autograd 对拍；note.md 是这两次实践的笔记
## task4中transformer_arch.py为按论文 Figure 1 重画的 Transformer 结构图（补上三种注意力的 Q/K/V 来源、残差走向与可见范围）；attention_scratch.py 为 NumPy 手写单头/多头注意力，并与 PyTorch 版、nn.MultiheadAttention、中心差分梯度三方对拍，含"为什么除以 √d_k""位置编码""causal mask"三组量化实验；vit_patch_demo.py 为 ViT 的 patch 化与 patch 间注意力演示；note.md 是《Attention Is All You Need》的阅读笔记、任务 4~11 的问题与拓展 3/4 的答案
## task5中clip_pairing.py为加载 OpenAI 官方 CLIP（RN50 公开权重，只推理不训练）做图文配对实验：4 张 COCO 图片 × 4 句英文描述，打印图像/文本特征形状与 L2 归一化前后的模长，算 4×4 余弦相似度矩阵并画热力图，含"故意写错关键内容（颜色/数量/动作）""语序颠倒/同义改写/否定句""零样本分类（拿 task2 的猫狗照片对照训练过的 CNN）""prompt 模板""温度系数"五组量化实验；download_assets.py 负责下载官方权重与公开图片；note.md 是 CLIP 论文的阅读笔记、任务 1 的 8 个问题与"趁热打铁"7 问的答案
