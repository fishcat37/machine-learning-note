# 关系抽取(Relation Extraction)

关系抽取可分为两个部分，NER(实体识别)和RC(实体对关系判别)，这两个都是classification任务，所以能用分类的指标进行评估，如F1，这种任务的分离造就了两种不同的关系抽取训练方式，pipeline和joint

## NER(Bert+CRF)

其本质可看作token classification，对每个token(这里要求字级划分)都进行分类，判断它是否属于实体，属于哪个部分，根据不同的token标注体系，他的实体数不同，比如BIO和BIOES，一个是$2*n+1$一个是$4*n+1$，所以实际上它也是能进行模型融合的，只需要取模型最后的类别概率进行融合就可，但是在crf中，他是需要训练的，如果使用这种方式融合，就只能选择其中一个模型的crf，或者未训练的crf抑或softmax，但这样似乎都不太好

- **B (Begin)**：实体的开头（Beginning of an entity）。
- **I (Inside)**：实体的中间或后续部分（Inside an entity）。
- **O (Outside)**：不属于任何实体（Outside of an entity）。
- **E (End)**：实体的结束（End of an entity）。
- **S (Single)**：单字/单词实体（Single entity）。

## RC(Bert+Classifier)

在RC中会枚举所有可能的实体对，然后判断它们之间的关系，所以label数量应该是场景关系+1，因为存在无关系这一种，同时可根据场景中的关系限定条件排除掉一些无效的关系对，同时在RC中给Bert类模型的往往是需要实体对的，同时为了更好的效果还需要上下文，也就是原句子，所以输入的文本一般是这样的

```
[CLS] 原 句 [SEP] 实 体 1 [SEP] 实 体 2 [SEP]
```

当然还有一种方式是，直接在原句子中将实体标记出来，这样做需要为每种实体设计出一种特殊标记，如下面这种

> '部件单元': ('[unused1]', '[unused2]'),

然后使用 `tokenizer.add_tokens(['[unused1]', '[unused2]'])`将特殊标记加到tokenizer中(因为tokenizer的逻辑是优先匹配更完整更长的，所以不会匹配成他的一部分)，让他能顺利的转化为单一token

## 训练方法

由于AS任务包含两个子任务，所以有两种训练方式，一种是先训练NER，然后训练RC，另一种是联合训练

### Pipeline(流水线)


#### 优点

- **简单清晰**：NER 和 RE 模块各自优化，便于调试。
- 可以直接利用成熟的 NER 模型。

#### 缺点

- **误差传播（error propagation）**：NER 出错会直接影响后续关系分类。
- **实体与关系脱节**：两个模块没有联合优化，关系预测可能无法反过来帮助识别实体。

### Joint(联合)

#### 思路

实体识别和关系抽取  **同时建模** ，在一个模型里完成。

常见的 joint 方法有几类：

1. **基于序列标注**
   * 把关系抽取改写成序列标注问题。
   * 例如  **TPLinker** ：把实体和关系转化为 token span 对的标注问题，用矩阵标记。
2. **基于表格/矩阵**
   * 例如  **CasRel** ：
     * 先预测 head 实体，再预测它对应的 tail 实体和关系。
     * 本质上是把关系抽取看作条件序列标注。
3. **基于指针/生成式方法**
   * 把 RE 当作“文本生成”，直接生成 `(head, relation, tail)` 三元组。
   * 代表方法：CopyRE、Text2Rel、R-BERT+seq2seq 等。

#### 优点

* **避免误差传播** ：模型直接建模实体-关系整体。
* **信息共享** ：关系信息能帮助实体识别，反之亦然。
* 更适合 **重叠关系（overlapping relations）** 场景。

#### 缺点

* **设计更复杂** ，训练难度大。
* 实现上需要精心建模，否则性能未必优于 pipeline。

# gplinker

在gplinker中很重要的是gp(global pointer)，经典gplinker中会有三个gp，其中实体分类一个，实体的gp不区分主体与客体，只区分实体，他会有一个矩阵存储句子中词元的实体分数，其形状为(d,d)，位置(i,j)表示从i到j的词元为实体的logit，所以它需要下掩码，同时对于padding的部分也要进行mask，对于关系分类，我们有两个gp，其中一个用于打分head，另外一个用于打分tail，其输出的形状为(n,d,d)这里的n就是关系种类数，这样才能指明关系实体对的边界，同时他也需要进行和实体gp一样的mask，使用-inf填充mask的位置。对于efficient GPLinker，他改动的点在于关系分类处，传统的gplinker在增加关系数时会导致占用大幅上升，efficient通过先判断一对实体是否有关系再判断他们具体的关系来减少了参数量，在计算MultilabelCategoricalCrossentropy的时候应该使用log-sum-exp trick来保证数值稳定。并且要使用RoPE 旋转位置编码
