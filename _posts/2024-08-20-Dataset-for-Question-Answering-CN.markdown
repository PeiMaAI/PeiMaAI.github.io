---
title: "Dataset-for-Question-Answering(CN)"
layout: post
date: 2024-08-15 09:02
headerImage: false  
category: blog 
author: Pei Ma
---

**Title:** Dataset for Question Answering (CN) [To be continued...]  
**Date:** 2024-08-14 13:21:39  
**Tags:** Dataset  

# QA数据集概览

问答系统（QA）是自然语言处理领域的核心研究方向之一，旨在从多种信息源中提取精确答案。为满足不同的研究需求和应用场景，各类问答数据集应运而生，并根据数据形式和任务特性进行了系统分类。以下概览介绍了一些重要的问答数据集，涵盖了多种数据类型和任务要求。这些数据集为研究人员提供了丰富的测试和训练资源，以推动在不同信息格式下问答系统的性能提升和技术进步。

### 开放领域问答

- **[HybridQA](https://hybridqa.github.io/)**: 每个问题与一个维基百科表格及多个段落对齐，表格单元格与段落链接。 
- **[OTT-QA](https://ott-qa.github.io/)**: 基于HybridQA的数据集，结合了文本和表格证据，构建了一个开放领域问答基准。 
- **[NQ-Table](https://arxiv.org/pdf/2103.12011)**: 开放领域问答数据集，结合了表格和文本证据。 
- **[TAT-QA](https://nextplusplus.github.io/TAT-QA/)**: 类似于FinQA，但包含了既有算术问题又有范围回答的问题。 
- **[FinQA](https://https://finqasite.github.io/)**: 基于HybridQA的财务报告数据集，仅包含算术回答的问题（忽略范围回答问题）。 
- **[MultiHiertt](https://github.com/psunlpgroup/MultiHiertt)**: 包含层次表格和文本的数值推理问题数据集。 
- **[TriviaQA](https://nlp.cs.washington.edu/triviaqa/)**: 包含多个领域的开放问答数据集，提供了丰富的证据段落用于回答问题。 
- **[Natural Questions](https://ai.google.com/research/NaturalQuestions)**: 提供自然语言问题及其对应的维基百科段落，支持开放领域问答研究。 
- **[HotpotQA](https://hotpotqa.github.io/)**: 提供需要多跳推理的问题和答案，要求在多个文档中找出信息以回答问题。 
- **[FEVER](https://fever.ai/)**: 主要用于事实验证任务，提供了新闻文章和对应的真值判断问题。 

### 表格问答

- **[WikiTableQuestions (WTQ)](https://ppasupat.github.io/WikiTableQuestions/)**: 包含22033个问题-答案对，配有2108个表格；人工标注的问题和答案。
- **[WikiSQL](https://github.com/salesforce/WikiSQL)**: 对Wiki表格进行SQL标注，包含81,000个问题和24,000个表格。
- **[AIT-QA](https://github.com/IBM/AITQA)**: 涉及层次表格的数据集，包含116个关于美国航班的表格和515个问题。
- **[HiTab](https://arxiv.org/pdf/2108.06712v3)**: 针对层次表格的数据集。 
- **[TAPAS](https://github.com/google-research/tapas)**: 针对表格的问答数据集，结合了SQL查询和自然语言问题，覆盖多种表格格式。 
- **[TabFact](https://github.com/wtmo/TabFact)**: 包含表格的真值判断数据集，提供了大量的表格和对应的事实验证问题。 
- **[TableQA](https://github.com/ABSA-HKUST/TableQA)**: 涵盖各种表格结构的问答任务，重点关注从表格中提取信息。 
- **[SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253)**: 专注于结构化数据的问答任务，通过SQL查询从表格中提取信息。 

### 知识图谱问答

- **MetaQA (2018)**: 知识图谱问答数据集。
- **GRAPHQ (2016)**: 知识图谱问答数据集。
- **WebQSP (2016)**: 基于Freebase的问答数据集。
- **CWQ (2018)**: 基于WebQSP的扩展数据集，使用SPARQL查询标注来研究语义解析。
- **LC-QuAD (2017)** 和 **LC-QuAD 2.0 (2019)**: 基于Wikidata和DBpedia的知识图谱数据集，包含30,000个问题，配有SPARQL查询。
- **GrailQA (2021)**: 基于Freebase的知识图谱问答数据集，包含64,000个问题，并进行查询标注。
- **KQA Pro (2022)**: NL问题与SPARQL注释的数据集，知识图谱包括FB15k-237和对齐的Wikidata实体及3,000个相同名称的其他实体。 
- **Freebase QA (2014)**: 使用Freebase知识图谱的早期问答数据集，提供了大量关于知识图谱的问答对。

### 知识图谱+文本问答

- 评估开放领域文本问答数据集：TriviaQA (2017)、WebQuestion (2013)、CWQ (2018)、WebQAP (2016)，使用WordNet、Freebase、ConceptNet等知识图谱。
- **WebQuestion (2013)**: 包含针对Freebase的自然语言问题和对应的答案，广泛用于知识图谱和文本问答的研究。 
- **ComplexWebQuestions (CWQ) (2018)**: 扩展了WebQuestion的数据集，使用复杂的SPARQL查询以测试知识图谱问答的能力。

### 文档问答

- **SQuAD (2018)**: 文档问答数据集，提供了大规模的段落和问题对，用于评估模型的理解能力。 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- **TrivialQA (2017)**: 文档问答数据集，侧重于简单的问题，主要用于基线模型的评估。
- **Natural Questions (NQ) (2019)**: 提供自然语言问题及其对应的维基百科段落，支持开放领域问答研究。 [Natural Questions](https://ai.google.com/research/NaturalQuestions)
- **SearchQA (2017)**: 提供了基于搜索引擎的问答数据集，包含从网络搜索结果中提取的问题和答案。 
- **DocVQA (2020)**: 文档问答数据集，专注于从复杂的文档中提取信息，包括扫描文档和表格。 
- **FEVEROUS (2020)**: 事实验证问答数据集，扩展了FEVER数据集，包含了更多文档和验证任务。
- **HotpotQA (2018)**: 提供需要多跳推理的问题和答案，要求在多个文档中找出信息以回答问题。 [HotpotQA](https://hotpotqa.github.io/)

# 问答数据集概览

## 开放领域问答数据集

### HybridQA

**介绍**：HybridQA由Chen等人在《HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data》中提出。现有的问答数据集大多专注于处理同质信息，基于文本或知识库/表格信息。但由于人类知识分布在异质形式中，仅使用同质信息可能会导致严重的覆盖问题。为填补这一空白，我们提出了HybridQA，一个新的大规模问答数据集，要求在异质信息上进行推理。每个问题与一个维基百科表格及多个与表格中实体相关的自由格式语料库对齐。问题设计为需要同时汇总表格信息和文本信息，即缺少任何一种形式都会使问题无法回答。我们测试了三种不同的模型：1) 仅表格模型；2) 仅文本模型；3) 结合异质信息的混合模型。实验结果表明，两种基线模型的EM分数均低于20%，而混合模型的EM分数超过40%。这一差距表明，在HybridQA中汇总异质信息的必要性。然而，混合模型的分数仍远低于人类表现。因此，HybridQA可以作为一个挑战性的基准，用于研究异质信息的问答任务。

**数据集及代码**：公开可用：[GitHub - HybridQA](https://github.com/wenhuchen/HybridQA)。


### TAT-QA

**介绍**：TAT-QA（Tabular And Textual dataset for Question Answering）由Zhu等人在《TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance》中引入。TAT-QA是一个大规模的问答数据集，旨在推动对更复杂且现实的表格和文本数据进行问答研究，特别是那些需要数值推理的问题。

**特点**：
- 上下文为混合型，包括一个半结构化的表格和至少两个相关段落，这些段落描述、分析或补充表格内容；
- 问题由具备丰富金融知识的专家生成，大多数为实际应用问题；
- 答案形式多样，包括单一跨度、多个跨度和自由格式；
- 回答问题通常需要各种数值推理能力，包括加法（+）、减法（-）、乘法（x）、除法（/）、计数、比较、排序及其组合；
- 除了真实答案外，还提供了对应的推导过程和规模（如有）。

**数据量**：TAT-QA包含16,552个问题，涉及2,757个来自真实金融报告的混合上下文。

### FinQA

**介绍**：FinQA由Chen等人在《FinQA: A Dataset of Numerical Reasoning over Financial Data》中提出。该数据集专注于对金融数据进行深入问答，旨在自动化分析大量金融文档。与现有的通用领域任务相比，金融领域涉及复杂的数值推理和异质表示理解。

**特点**：
- 提供了金融专家编写的问答对，并注释了黄金推理程序，以确保完整的可解释性；
- 引入了基线模型并进行了全面实验，结果显示，流行的大型预训练模型在获取金融知识和复杂的多步骤数值推理上远不及专家；
- 数据集及代码公开可用：[GitHub - FinQA](https://github.com/czyssrs/FinQA)。

### MultiHiertt

**介绍**：MultiHiertt由Zhao等人在《Numerical Reasoning over Multi Hierarchical Tabular and Textual Data》中提出。该数据集针对包含文本和表格内容的混合数据进行数值推理，特别是多层次表格的情况。

**特点**：
- 每个文档包含多个表格和较长的非结构化文本；
- 多数表格为层次结构；
- 问题所需的推理过程比现有基准更加复杂且具有挑战性；
- 提供了详细的推理过程和支持事实的注释。

**模型**：引入了一种新型问答模型MT2Net，该模型首先应用事实检索从表格和文本中提取相关支持事实，然后使用推理模块对检索到的事实进行符号推理。实验结果表明，MultiHiertt对现有基线模型提出了强有力的挑战，其表现远落后于人类专家。

**数据集及代码**：公开可用：[GitHub - MultiHiertt](https://github.com/psunlpgroup/MultiHiertt)。

### HotpotQA

**介绍**：HotpotQA由Yang等人在《HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering》中引入。该数据集基于英语维基百科，包含约113K个众包问题，这些问题需要引入两个维基百科文章的介绍段落来回答。

**特点**：
- 涵盖多种推理策略，包括涉及缺失实体的问题、交集问题（满足属性A和属性B的是什么？）、比较问题等；
- 提供了十个段落，其中包含了黄金段落；在开放领域的全维基百科设置中，模型仅给出问题和整个维基百科；
- 模型评估包括答案准确性（通过精确匹配（EM）和单词F1测量）和可解释性（评估预测的支持事实句子与人工注释的匹配度）。

### [FEVER](https://fever.ai/)

**介绍**：FEVER（Fact Extraction and VERification）由Thorne等人在《FEVER: a large-scale dataset for Fact Extraction and VERification》中提出。该数据集用于验证文本源的事实，包含185,445个通过改变维基百科句子生成的声明，随后被验证。

**特点**：
- 声明被分类为“支持”、“反驳”或“信息不足”；
- 对于前两类，注释人员记录了形成判断所需的句子；
- 发展了一个流水线方法，并与设计良好的预言者进行了比较，表明FEVER是一个具有挑战性的测试平台，有助于推动文本源的声明验证研究。

**数据集及代码**：[FEVER](https://fever.ai/)

## 表格问答数据集

### WikiTableQuestions

**介绍**：WikiTableQuestions由Pasupat等人在《Compositional Semantic Parsing on Semi-Structured Tables》中提出。该数据集基于HTML表格，包括22,033个问题-答案对，问题由Amazon Mechanical Turk工人编写，表格来自维基百科，包含至少8行和5列。

**特点**：
- 问题不是通过预定义模板设计，而是由用户手工编写，展现出高度的语言变异性；
- 相较于之前的知识库数据集，涵盖了近4,000个独特的列标题，涉及的关系比封闭领域数据集更多；
- 问题涵盖广泛领域，需要进行表格查找、聚合、超类（如最大值、最小值）、算术运算、连接和合并等操作。

### AIT-QA

**介绍**：AIT-QA（Airline Industry Table QA）由Katsis等人在《AIT-QA: Question Answering Dataset over Complex Tables in the Airline Industry》中提出。AIT-QA是一个特定于航空行业的表格问答数据集。该数据集包含515个问题，这些问题由人工注释者编写，基于从美国证券交易委员会（SEC）公开的主要航空公司2017-2019财年的报告中提取的116个表格。数据集还包含关于问题性质的注释，标记出那些需要层次化标题、特定领域术语和同义改写形式的问题。

**特点**：
- 表格布局更加复杂，相较于传统的表格问答数据集具有更高的挑战性；
- 包含注释，标记出需要层次化标题、特定领域术语和同义改写形式的问题。

**数据集及代码**：公开可用：[GitHub - AIT-QA](https://github.com/IBM/AITQA).

### TabFact

**介绍**：TabFact由Chen等人在《TabFact: A Large-scale Dataset for Table-based Fact Verification》中提出。TabFact是一个大规模数据集，包含117,854个手工注释的陈述，涉及16,573个维基百科表格。这些陈述的关系被分类为“支持”（ENTAILED）和“反驳”（REFUTED）。TabFact是第一个用于评估结构化数据上语言推理的数据集，涉及符号和语言方面的混合推理技能。

**特点**：
- 提供大规模的表格基础事实验证数据；
- 关系分类为“支持”或“反驳”，挑战语言推理和结构化数据处理能力。

**数据集及代码**：公开可用：[TabFact](https://tabfact.github.io/)

### SQA (SequentialQA)

**介绍**：SQA由Iyyer等人在《Search-based Neural Structured Learning for Sequential Question Answering》中提出。SQA数据集旨在探讨在HTML表格上回答一系列相关问题的任务。该数据集包含6,066个序列，总计17,553个问题。

**特点**：
- 专注于在HTML表格中回答相关问题的序列；
- 提供了丰富的序列问题集，涵盖了多个问题的顺序和关系。

**数据集及代码**：公开可用：[SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253)

# (To be continued...)


## Acknowledgment
I would like to express my sincere gratitude to my advisor, Jiaoyan Chen, for his invaluable guidance throughout this research. His generous sharing of datasets and resources has been instrumental in the development of this study.

## Reference
1. Chen, W., Zha, H., Chen, Z., Xiong, W., Wang, H. and Wang, W., 2020. Hybridqa: A dataset of multi-hop question answering over tabular and textual data. arXiv preprint arXiv:2004.07347.
2. Chen, W., Chang, M.W., Schlinger, E., Wang, W. and Cohen, W.W., 2020. Open question answering over tables and text. arXiv preprint arXiv:2010.10439.
3. Herzig, J., Müller, T., Krichene, S. and Eisenschlos, J.M., 2021. Open domain question answering over tables via dense retrieval. arXiv preprint arXiv:2103.12011.
4. Chen, Z., Chen, W., Smiley, C., Shah, S., Borova, I., Langdon, D., Moussa, R., Beane, M., Huang, T.H., Routledge, B. and Wang, W.Y., 2021. Finqa: A dataset of numerical reasoning over financial data. arXiv preprint arXiv:2109.00122.
5. Zhao, Y., Li, Y., Li, C. and Zhang, R., 2022. MultiHiertt: Numerical reasoning over multi hierarchical tabular and textual data. arXiv preprint arXiv:2206.01347.
6. Qi, P., Lin, X., Mehr, L., Wang, Z. and Manning, C.D., 2019. Answering complex open-domain questions through iterative query generation. arXiv preprint arXiv:1910.07000.
7. Joshi, M., Choi, E., Weld, D.S. and Zettlemoyer, L., 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551.
8. Thorne, J., Vlachos, A., Christodoulopoulos, C. and Mittal, A., 2018. FEVER: a large-scale dataset for fact extraction and VERification. arXiv preprint arXiv:1803.05355.
9. Pasupat, P. and Liang, P., 2015. Compositional semantic parsing on semi-structured tables. arXiv preprint arXiv:1508.00305.
10. Katsis, Y., Chemmengath, S., Kumar, V., Bharadwaj, S., Canim, M., Glass, M., Gliozzo, A., Pan, F., Sen, J., Sankaranarayanan, K. and Chakrabarti, S., 2021. AIT-QA: Question answering dataset over complex tables in the airline industry. arXiv preprint arXiv:2106.12944.
11. Cheng, Z., Dong, H., Wang, Z., Jia, R., Guo, J., Gao, Y., Han, S., Lou, J.G. and Zhang, D., 2021. Hitab: A hierarchical table dataset for question answering and natural language generation. arXiv preprint arXiv:2108.06712.
12. Chen, W., Wang, H., Chen, J., Zhang, Y., Wang, H., Li, S., Zhou, X. and Wang, W.Y., 2019. Tabfact: A large-scale dataset for table-based fact verification. arXiv preprint arXiv:1909.02164.