---
title: "TableRAG(CN)"
layout: post
date: 2024-08-15 13:51
headerImage: false  
category: blog 
author: Pei Ma

---

# 背景
表格作为一种基础且广泛应用的半结构化数据类型，广泛存在于关系数据库、电子表格应用程序和用于数据处理的编程语言中，涵盖了金融分析（Zhang et al., 2020; Li et al., 2022）、风险管理（Babaev et al., 2019）和医疗保健分析等多个领域。在这些应用中，表格问答（TableQA）是对表格数据进行推理的一个关键下游任务（Ye et al., 2023a; Cheng et al., 2023）。

表格问答的目标是使计算机能够理解人类针对表格内容的查询，并以自然语言作出回答。随着近年来大规模语言模型（LLMs）的快速发展，表格问答已成为一个重要的子领域，并取得了显著的进展（Ray, 2023）。目前，大多数利用LLM进行表格问答的研究都是基于单个表格的（Li et al., 2023）。这些方法通常通过将表格预处理后，将问题和表格逐个输入LLM，侧重于让LLM更好地理解表格结构。这类方法在实际应用中主要集中于金融领域，如金融表格问答、金融审计表格处理(Zhu et al., 2021)和金融数值推理等(Chen et al., 2021, Chen et al., 2020)。然而，在现实场景中，往往面临的是一组表格（a set of tables）而非单个表格，用户可能会提出涉及多个表格的任意相关问题。在这种情况下，LLM不仅需要逐个输入回答，更重要的是能够从大量表格中召回相关表格并给出答案。然而，目前在这方面的研究还相对欠缺，我们的研究旨在弥补这一差距。

微调大规模语言模型是解决表格问答挑战的常见方法，但这种方法需要大量的领域特定的标注数据和巨大的计算资源。此外，大多数模型在处理领域特定和复杂的表格数据时，往往过度依赖预训练知识，从而导致幻觉和错误信息（Ray, 2023; Gao et al., 2023）。

为了解决这些挑战，检索增强生成（RAG）方法将检索机制与生成模型相结合，引用外部知识库，以减少模型幻觉并提高领域特定问答的准确性，同时降低资源消耗（Gao et al., 2023）。然而，尽管RAG在处理非结构化文本数据方面表现出色，但在应用于半结构化表格数据时仍存在若干挑战。具体而言，我们识别了以下三个局限性：

1. 为回答问题所需的表格可能非常庞大，包含大量与查询无关的噪声（Lu et al., 2024）。这不仅增加了不必要的计算，还会影响检索器检索时召回的准确性以及生成器响应的准确性。为了解决这个问题，我们可以采用表格采样（Sui et al., 2024）或表格过滤的方法，检索相关的行和列，从而生成最相关的子表（Jiang et al., 2023）。
2. 表格的原始内容可能包含需要进一步澄清的信息，如领域特定术语或缩写（Sui et al., 2024）。这些领域特定的细节可能导致生成器的误解或偏见。为了解决这个问题，我们可以利用外部知识库为表格提供额外的上下文信息（Bian et al., 2023），或通过LLM生成术语解释，这一过程我们称之为table clarifier。
3. 表格通常在不同列中包含多种类型的信息，而传统的检索方法如BM25（Robertson et al., 2009）或Dense Passage Retriever（DPR）（Karpukhin, et al., ）可能会忽略表格细节，影响生成结果。我们可以通过采用ColBERT模型作为检索器来解决这一问题，该模型在标记级别对文本进行编码，使得检索更加细粒度（Li et al., 2023）。

通过结合这些改进，我们的研究旨在为处理多个表格的大规模表格问答任务提供一个更有效的解决方案，以应对更复杂的现实场景。
# Overview
在处理复杂表格问答任务时，我们设计了一个结合最新大规模语言模型（LLM）与检索增强生成（RAG）技术的系统，以应对实际应用中的多表格问题。以下是项目核心思想的图示与介绍。

### 基于RAG的多表格问答系统架构

![The overall structure](/insert_images/The_overall_structure.png)

在这个系统架构中，我们的目标是从多个表格中检索相关信息，并生成准确的自然语言答案。流程可以分为以下几个关键步骤：

1. **表格处理与文本切分**：首先，原始表格数据经过预处理和文本切分，将表格内容转换为多个文本片段。这样做的目的是使得数据更易于处理，并能够针对查询进行高效的检索。
2. **向量数据库的构建**：切分后的文本和表格片段经过嵌入处理并存储在向量数据库中。向量数据库通过高效的向量化检索技术，可以迅速找到与查询相关的内容片段。
3. **查询与检索**：当用户提出问题时，检索器会从向量数据库中查找与问题相关的表格片段。在这个过程中，我们引入了ColBERT模型来增强检索器的精度。ColBERT通过在标记级别编码文本，能够实现更细粒度的检索，从而提高检索结果的相关性。
4. **生成答案**：检索到的相关文本片段与用户的提问一起输入到大规模语言模型（LLM）中，由LLM生成最终的自然语言答案。

### 多表格问答的增强机制

![Enhancement](/insert_images/Enhancement.png)

在处理来自多张表格的数据时，我们的系统引入了多种增强机制，以提高问答任务的精确性和有效性。

1. **基于语义的表格过滤器**：当面对大量表格时，系统首先通过语义分析对表格进行过滤，选择最相关的表格。在此过程中，我们采用了以下两种不同的模型进行文本嵌入，并进行了对比：
    
    ![Overview of table filter](/insert_images/filter_overview.png)

    
    - **利用OpenAI的Embedding模型**：我们使用OpenAI的Embedding模型对表格内容进行嵌入处理，然后利用FAISS向量数据库对嵌入后的数据进行存储和检索，从中返回与查询最相关的表格行和列。
    - **利用ColBERT模型**：我们也使用ColBERT模型对表格内容进行嵌入，并在检索过程中使用ColBERT进行更细粒度的检索。通过与OpenAI Embedding模型的结果进行对比，我们能够选择更适合特定任务的语义过滤方法。
2. **基于LLM的过滤器**：除了语义过滤器，我们还使用大规模语言模型（LLM）对表格进行智能过滤。通过分析表格内容与查询之间的深层语义关联，LLM能够更精准地选择出最相关的表格片段，进一步提高检索的准确性。
3. **表格澄清器**：在过滤后的表格基础上，我们引入了两个澄清模块：
    
    ![image.png](/insert_images/clarifier_overview.png)
    
    - **术语澄清**：对于表格中的领域特定术语或缩写，我们调用LLM进行解释，帮助LLM更好地理解问题和表格内容。
    - **基于Wiki的摘要生成**：首先，我们通过表格标题、表头或上下文信息，搜索维基百科并返回相关的元数据。接着，将这些维基数据与表格的原始上下文信息打包处理，生成与需要判断的问题或澄清的陈述相关的摘要。这种方式不仅提高了信息的准确性，还为复杂表格的理解提供了更全面的背景支持。

上述架构与增强机制有效地应对了当前表格问答任务中存在的挑战，特别是在多表格环境下的实际应用。通过结合先进的检索技术、语义与LLM过滤，以及大规模语言模型，我们的系统能够从大量表格中迅速找到相关信息并生成精确的答案，为各类复杂数据分析任务提供了有力的支持。

# 数据集的选择

## [Tablefact](https://tabfact.github.io/)

在现有的表格问答数据集中，我们已经进行了广泛的尝试和研究。关于详细的数据集整理，请参阅我的另一篇博客：[Dataset for Question Answering](https://yuhangwuai.github.io/2024/08/14/Dataset-for-Question-Answering/)：。通过这些经验，我们在使用数据集进行表格问答的检索增强生成时，发现主要面临以下几个问题：

1. **问题简短导致召回效果不佳**：
    - 许多问答数据集中的问题通常非常简短，仅由几个单词组成。这种简短的提问在相似度检索或其他密集型检索过程中，往往导致相关表格的召回效果不佳。
2. **问题形式单一**：
    - 问题通常以相似的疑问词和连词开头。例如，在SQA数据集中，"What are the schools?" 和 "What are the countries?" 这个问题尽管涉及完全不同的内容，但它们的开头 "What are the" 两是相同的。如果数据集中有近500个以 "What are the" 开头的问题，这种形式上的重复会使得相关表格的准确召回变得非常困难。
3. **缺乏表标题**：
    - 大量问答数据集不包含表标题，通常一个表格仅对应一个问题，完全不涉及检索阶段。在这种情况下，每次输入时将表格和问题直接一起输入模型。然而，当缺乏表标题时，从大量表格中精准返回相关表格的难度大大增加。

基于这些挑战，在我们最初的实验中，TableFact数据集是我们首选的基础数据集。TableFact的数据集专注于表格事实验证这一任务，能够有效地评估模型在推理和判断方面的能力。

TableFact是一个大规模的数据集，包含117,854条手动标注的声明，涉及16,573个维基百科表格。这些表格和声明之间的关系被分类为“ENTAILMENT”（蕴含）和“REFUTATION”（反驳）。该数据集首次提出在结构化数据上评估语言推理能力，涉及符号推理和语义推理的混合推理技能。这种复杂性使得TableFact成为评估深度学习模型在同时处理语义和符号推理任务时的能力的理想数据集。

| Channel | Sentence | Table |
| --- | --- | --- |
| Simple (r1) | 50,244 | 9,189 |
| Complex (r2) | 68,031 | 7,392 |
| Total (r1 + r2) | 118,275 | 16,573 |
| Split | Sentence | Table |
| Train | 92,283 | 13,182 |
| Val | 12,792 | 1,696 |
| Test | 12,779 | 1,695 |

该数据集的示例如下：

![Tablefact sample instances（Chen et al., 2019）](/insert_images/tablefact.png)

Tablefact sample instances（Chen et al., 2019）

TableFact数据集的主要优势在于其专注于表格事实验证这一任务，能够有效地评估模型在推理和判断方面的能力。具体任务是：给定一个表格和一个声明，要求模型判断该声明是否与表格中的信息一致。模型需要对表格内容进行深入推理，并对声明标记“True”（真实）或“False”（虚假）。

TableFact数据集不仅包含大量复杂的表格和声明对，覆盖多种领域和主题，能够很好地模拟现实中可能遇到的多表格问答场景。这为我们提供了一个具有挑战性的测试平台，可以帮助我们更全面地评估和优化我们的多表格问答系统。使用这个数据集的另一个重要原因是，它能够更好地控制LLM的输出，使我们能够精确评估模型的表现。

*我们选择使用[TableFact数据集](https://tabfact.github.io/)的原因如下：*

1. **纯表格数据集**：TableFact的数据主要以表格形式呈现，声明内容的相似性较低，使得在检索和召回过程中难度相对较小，有助于模型准确定位相关信息。
2. **明确的分类任务**：TableFact的数据集任务明确，即判断声明的真假。这种任务设置使得在生成答案时更容易控制大模型的输出，从而更准确地评估模型的推理能力。

## [Feverous](https://fever.ai/dataset/feverous.html)

在使用TableFact之后，我们选择了FEVEROUS（Fact Extraction and VERification Over Unstructured and Structured information）数据集。FEVEROUS是一个专为事实验证任务设计的大规模数据集，与TableFact不同，它不仅包含结构化表格数据，还包含非结构化文本数据。这使得FEVEROUS在检索和推理过程中更加复杂和具有挑战性。

![Feverous sample instances(Aly et al., 2021)](/insert_images/feverous.png)
Feverous sample instances(Aly et al., 2021)

[Feverous](https://fever.ai/dataset/feverous.html)的数据集包含超过80,000个表格和文本段落对，以及与之关联的超过120,000个事实验证问题。模型在处理FEVEROUS数据集时，除了判断声明的真假之外，还需在三个选项之间做出选择：**Supported**（支持）、**Refuted**（反驳）、或**Not Enough Information**（信息不足）。这种三选一的任务设置进一步增加了模型的推理复杂度，与TableFact的二元分类任务相比，FEVEROUS能够更全面地评估模型的推理能力，尤其是在多源信息整合和判断中的表现。

*选择[Feverous](https://fever.ai/dataset/feverous.html)的原因*：

- 结合结构化和非结构化数据，增加了模型的推理难度。
- 三选一任务设置，能够更好地评估模型在复杂推理任务中的表现。

## [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253)

在进一步扩展实验时，我们引入了[SQA（Sequential Question Answering）](https://www.microsoft.com/en-us/download/details.aspx?id=54253)数据集。SQA数据集的设计旨在评估模型在复杂、多步骤问答场景中的表现。这一数据集包含超过6,000个对话式问答对，每个对话涉及多个相关联的问题，这些问题通常与先前的问答上下文相关联。与TableFact和FEVEROUS不同，SQA要求模型在一个连续的问答过程中保持上下文的理解和一致性。

SQA中的问题不仅需要回答当前的问题，还需要基于之前的问答进行推理。更重要的是，SQA要求模型给出的答案是自由的，涵盖文本、数字等多种形式。这种开放式的问答增加了模型推理的复杂性，也考验了模型在处理自由回答时的生成能力。

![SQA sample instances (Lyyer et al., 2017)](/insert_images/sqa.png)

SQA sample instances (Lyyer et al., 2017)

*选择[SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253)的原因*：

- 纯结构化数据，暂时不涉及两个类型数据的整合
- 专注于多步骤问答，增加了模型在处理对话上下文和连续推理时的挑战。
- 自由回答形式的引入，考验了模型在开放式问答任务中的表现。

## [HybridQA](https://hybridqa.github.io/)

最后，我们选择了HybridQA数据集，以进一步提升对模型多模态信息处理能力的评估。[HybridQA](https://hybridqa.github.io/)是一个融合了表格和文本信息的数据集，旨在测试模型在多模态信息上的综合问答能力。该数据集包含6,241个问答对，每个问题涉及多个不同信息源的内容，包括表格和关联的非结构化文本信息。

HybridQA的独特之处在于，模型不仅需要从多个信息源中提取和整合相关信息，还需要在回答过程中涉及数值推理的步骤。这种多模态、多步骤的问答形式要求模型在复杂任务中表现出色，尤其是在跨模态信息整合和数值推理方面。

![HybridQA sample instances (Chen et al., 2020)](/insert_images/hybridqa.png)
HybridQA sample instances (Chen et al., 2020)

*选择[HybridQA](https://hybridqa.github.io/)的原因*：

- 涉及表格和文本的两个类型信息，进一步测试模型的跨模态整合能力。
- 复杂的问答形式和数值推理步骤，提供了更高的挑战性，用以评估模型在处理多源信息时的综合表现。
- 自由回答形式的引入，考验了模型在开放式问答任务中的表现。

# 实施方案

## 第一部分：表格过滤器

![Overview of table filter](/insert_images/filter_overview.png)

1. **基于语义的过滤**
    - **生成嵌入向量**：为表格中的每一行和列生成语义嵌入向量，并为用户的查询生成相应的嵌入向量。我们采用两种方法来实现这一过程：
        1. **向量数据库匹配**：使用OpenAI或其他嵌入模型生成嵌入向量，然后通过FAISS等向量数据库计算相似度，快速返回与查询相关的行列。
        2. **细粒度匹配**：使用ColBERT预训练模型对表格数据和查询进行嵌入和匹配，以实现更高的细粒度匹配，从而选择最相关的行列。
    - **选择相关行列**：根据相似度得分，选取与查询最相关的前k行和前k列，构建新的子表格。
2. **基于大型语言模型（LLM）的过滤**
    - **转换为字符串**：将查询和表格内容转化为字符串并拼接，形成上下文。
    - **调用GPT过滤**：使用GPT模型过滤并提取与查询相关的行列，同时生成相应的Python代码以实现筛选。为了提高代码生成的准确性和一致性，采用了自一致性策略：
        1. **自一致性策略**：让GPT生成5次代码，选择出现频率最高的代码作为最终筛选代码。如果生成的代码版本各不相同，则选择第一次生成的结果。
        2. **执行和错误处理**：执行最终选择的代码段，更新表格。如果代码执行过程中出现错误，则捕获错误信息并返回原始表格，以确保流程的鲁棒性。

### LLM过滤过程中的难点及解决方案

在表格过滤过程中，尤其是基于LLM的表格过滤器中，存在以下几个主要难点：

1. **列名的一致性问题**：GPT在生成筛选代码时，有时会误识别列名，导致生成的代码与原始表格中的列名不一致，从而引发错误。例如，把scheduled', 'capacity (mw)理解为scheduled capacity (mw)是一个列名，LLM将多个列名合并为一个，或者将单个列名错误分拆。
    
    **解决方案**：为了解决这一问题，可以在Prompt中明确提供整理后的列名作为参数传递给GPT，以确保生成的代码使用的列名与原始表格完全一致。这种方式能够从根本上减少列名识别错误的发生。
    
2. **信息丢失问题**：在LLM过滤表格过程中，筛选后的表格可能会因为过度过滤而丢失回答问题所需的关键信息。这种情况会导致在后续生成回答时，由于缺乏必要的证据，生成的答案不准确甚至错误。
    
    **解决方案**：为了解决这一问题，可以采用“保守筛选”策略，即让LLM仅过滤掉自己非常确定与陈述无关的内容。如果LLM在判断某些内容是否与陈述相关时存在不确定性，应倾向于保留这些内容。这种策略能够最大程度地保留潜在的关键证据，确保生成的回答能够基于完整的信息进行推理，从而提高答案的准确性和可信度。
    
3. **数据类型不匹配导致的筛选问题**：在处理表格数据时，尤其是在筛选数值类型的数据时，可能会因为数据类型不一致而导致筛选结果为空或不准确。
    
    **解决方案**：即使是在处理数值数据时，也建议通过字符串匹配的方式进行筛选。这种做法可以避免由于数据类型不匹配引起的筛选错误，从而提高筛选的准确性和可靠性。
    
4. **Prompt设计的有效性**：为了让GPT能够准确理解任务并生成正确的筛选代码，Prompt的设计至关重要。一个不明确的Prompt可能导致GPT生成不符合预期的代码。
    
    **解决方案**：在设计Prompt时，应确保其清晰、具体，并包含足够的上下文信息，以便GPT能够准确理解任务要求。同时，可以通过反复测试和调整Prompt，找到最适合的表达方式，提高代码生成的准确性。
    
5. **代码生成的一致性问题**：GPT在生成代码时可能会产生多个不同版本的代码，导致结果不一致。
    
    **解决方案**：通过自一致性策略，生成多个版本的代码并选择出现频率最高的版本，确保结果的一致性和可靠性。如果所有生成的代码都不一致，则使用第一次生成的代码并进行错误捕获处理，以确保流程的稳定性。
    

最后我们使用的详细的设置如下：

```python

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def call_llm_code_generation(self, context: str) -> str:
        """Synthesize code snippet from the table context."""
        prompt = f"""
        Example: Synthesize code snippet from the table context to select the proper rows and columns for verifying a statement / answering query.
        The generated code must use the exact column names provided, including spaces, capitalization, and punctuation.
        The generated code should treat all data as strings, even if they look like numbers.
        Only filter out rows and columns that are definitely not needed to verify the statement / answering query.

        User 1:
        I need an expert to help me verify the statement by filtering the table to make it smaller. Statement: The scheduled date for the farm with 17 turbines be 2012.
        Columns: ['wind farm', 'scheduled', 'capacity (mw)', 'turbines', 'type', 'location']
        df = pd.DataFrame({{
            'wind farm': ['codling', 'carrowleagh', 'dublin array', 'glenmore', 'glenough', 'gortahile', 'grouse lodge', 'moneypoint', 'mount callan', 'oriel', 'skerd rocks', 'shragh', 'garracummer', 'knockacummer', 'monaincha', 'gibbet hill', 'glenough extension'],
            'scheduled': ['unknown', '2012', '2015', '2009 summer', '2010 winter', '2010 autumn', '2011 summer', 'unknown', 'unknown', '2013', 'unknown', 'planning submitted oct 2011', '2012', '2013', '2013', '2013', '2013'],
            'capacity (mw)': [1100, 36.8, 364, 30, 32.5, 20, 20, 22.5, 90, 330, 100, 135, 42.5, 87.5, 36, 15, 2.5],
            'turbines': [220, 16, 145, 10, 13, 8, 8, 9, 30, 55, 20, 45, 17, 35, 15, 6, 1],
            'type': ['unknown', 'enercon e - 70 2.3', 'unknown', 'vestas v90', 'nordex n80 / n90', 'nordex n90', 'nordex n90', 'unknown', '3 mw', 'unknown', '5 mw', 'enercon e82 3.0 mw', 'nordex n90 2.5 mw', 'nordex n90 2.5 mw', 'nordex n117 2.4 mw', 'nordex n90 2.5 mw', 'nordex n90 2.5 mw'],
            'location': ['county wicklow', 'county cork', 'county dublin', 'county clare', 'county tipperary', 'county laois', 'county tipperary', 'county clare', 'county clare', 'county louth', 'county galway', 'county clare', 'county tipperary', 'county cork', 'county tipperary', 'county wexford', 'county tipperary']
        }})
        User 2:
        To verify the statement 'The scheduled date for the farm with 17 turbines be 2012', we need to filter the rows and columns to focus on relevant information. 
        Since we are interested in the 'wind farm', 'scheduled', and 'turbines' columns, the most impactful change will be to filter the rows and columns as follows:
        filtered_table = df[['wind farm', 'scheduled', 'turbines']].query("turbines == '17'")

        User 1:
        I need an expert to help me verify the statement by filtering the table to make it smaller. Statement: All 12 club play a total of 22 game for the wru division one east.
        Columns: ['club', 'played', 'drawn', 'lost', 'points for', 'points against', 'tries for', 'tries against', 'try bonus', 'losing bonus', 'points']
        df = pd.DataFrame({{
            'club': ['pontypool rfc', 'caerphilly rfc', 'blackwood rfc', 'bargoed rfc', 'uwic rfc', 'llanharan rfc', 'newbridge rfc', 'rumney rfc', 'newport saracens rfc', 'beddau rfc', 'fleur de lys rfc', 'llantrisant rfc'],
            'played': [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
            'drawn': [2, 2, 2, 0, 2, 1, 2, 2, 0, 0, 1, 0],
            'lost': [2, 4, 6, 8, 7, 12, 11, 12, 14, 15, 16, 18],
            'points for': [648, 482, 512, 538, 554, 436, 355, 435, 344, 310, 300, 402],
            'points against': [274, 316, 378, 449, 408, 442, 400, 446, 499, 483, 617, 592],
            'tries for': [81, 56, 60, 72, 71, 44, 36, 56, 45, 32, 34, 55],
            'tries against': [32, 37, 42, 52, 50, 51, 47, 52, 64, 61, 77, 77],
            'try bonus': [12, 7, 8, 10, 6, 1, 2, 5, 2, 2, 2, 4],
            'losing bonus': [1, 3, 3, 4, 2, 7, 3, 3, 3, 4, 4, 6],
            'points': [89, 78, 71, 70, 64, 46, 45, 44, 37, 34, 28, 26]
        }})
        User 2:
        To verify the statement 'All 12 club play a total of 22 game for the wru division one east', we need to filter the rows and columns to focus on relevant information. 
        Since we are interested in the 'club' and 'played' columns, the most impactful change will be to filter the rows and columns as follows:
        filtered_table = df[['club', 'played']].query("played == '22'")

        User 1:
        I need an expert to help me verify the statement by filtering the table to make it smaller. Statement: Touchdown Atlantic, in the category of sporting, be established in 2010.
        Columns: ['event name', 'established', 'category', 'sub category', 'main venue']
        df = pd.DataFrame({{
            'event name': ['dieppe kite international', 'the frye festival', 'hubcap comedy festival', 'touchdown atlantic', 'atlantic nationals automotive extravaganza', 'world wine & food expo', 'shediac lobster festival', 'mosaïq multicultural festival'],
            'established': [2001, 2000, 2000, 2010, 2000, 1990, 1950, 2004],
            'category': ['sporting', 'arts', 'arts', 'sporting', 'transportation', 'arts', 'arts', 'festival'],
            'sub category': ['kite flying', 'literary', 'comedy', 'football', 'automotive', 'food & drink', 'food & drink', 'multicultural'],
            'main venue': ['dover park', 'university of moncton', 'various', 'moncton stadium', 'moncton coliseum', 'moncton coliseum', 'shediac festival grounds', 'moncton city hall plaza']
        }})
        User 2:
        To verify the statement 'Touchdown Atlantic, in the category of sporting, be established in 2010', we need to filter the rows and columns to focus on relevant information. 
        Since we are interested in the 'event name' and 'established' columns, the most impactful change will be to filter the rows and columns as follows:
        filtered_table = df[['event name', 'established']].query("`event name` == 'touchdown atlantic' and established == '2010'")

        Now, generate a code snippet from the table context to select the proper rows and columns to verify the given statement / answering query.
        Use the existing column names from the provided DataFrame.
        The column names in the generated code must match the provided column names exactly, including spaces, capitalization, and punctuation.
        Only filter out rows and columns that are definitely not needed to verify the statement.
        Only return the code. 
        {context}
        \n\n:
        """

        if self.USE_SELF_CONSISTENCY:
            generated_codes = [self.generate_text(prompt) for _ in range(5)]
            print("Generated codes:", generated_codes)
            
            # Find the most common code
            code_counter = Counter(generated_codes)
            most_common_code, count = code_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_code
            else:
                return generated_codes[0]
        else:
            return self.generate_text(prompt)
```

# 第二部分：表格澄清器

在处理复杂的表格数据时，提供澄清信息有助于增强对表格内容的理解。然而，选择合适的澄清方法至关重要。在最初的设计中，我们尝试使用Google API来检索术语解释，并通过Wikipedia的文档来增强表格内容。具体来说，在最初的设计中，我们采用了以下流程来对表格进行澄清处理。

## 早期方法的全流程

### **术语澄清**

- 首先，使用大型语言模型（LLM）对表格中的内容进行分析，筛选出需要进一步解释的术语。
- 对筛选出的术语，利用Google API进行搜索，以获取相关解释。
- 然后，将检索到的解释附加到表格中，作为术语澄清信息。这个过程可以借助Langchain中的`GoogleSearchAPIWrapper()`来实现。

### **Wiki文档澄清**

- 根据表格的标题、上下文或表头信息，构建Wikipedia查询。例如，如果表格的表头为“Company Name”、“Revenue”、“Number of Employees”等，可以构建类似“company revenue employees market capitalization”的查询。
- 使用Langchain中的`WikipediaRetriever.get_relevant_documents()`进行检索，获取相关的Wikipedia文档。
- 从检索到的文档中提取元数据，如标题、摘要和链接，将其与表格内容结合，作为进一步的澄清数据。

我们使用了下面的Prompt：

```python
You are an expert in data analysis and natural language processing. Your task is to help identify terms in a table that may need further explanation for better understanding. The table contains various fields, some of which might include technical jargon, abbreviations, or specialized terms that are not commonly understood by a general audience.

Here is the table:
[Insert table here]

Please follow these steps:
1. Analyze the content of each cell in the table.
2. Identify any terms that are technical, specialized, or abbreviations that may need further explanation.
3. Generate a list of these terms along with the corresponding cell reference (row and column).

Consider the following when identifying terms:
- Technical terms related to specific industries (e.g., finance, healthcare, technology).
- Abbreviations that are not universally known.
- Jargon that may be specific to a particular field or context.

Output the terms that need explanation in the following format:
- Term: [Term]
  Cell Reference: [Row, Column]

Example output:
- Term: Revenue (million dollars)
  Cell Reference: [2, 2]
- Term: Market Cap (billion dollars)
  Cell Reference: [2, 4]

Be thorough and ensure that all potentially confusing or specialized terms are included in the list.

```

然后我们将其传递给Lanchain，利用GoogleSearchAPIWrapper()实现检索，并将结果加入作为澄清信息。

对于Wikipedia的方法，我们具体实现如下：

例如，下列表格：

| Company Name | Revenue (Million USD) | Number of Employees | Market Cap (Billion USD) |
| --- | --- | --- | --- |
| Company A | 1000 | 5000 | 50 |
| Company B | 2000 | 10000 | 100 |
| Company C | 1500 | 7500 | 75 |

利用表头信息构建查询：

```
"company revenue employees market capitalization"
```

查询到的信息如下：

```python
{
    "title": "List of largest technology companies by revenue",
    "summary": "This is a list of the largest technology companies in the world by revenue.",
    "url": "<https://en.wikipedia.org/wiki/List_of_largest_technology_companies_by_revenue>"
}
```

将上述文档原数据内容与表格结合，作为澄清数据。

> Sui, Y., Zou, J., Zhou, M., He, X., Du, L., Han, S. and Zhang, D., 2023.
 Tap4llm: Table provider on sampling, augmenting, and packing 
semi-structured data for large language model reasoning. *arXiv preprint arXiv:2312.09039*.
> 

---

这个方法理论上可以帮助我们获取丰富的信息资源，但在实践中却暴露出了一些不可忽视的问题。

首先，**Google API的结果准确性问题**。尽管通过Google API检索术语解释在处理某些专有术语时可能较为有效，因为这些术语通常具有唯一的定义。但当面对缩写或多义词时，问题就变得复杂了。例如，“ABC”这一缩写可能对应多个不同的概念，如“美国广播公司”（American Broadcasting Company）或“活动为基础的成本核算”（Activity-Based Costing），甚至还有其他可能的解释。在这种情况下，从Google检索到的术语解释可能会存在不一致性，不仅无法达到预期的增强效果，反而可能导致信息混淆，使结果变得更加复杂和不可靠。

其次，**检索内容的冗长性问题**。Google检索到的查询内容和Wikipedia返回的文档可能过于冗长，包含大量与表格内容相关但与实际查询需求无关的信息。这些冗长的文档在进一步处理时，可能对数据管道（pipeline）的检索效果产生负面影响。目前的研究主要侧重于将每条查询分别传入LLM或预训练模型中进行处理，而我们当前的任务有所不同，这种方法可能会导致效果不佳。如果文档过长且包含过多无关信息，可能会降低模型的准确性和效率，从而影响最终的结果质量。

# 表格澄清策略的改进与完善

## 术语澄清模块的精准优化

基于上述原因，在对大量文献的阅读和深思熟虑之后，我们对表格澄清信息提出了以下两个关键要求：

1. **澄清信息必须提升对表格的理解能力**
    
    澄清信息的首要目标是帮助模型更好地理解表格内容。信息的添加应当是精准且有助于模型在处理表格时，能够更准确地把握其结构和含义，从而提高整体的理解水平。
    
2. **澄清信息必须提高对表格的召回能力**
    
    其次，澄清信息应当有助于提高模型对表格相关内容的召回能力。这意味着在面对查询或分析任务时，模型能够更有效地提取和利用表格中的关键信息。
    

在提出这些要求的同时，我们实际上也明确了两个必须避免的情况：

1. **澄清后的信息有误，影响了LLM对表格的理解能力**
    
    如果澄清信息存在错误，可能会导致模型对表格的误解，从而降低其对表格内容的正确解析。这不仅违背了澄清信息的初衷，还可能使模型的输出结果产生偏差。
    
2. **澄清信息过长，过多冗余，影响模型对相关表格的召回能力**
    
    过长或冗余的信息可能会增加模型处理时的负担，干扰其对核心内容的关注，从而削弱模型在召回相关表格信息时的效率和准确性。
    

## Table澄清器的改进

基于前述对表格增强信息的要求和潜在问题的分析，我们提出了进一步的改进方案，以优化表格增强的方法。这些改进旨在确保增强信息既能提升模型的理解能力，又能提高相关信息的召回效率，从而避免常见的误解和冗余问题。

### **术语澄清模块的改进**

针对术语澄清模块，我们决定直接利用LLM从表格中提取术语并进行解释，而不再依赖GoogleSearchAPIWrap进行外部检索。尽管这一方法无法获得网络上更为广泛的综合信息，但LLM已经能够理解大部分术语和缩写，并且能够结合具体情境提供解释。这样做不仅提高了对表格的理解能力，还有效避免了可能由于外部检索带来的误导信息和冗余信息的问题，确保增强信息的精准和简洁。

### **Wiki参考模块的改进**

### **1. 表格用途的澄清**

我们引入了一个新的澄清信息，即简要的说明表格的用途，是用来回答什么问题的。这种通过明确表格目的生成的方式，可以在使用ColBERT进行信息检索时，显著提高召回率。

通过这种方式，我们实现了增强信息对表格召回能力的提升，确保模型在面对特定查询时能更准确地提取相关数据。具体使用prompt和用例如下：

```python
    def generate_terms_explanation(self, table: dict, statement: str, caption: str) -> str:
        prompt = f"""
        Example: You will be given a table, a statement, and the table's caption. Your task is to identify difficult to understand column names, terms, or abbreviations in the table and provide simple explanations for each. Only explain terms related to the statement.

        User 1:
        I need an expert to help me explain the terms in this table. Here is the statement: The scheduled date for the farm with 17 turbines be 2012.
        Here is the table caption: Wind Farm Details in Ireland
        Here is the table:
        {{
            "wind farm": ["codling", "carrowleagh", "dublin array", "glenmore", "glenough", "gortahile", "grouse lodge", "moneypoint", "mount callan", "oriel", "skerd rocks", "shragh", "garracummer", "knockacummer", "monaincha", "gibbet hill", "glenough extension"],
            "scheduled": ["unknown", "2012", "2015", "2009 summer", "2010 winter", "2010 autumn", "2011 summer", "unknown", "unknown", "2013", "unknown", "planning submitted oct 2011", "2012", "2013", "2013", "2013", "2013"],
            "capacity (mw)": [1100, 36.8, 364, 30, 32.5, 20, 20, 22.5, 90, 330, 100, 135, 42.5, 87.5, 36, 15, 2.5],
            "turbines": [220, 16, 145, 10, 13, 8, 8, 9, 30, 55, 20, 45, 17, 35, 15, 6, 1],
            "type": ["unknown", "enercon e - 70 2.3", "unknown", "vestas v90", "nordex n80 / n90", "nordex n90", "nordex n90", "unknown", "3 mw", "unknown", "5 mw", "enercon e82 3.0 mw", "nordex n90 2.5 mw", "nordex n90 2.5 mw", "nordex n117 2.4 mw", "nordex n90 2.5 mw", "nordex n90 2.5 mw"],
            "location": ["county wicklow", "county cork", "county dublin", "county clare", "county tipperary", "county laois", "county tipperary", "county clare", "county clare", "county louth", "county galway", "county clare", "county tipperary", "county cork", "county tipperary", "county wexford", "county tipperary"]
        }}

        User 2:
        Explanations:
        "scheduled": "The planned date for the wind farm to be operational.",
        "turbines": "The number of wind turbines in the wind farm."

        User 1:
        I need an expert to help me explain the terms in this table. Here is the statement: All 12 clubs play a total of 22 games for the WRU Division One East.
        Here is the table caption: WRU Division One East Standings
        Here is the table:
        {{
            "club": ["pontypool rfc", "caerphilly rfc", "blackwood rfc", "bargoed rfc", "uwic rfc", "llanharan rfc", "newbridge rfc", "rumney rfc", "newport saracens rfc", "beddau rfc", "fleur de lys rfc", "llantrisant rfc"],
            "played": [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
            "drawn": [2, 2, 2, 0, 2, 1, 2, 2, 0, 0, 1, 0],
            "lost": [2, 4, 6, 8, 7, 12, 11, 12, 14, 15, 16, 18],
            "points for": [648, 482, 512, 538, 554, 436, 355, 435, 344, 310, 300, 402],
            "points against": [274, 316, 378, 449, 408, 442, 400, 446, 499, 483, 617, 592],
            "tries for": [81, 56, 60, 72, 71, 44, 36, 56, 45, 32, 34, 55],
            "tries against": [32, 37, 42, 52, 50, 51, 47, 52, 64, 61, 77, 77],
            "try bonus": [12, 7, 8, 10, 6, 1, 2, 5, 2, 2, 2, 4],
            "losing bonus": [1, 3, 3, 4, 2, 7, 3, 3, 3, 4, 4, 6],
            "points": [89, 78, 71, 70, 64, 46, 45, 44, 37, 34, 28, 26]
        }}

        User 2:
        Explanations:
        "played": "The number of games played by the club.",
        "points for": "The total points scored by the club.",
        "points against": "The total points scored against the club."

        Now, explain the terms in the following table.

        Table caption:
        {caption}

        Statement:
        {statement}

        Table:
        {json.dumps(table, indent=2)}

        Please return the result in the following format:
        {{
            "explanations": {{
                "term1": "explanation1",
                "term2": "explanation2",
                ...
            }}
        }}
        """
        generated_text = self.generate_text(prompt)
        return generated_text  
```

### **2. WikiPedia外部信息增强的优化**

![image.png](/insert_images/clarifier_overview.png)

1. **初步检索**：
    - **基于表格标题进行WikiPedia检索**：首先使用表格标题作为关键词进行WikiPedia的检索，获取相关的增强信息。
    - **备用检索**：如果标题检索失败，则使用表头信息进行检索，以提供与表格内容相关的增强信息。
2. **信息打包：**
    - 将Wikipedia中的数据提取元数据，但是我们不直接将这些信息加入澄清内容中，以避免冗余。
    - 我们把Wikipedia的元数据，query、table（包括筛选后的表格或原始表格）以及caption，还有context(如果有context的话)一起打包，发送给LLM进行处理，让LLM根据多方面的信息生成一个表格摘要。

注意事项：

- **避免直接揭示问题答案**：在生成summary时，要注意引导类摘要的撰写，避免直接透露问题的答案或提供直接的解答。总结的目的是帮助LLM更好地理解和引导他们进行进一步探索，而不是直接提供解决方案，并且直接揭示答案的话，可能这个答案也有误导性。
- **聚焦相关内容**：确保LLM生成的摘要仅包括与查询内容相关的信息，避免冗余或不必要的细节。这样可以保持摘要的简洁和聚焦。

具体来说我们的详细实现如下：

```python
    def get_docs_references(self, parsed_example: dict) -> dict:
        print("Starting get_docs_references method")

        retriever = WikipediaRetriever(lang="en", load_max_docs=2)
        
        try:
            # Use caption for document retrieval if available
            if parsed_example["table"].get("caption"):
                print("Using caption for document retrieval:", parsed_example["table"]["caption"])
                docs = retriever.get_relevant_documents(parsed_example["table"]["caption"])
            # If caption is also not available, use table headers
            else:
                print("No caption found, using header instead:", parsed_example['table']['header'])
                docs = retriever.get_relevant_documents(" ".join(parsed_example["table"]["header"]))
            
            
            # Extract relevant metadata from the retrieved documents
            metadata_list = []
            for doc in docs:
                metadata = {
                    'title': doc.metadata.get('title', 'N/A'),
                    'summary': doc.metadata.get('summary', 'N/A'),
                    'source': doc.metadata.get('source', 'N/A')
                }
                metadata_list.append(metadata)

            # Print the metadata for debugging
            print("Retrieved metadata: ", metadata_list)

            # Extract table, statement, and caption from parsed_example
            table = {
                "header": parsed_example.get("table", {}).get("header", []),
                "rows": parsed_example.get("table", {}).get("rows", [])
            }
            statement = parsed_example.get("query", "")
            caption = parsed_example.get("table", {}).get("caption", "")

            # Call the method to generate table summary using metadata
            print("Calling generate_table_summary with metadata")
            generated_summary = self.call_llm.generate_table_summary(metadata_list, table, statement, caption)
            print("Generated summary:", generated_summary)
            
            # Return the generated summary in a dictionary under the 'table_summary' key
            return {"table_summary": generated_summary}
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while retrieving documents: {e}")
            return {"table_summary": "Document retrieval failed"}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"table_summary": "An unexpected error occurred"}
```

使用的具体prompt以及用例内容如下：

```python
    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_table_summary(self, metadata_list: list, context: list, table: dict, query: str, caption: str) -> str:
        """
        Generate a summary for a table that directly addresses a given query, using metadata and context.

        :param metadata_list: List of metadata from related Wikipedia documents.
        :param context: Additional context about the table.
        :param table: Dictionary representing the table's data.
        :param query: The query or statement to be addressed by the summary.
        :param caption: Caption of the table for context.
        :return: JSON string containing the generated summary.
        """
        prompt = f"""
        Example: You will be given a table, a query, the table's caption, metadata from related Wikipedia documents, and the context of the table. 
        Your task is to generate a concise summary for the table that directly addresses the query, using the Wikipedia metadata and the context to enhance understanding. 
        Ensure the summary begins by rephrasing or summarizing the query in a way that naturally introduces the purpose of the table. 
        Do not directly reveal the answer, but guide the reader to make an informed decision based on the provided information.

        Now, generate a summary for the given table, addressing the query and using the Wikipedia metadata and the context provided for enhanced understanding. 
        Ensure the summary starts by rephrasing or summarizing the query to introduce the table's purpose and includes only content related to the query. 
        Please avoid directly revealing the answer.

        Query:
        {query}

        Table caption:
        {caption}

        Table:
        {json.dumps(table, indent=2)}

        Wikipedia metadata:
        {json.dumps(metadata_list, indent=2)}

        Context:
        {json.dumps(context, indent=2)}

        Please return the result in the following format:
        {{
            "summary": "The summary that rephrases the query, includes context from the caption, and incorporates relevant Wikipedia information."
        }}

        """

        generated_text = self.generate_text(prompt)
        return generated_text

```

我们对表格增强方法进行了深入的思考和优化，通过上述方法，我们基本可以确保在处理复杂数据时，模型能够更加准确地理解和召回表格中的关键信息。通过改进术语澄清模块和Wiki参考模块，我们成功避免了外部信息可能带来的误导和冗余问题，提升了模型在不同场景下的整体性能。这些改进不仅为增强信息的质量提供了保障，也为模型在实际应用中的可靠性和效率奠定了坚实基础。

# 第三部分：检索过程增强

在检索过程中，传统的方法如BM25、DPR（Dense Passage Retrieval）、或者直接利用向量数据库进行检索，通常被广泛应用。BM25通过统计关键词在文档中的出现频率进行检索，是一种经典且高效的文本检索方法。而DPR采用双塔模型，利用深度学习技术，将查询和文档嵌入到高维向量空间中，通过向量的近似相似度进行匹配。这两种方法在简单查询场景中表现较好，但在处理复杂、多样化的查询时，可能存在精度和效率的局限性。向量数据库检索则依赖于高效的向量相似性搜索库，如Faiss，来实现快速的相似度计算，适合大规模数据的检索需求。

然而，这些方法在面对复杂的查询或表格类数据时，检索精度都不够。因此，我们在TableRAG系统中最终选择使用ColBERT进行增强。这一选择不仅基于ColBERT独特的创新点和优点，还因为其在实际应用中展现出的高效性和准确性。目前，ColBERT的实现可以通过[RAGatouille](https://github.com/bclavie/RAGatouille)轻松集成到RAG管道中，而Llamaindex也提供了对该仓库的集成，这使得其应用变得更加便捷。

## ColBERT 的创新与优点

### **创新点**

1. **延迟交互框架**：ColBERT通过将查询和文档的编码过程分离，并在编码后再进行相似度计算，减少了在线查询时的计算量。这使得系统能够预先计算文档的表示，大大提高了计算效率。
2. **最大相似度操作（MaxSim）**：ColBERT采用最大相似度操作来评估查询和文档之间的相关性，每个查询嵌入与文档嵌入之间的最大余弦相似度或L2距离相加，简单高效。
3. **BERT编码器共享**：通过共享BERT编码器，并在输入前分别加上特殊标记（[Q]和[D]），ColBERT在节省计算资源的同时，保留了上下文理解能力。
4. **文档的分段和过滤**：过滤掉无关信息，如标点符号，减少计算和存储负担。
5. **基于向量相似性的检索**：利用向量相似性搜索库（如faiss），ColBERT能够高效地从大型文档集合中进行端到端检索。

### **优点**

1. **计算效率高**：预计算文档表示和延迟交互机制使ColBERT在查询处理时的计算量大幅降低，速度提高了两个数量级。
2. **空间利用率高**：通过归一化和降维处理，ColBERT有效地减少了存储空间需求，提升了实际应用的可行性。
3. **强大的扩展性**：ColBERT的架构设计允许其处理大规模文档集合而不牺牲精度，尤其是在向量相似性搜索中的高效剪枝操作中表现突出。
4. **端到端检索能力**：ColBERT能够直接从大型文档集合中检索，提高了系统的召回率和精度。

### ColBERTv2 的改进

在ColBERTv2中，这些优势得到了进一步增强。特别是引入的**残差压缩机制**和**降噪监督**，显著降低了存储需求并提高了训练效果。此外，ColBERTv2通过优化索引和检索过程，实现了更高效的候选生成和段落排序，进一步提升了检索性能。

### 检索过程中的实际应用

在我们的TableRAG系统中，ColBERT不仅用于重新排序预检索的文档集，还通过其端到端的检索能力直接提升了系统的召回率和精度。为进一步优化检索结果的质量，我们还引入了rerank机制，对初步检索到的文档集进行重新排序。这一机制帮助我们在获得初步结果后，进一步细化和提升结果的相关性和准确性。

具体来说，当我们使用ColBERT进行查询时，系统首先对表格中的所有文档进行预处理和编码，生成高效的向量表示。在查询过程中，ColBERT利用这些预先生成的文档向量，通过最大相似度操作快速找到最相关的文档。接下来，rerank机制对这些初步结果进行精细化排序，确保最终呈现给用户的文档是最符合查询意图的。

我们对这一组合策略进行了测试，结果显示，使用ColBERT结合rerank机制不仅大幅度提高了检索的准确性，还进一步优化了查询的响应时间。通过这种多层次的检索与排序方法，我们能够确保检索结果的高精度，同时避免了传统方法中高计算成本和长响应时间的问题。

最终，通过集成ColBERT和rerank机制到我们的TableRAG系统中，我们实现了检索过程中增强信息的有效利用。这一增强策略不仅提升了系统的计算效率和存储利用率，还通过其创新的检索和排序机制，在不牺牲精度的情况下，大幅度提高了检索速度和结果的相关性。这样，我们的系统在处理复杂表格查询时，能够快速且准确地返回最相关的信息，从而显著提升了用户体验和系统的整体性能。

# 第四部分：传入格式增强

## 传入给LLM的表格格式优化

在进行表格增强和检索的过程中，传入给大型语言模型（LLM）的表格格式对最终的处理效果有着至关重要的影响。已有研究探讨了不同的表格转换方法，并比较了它们对LLM问答系统性能的影响。这些方法包括Markdown格式、模板序列化、传统预训练语言模型（TPLM）方法以及使用大型语言模型（LLM）直接生成文本。研究表明，在不同的范式下，表格转换方法的表现各不相同。

在 **Exploring the Impact of Table-to-Text Methods on Augmenting LLM-based Question Answering with Domain Hybrid Data** 一文中，作者比较了不同表格转换方法在混合数据集上的表现，特别是它们在LLM问答系统中的效果：

- **Markdown格式**：使用Markdown格式表示表格内容。
- **模板序列化**：利用预定义模板将表格转换为文本。
- **传统预训练语言模型（TPLM）方法**：使用像T5和BART这样的模型进行表格到文本任务的微调。
- **大型语言模型（LLM）方法**：如使用ChatGPT等模型进行一次性文本生成。

研究结论显示：

- 在数据特征学习与迁移（DSFT）范式中，使用语言模型（TPLM和LLM）进行表格到文本转换的方法表现最佳。
- 在检索增强生成（RAG）范式中，Markdown格式展现了意想不到的效率，但LLM方法依然表现出色。

> [Exploring the Impact of Table-to-Text Methods on Augmenting LLM-based Question Answering with Domain Hybrid Data](https://arxiv.org/abs/2402.12869)
> 

## 传入格式优化

基于上述研究，我们在实验中选择了两种表格格式将其传入LLM，以进一步优化系统的性能：

1. **HTML格式**：HTML格式提供了清晰的结构化表示，使得模型能够准确理解表格的层次和内容关系。这种格式适合在需要保留复杂表格结构时使用，特别是在多维表格或嵌套表格的场景中，HTML格式能有效传达表格的语义信息。
2. **Markdown格式**：Markdown格式因其简洁性和人类可读性，在各种文本表示任务中广泛使用。研究表明，在RAG范式中，Markdown格式不仅能有效表示表格内容，还能提高模型的处理效率。因此，我们在实验中采用Markdown格式来评估其在实际应用中的表现。

通过采用这两种格式，我们希望能够最大限度地发挥LLM在表格处理任务中的潜力。HTML格式的结构化优势和Markdown格式的简洁高效性为我们提供了不同场景下的灵活选择，确保表格内容能够被LLM准确理解和高效处理，从而进一步提高表格问答系统的整体性能。

这种格式优化策略的实施，不仅基于现有研究的理论支持，还在我们的实验中得到了实际验证，为后续的系统开发提供了坚实的基础。我们将继续探索其他可能的格式，以进一步优化表格传入LLM的方式，确保系统在各种复杂场景下都能保持卓越的表现。

# 评估实验

## 1. 对照实验

对照实验的目的是评估在基础模型上逐步添加各个模块后的性能变化。具体设计如下：

- **Baseline**（基线模型）：不包含任何额外模块的原始模型，用作参考标准。
- **Filter**（过滤器）：在基线模型上逐步添加不同的过滤模块。
    - **Semantics-based**：这里进一步分为两个小部分：
        - **Colbert**：加入 Colbert 语义相似度比较模块。
        - **OpenAI Embedding Model**：加入 OpenAI Embedding Model 进行语义相似度比较的模块。
    - **LLM-based**：加入基于大型语言模型的过滤器。
- **Clarifier**（澄清器）：在基线模型上逐步添加不同的澄清策略。
    - **Term Exp.**：加入术语扩展模块。
    - **Table Summary**：加入表格摘要模块。
    - **Exp. & Summary**（术语扩展与表格摘要组合）：同时加入术语扩展与表格摘要模块。
- **Formatter**（格式化器）：在基线模型上逐步添加不同的格式化方式。
    - **String**：使用字符串格式化。
    - **Markdown**：使用 Markdown 格式化。
    - **Html**：使用 Html 格式化。
- **Retriever**（检索器）：在基线模型上测试不同的检索策略，特别是对于 Colbert 模型，还评估了是否使用 rerank 机制对结果进行重新排序的影响。
    - **BM25**：使用 BM25 进行检索。
    - **DPR**：使用 DPR 进行检索。
    - **Colbert**：使用 Colbert 进行检索，同时评估是否使用 rerank 机制对检索结果进行重新排序。
- **Consist.**（一致性）：在基线模型上测试加入一致性模块后的性能。

## 2. 消融实验

- **Filter**（过滤器）：探讨不同过滤器对模型性能的影响。
    - **Semantics-based**（语义基础过滤器）：这里进一步分为两个小部分，分别移除使用 Colbert 和 OpenAI Embedding Model 进行语义相似度比较的模块。
    - **LLM-based**（基于大型语言模型的过滤器）：移除基于LLM的过滤模块。
- **Clarifier**（澄清器）：评估不同澄清策略对模型的贡献。
    - **Term Exp.**（术语扩展）：移除术语扩展模块。
    - **Table Summary**（表格摘要）：移除表格摘要模块。
    - **All Removed**（全部移除）：移除所有澄清相关模块。
- **Formatter**（格式化器）：测试不同格式化方式对模型的影响。
    - **Markdown**：移除 Markdown 格式化。
    - **Html**：移除 Html 格式化。
- **Consist.**（一致性）：测试模型在没有一致性模块时的性能表现。
1. 检索器评估

为了评估不同检索器的召回率，对四个数据集进行了以下实验，并且对每个实验设置开启表格摘要和不开启表格摘要：

- **BM25**：传统的 TF-IDF 检索器。
- **ColBERT**：
    - 不使用 rerank：直接使用 ColBERT 生成的初始检索结果。
    - 使用 rerank：对初始检索结果进行重新排序。
- **DPR**：基于深度学习的稠密向量检索器。
- **FASSI 向量数据库**：高效向量检索数据库。

# Acknowledgments
I would like to express my sincere gratitude to the authors of the paper [“Tap4llm: Table provider on sampling, augmenting, and packing semi-structured data for large language model reasoning”](https://arxiv.org/abs/2312.09039) for providing valuable insights that influenced some of the ideas presented in this article. Additionally, I would like to thank PeiMa from the University of Leeds for her significant contributions to this project. Her expertise and support were instrumental in shaping the outcome of this work.

### Copyright Notice
© Wuyuhang, 2024. All rights reserved. This article is entirely the work of Wuyuhang from the University of Manchester. It may not be reproduced, distributed, or used without explicit permission from the author. For inquiries, please contact me at yuhang.wu-4 [at] postgrad.manchester.ac.uk.

## Reference

1. Zhang, T., Li, Y., Jin, Y. and Li, J., 2020. Autoalpha: an efficient hierarchical evolutionary algorithm for mining alpha factors in quantitative investment. *arXiv preprint arXiv:2002.08245*.
2. Li, L., Wang, H., Zha, L., Huang, Q., Wu, S., Chen, G. and Zhao, J., 2023. Learning a data-driven policy network for pre-training automated feature engineering. In *The Eleventh International Conference on Learning Representations*.
3. Chen, Z., Chen, W., Smiley, C., Shah, S., Borova, I., Langdon, D., Moussa, R., Beane, M., Huang, T.H., Routledge, B. and Wang, W.Y., 2021. Finqa: A dataset of numerical reasoning over financial data. *arXiv preprint arXiv:2109.00122*.
4. Chen, W., Zha, H., Chen, Z., Xiong, W., Wang, H. and Wang, W., 2020. Hybridqa: A dataset of multi-hop question answering over tabular and textual data. *arXiv preprint arXiv:2004.07347*.
5. Zhu, F., Lei, W., Huang, Y., Wang, C., Zhang, S., Lv, J., Feng, F. and Chua, T.S., 2021. TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. *arXiv preprint arXiv:2105.07624*.
6. Babaev, D., Savchenko, M., Tuzhilin, A. and Umerenkov, D., 2019, July. Et-rnn: Applying deep learning to credit loan applications. In *Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining* (pp. 2183-2190).
7. Ye, Y., Hui, B., Yang, M., Li, B., Huang, F. and Li, Y., 2023, July. Large language models are versatile decomposers: Decomposing evidence and questions for table-based reasoning. In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 174-184).
8. Cheng, Z., Xie, T., Shi, P., Li, C., Nadkarni, R., Hu, Y., Xiong, C., Radev, D., Ostendorf, M., Zettlemoyer, L. and Smith, N.A., 2022. Binding language models in symbolic languages.*arXiv preprint arXiv:2210.02875*.
9. Robertson, S. and Zaragoza, H., 2009. The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends® in Information Retrieval*, *3*(4), pp.333-389.
10. Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D. and Yih, W.T., 2020. Dense passage retrieval for open-domain question answering. *arXiv preprint arXiv:2004.04906*.
11. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J. and Wang, H., 2023. 
Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997*.
12. Lu, W., Zhang, J., Zhang, J. and Chen, Y., 2024. Large language model for table processing: A survey. *arXiv preprint arXiv:2402.05121*.
13. Jiang, J., Zhou, K., Dong, Z., Ye, K., Zhao, W.X. and Wen, J.R., 2023. Structgpt: A general framework for large language model to reason over structured data. *arXiv preprint arXiv:2305.09645*.
14. Sui, Y., Zou, J., Zhou, M., He, X., Du, L., Han, S. and Zhang, D., 2023. Tap4llm: Table provider on sampling, augmenting, and packing semi-structured data for large language model reasoning. *arXiv preprint arXiv:2312.09039*.
15. Bian, N., Han, X., Sun, L., Lin, H., Lu, Y., He, B., Jiang, S. and Dong, B., 2023. Chatgpt is a knowledgeable but inexperienced solver: An investigation of commonsense problem in large language models. *arXiv preprint arXiv:2303.16421*.
16. Lin, W., Blloshmi, R., Byrne, B., de Gispert, A. and Iglesias, G., 2023, July. LI-RAGE: Late Interaction Retrieval Augmented Generation with Explicit Signals for Open-Domain Table Question Answering. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* (pp. 1557-1566).
17. Li, X., Chan, S., Zhu, X., Pei, Y., Ma, Z., Liu, X. and Shah, S., 2023.  Are ChatGPT and GPT-4 general-purpose solvers for financial text analytics? A study on several typical tasks. *arXiv preprint arXiv:2305.05862*.
18. Chen, W., Wang, H., Chen, J., Zhang, Y., Wang, H., Li, S., Zhou, X. and Wang, W.Y., 2019. Tabfact: A large-scale dataset for table-based fact verification. *arXiv preprint arXiv:1909.02164*.
19. Aly, R., Guo, Z., Schlichtkrull, M., Thorne, J., Vlachos, A., Christodoulopoulos, C., Cocarascu, O. and Mittal, A., 2021. Feverous: Fact extraction and verification over unstructured and structured information. *arXiv preprint arXiv:2106.05707*.
20. Iyyer, M., Yih, W.T. and Chang, M.W., 2017, July. Search-based neural structured learning for sequential question answering. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1821-1831).
