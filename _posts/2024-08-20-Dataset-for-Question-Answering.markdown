---
title: "Dataset-for-Question-Answering"
layout: post
date: 2024-08-20 07:42
headerImage: false  
category: blog 
author: Pei Ma
---

# Overview of QA Datasets

Question Answering (QA) systems are a core area of research in natural language processing, aimed at extracting precise answers from various information sources. To meet diverse research needs and application scenarios, various QA datasets have emerged, systematically classified according to data format and task characteristics. This overview introduces several important QA datasets, covering a range of data types and task requirements. These datasets provide rich resources for testing and training, driving performance improvements and technological advancements in QA systems across different information formats.

## Open-Domain QA

- **[HybridQA](https://hybridqa.github.io/)**: Each question is aligned with a Wikipedia table and multiple paragraphs, linking table cells to paragraphs.
- **[OTT-QA](https://ott-qa.github.io/)**: Based on the HybridQA dataset, combining text and table evidence to create an open-domain QA benchmark.
- **[NQ-Table](https://arxiv.org/pdf/2103.12011)**: An open-domain QA dataset that combines table and text evidence.
- **[TAT-QA](https://nextplusplus.github.io/TAT-QA/)**: Similar to FinQA but includes both arithmetic questions and range questions.
- **[FinQA](https://finqasite.github.io/)**: A financial report dataset based on HybridQA, including only arithmetic questions (excluding range questions).
- **[MultiHiertt](https://github.com/psunlpgroup/MultiHiertt)**: A numerical reasoning dataset with hierarchical tables and text.
- **[TriviaQA](https://nlp.cs.washington.edu/triviaqa/)**: An open-domain QA dataset with multiple domains, providing rich evidence paragraphs for answering questions.
- **[Natural Questions](https://ai.google.com/research/NaturalQuestions)**: Provides natural language questions and their corresponding Wikipedia paragraphs, supporting open-domain QA research.
- **[HotpotQA](https://hotpotqa.github.io/)**: Provides multi-hop reasoning questions and answers, requiring information retrieval from multiple documents to answer.
- **[FEVER](https://fever.ai/)**: Primarily used for fact verification tasks, offering news articles and corresponding truth judgment questions.

## Table-Based QA

- **[WikiTableQuestions (WTQ)](https://ppasupat.github.io/WikiTableQuestions/)**: Contains 22,033 question-answer pairs with 2,108 tables; manually annotated questions and answers.
- **[WikiSQL](https://github.com/salesforce/WikiSQL)**: SQL-annotated tables from Wikipedia, including 81,000 questions and 24,000 tables.
- **[AIT-QA](https://github.com/IBM/AITQA)**: Involves hierarchical tables with 116 tables about U.S. flights and 515 questions.
- **[HiTab](https://arxiv.org/pdf/2108.06712v3)**: A dataset for hierarchical tables.
- **[TAPAS](https://github.com/google-research/tapas)**: A table-based QA dataset combining SQL queries and natural language questions, covering various table formats.
- **[TabFact](https://github.com/wtmo/TabFact)**: A fact verification dataset with tables, providing a large number of tables and corresponding fact-checking questions.
- **[TableQA](https://github.com/ABSA-HKUST/TableQA)**: Covers QA tasks across various table structures, focusing on extracting information from tables.
- **[SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253)**: Focuses on QA tasks involving structured data through SQL queries.

## Knowledge Graph QA

- **MetaQA (2018)**: A knowledge graph QA dataset.
- **GRAPHQ (2016)**: A knowledge graph QA dataset.
- **WebQSP (2016)**: A QA dataset based on Freebase.
- **CWQ (2018)**: An extended dataset based on WebQSP, using SPARQL query annotations to study semantic parsing.
- **LC-QuAD (2017)** and **LC-QuAD 2.0 (2019)**: Knowledge graph datasets based on Wikidata and DBpedia, including 30,000 questions with SPARQL query annotations.
- **GrailQA (2021)**: A knowledge graph QA dataset based on Freebase, with 64,000 questions and query annotations.
- **KQA Pro (2022)**: A dataset with NL questions and SPARQL annotations, including knowledge graphs from FB15k-237 and aligned Wikidata entities and 3,000 additional entities.
- **Freebase QA (2014)**: An early QA dataset using the Freebase knowledge graph, providing a large number of QA pairs about the knowledge graph.

## Knowledge Graph + Text QA

- Evaluates open-domain text QA datasets: TriviaQA (2017), WebQuestion (2013), CWQ (2018), WebQAP (2016), using knowledge graphs like WordNet, Freebase, ConceptNet.
- **WebQuestion (2013)**: Contains natural language questions targeting Freebase and corresponding answers, widely used in knowledge graph and text QA research.
- **ComplexWebQuestions (CWQ) (2018)**: Extends WebQuestion with complex SPARQL queries to test knowledge graph QA capabilities.

## Document-Based QA

- **SQuAD (2018)**: A document QA dataset providing a large-scale collection of paragraphs and question pairs for assessing model comprehension. [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- **TrivialQA (2017)**: A document QA dataset focusing on simple questions, mainly used for baseline model evaluation.
- **Natural Questions (NQ) (2019)**: Provides natural language questions and their corresponding Wikipedia paragraphs, supporting open-domain QA research. [Natural Questions](https://ai.google.com/research/NaturalQuestions)
- **SearchQA (2017)**: A search engine-based QA dataset, including questions and answers extracted from web search results.
- **DocVQA (2020)**: A document QA dataset focusing on extracting information from complex documents, including scanned documents and tables.
- **FEVEROUS (2020)**: A fact verification QA dataset, expanding the FEVER dataset with additional documents and verification tasks.
- **HotpotQA (2018)**: Provides multi-hop reasoning questions and answers, requiring information retrieval from multiple documents to answer. [HotpotQA](https://hotpotqa.github.io/)



Sure, here is the refined text in natural, academic English and formatted in Markdown:

---

# Overview of QA Datasets

Question Answering (QA) systems are a core area of research in natural language processing, aimed at extracting precise answers from various information sources. To meet diverse research needs and application scenarios, various QA datasets have emerged, systematically classified according to data format and task characteristics. This overview introduces several important QA datasets, covering a range of data types and task requirements. These datasets provide rich resources for testing and training, driving performance improvements and technological advancements in QA systems across different information formats.

## Open-Domain QA

- **[HybridQA](https://hybridqa.github.io/)**: Each question is aligned with a Wikipedia table and multiple paragraphs, linking table cells to paragraphs.
- **[OTT-QA](https://ott-qa.github.io/)**: Based on the HybridQA dataset, combining text and table evidence to create an open-domain QA benchmark.
- **[NQ-Table](https://arxiv.org/pdf/2103.12011)**: An open-domain QA dataset that combines table and text evidence.
- **[TAT-QA](https://nextplusplus.github.io/TAT-QA/)**: Similar to FinQA but includes both arithmetic questions and range questions.
- **[FinQA](https://finqasite.github.io/)**: A financial report dataset based on HybridQA, including only arithmetic questions (excluding range questions).
- **[MultiHiertt](https://github.com/psunlpgroup/MultiHiertt)**: A numerical reasoning dataset with hierarchical tables and text.
- **[TriviaQA](https://nlp.cs.washington.edu/triviaqa/)**: An open-domain QA dataset with multiple domains, providing rich evidence paragraphs for answering questions.
- **[Natural Questions](https://ai.google.com/research/NaturalQuestions)**: Provides natural language questions and their corresponding Wikipedia paragraphs, supporting open-domain QA research.
- **[HotpotQA](https://hotpotqa.github.io/)**: Provides multi-hop reasoning questions and answers, requiring information retrieval from multiple documents to answer.
- **[FEVER](https://fever.ai/)**: Primarily used for fact verification tasks, offering news articles and corresponding truth judgment questions.

## Table-Based QA

- **[WikiTableQuestions (WTQ)](https://ppasupat.github.io/WikiTableQuestions/)**: Contains 22,033 question-answer pairs with 2,108 tables; manually annotated questions and answers.
- **[WikiSQL](https://github.com/salesforce/WikiSQL)**: SQL-annotated tables from Wikipedia, including 81,000 questions and 24,000 tables.
- **[AIT-QA](https://github.com/IBM/AITQA)**: Involves hierarchical tables with 116 tables about U.S. flights and 515 questions.
- **[HiTab](https://arxiv.org/pdf/2108.06712v3)**: A dataset for hierarchical tables.
- **[TAPAS](https://github.com/google-research/tapas)**: A table-based QA dataset combining SQL queries and natural language questions, covering various table formats.
- **[TabFact](https://github.com/wtmo/TabFact)**: A fact verification dataset with tables, providing a large number of tables and corresponding fact-checking questions.
- **[TableQA](https://github.com/ABSA-HKUST/TableQA)**: Covers QA tasks across various table structures, focusing on extracting information from tables.
- **[SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253)**: Focuses on QA tasks involving structured data through SQL queries.

## Knowledge Graph QA

- **MetaQA (2018)**: A knowledge graph QA dataset.
- **GRAPHQ (2016)**: A knowledge graph QA dataset.
- **WebQSP (2016)**: A QA dataset based on Freebase.
- **CWQ (2018)**: An extended dataset based on WebQSP, using SPARQL query annotations to study semantic parsing.
- **LC-QuAD (2017)** and **LC-QuAD 2.0 (2019)**: Knowledge graph datasets based on Wikidata and DBpedia, including 30,000 questions with SPARQL query annotations.
- **GrailQA (2021)**: A knowledge graph QA dataset based on Freebase, with 64,000 questions and query annotations.
- **KQA Pro (2022)**: A dataset with NL questions and SPARQL annotations, including knowledge graphs from FB15k-237 and aligned Wikidata entities and 3,000 additional entities.
- **Freebase QA (2014)**: An early QA dataset using the Freebase knowledge graph, providing a large number of QA pairs about the knowledge graph.

## Knowledge Graph + Text QA

- Evaluates open-domain text QA datasets: TriviaQA (2017), WebQuestion (2013), CWQ (2018), WebQAP (2016), using knowledge graphs like WordNet, Freebase, ConceptNet.
- **WebQuestion (2013)**: Contains natural language questions targeting Freebase and corresponding answers, widely used in knowledge graph and text QA research.
- **ComplexWebQuestions (CWQ) (2018)**: Extends WebQuestion with complex SPARQL queries to test knowledge graph QA capabilities.

## Document-Based QA

- **[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)**: A document QA dataset providing a large-scale collection of paragraphs and question pairs for assessing model comprehension. 
- **TrivialQA (2017)**: A document QA dataset focusing on simple questions, mainly used for baseline model evaluation.
- **[Natural Questions](https://ai.google.com/research/NaturalQuestions)**: Provides natural language questions and their corresponding Wikipedia paragraphs, supporting open-domain QA research. 
- **SearchQA (2017)**: A search engine-based QA dataset, including questions and answers extracted from web search results.
- **DocVQA (2020)**: A document QA dataset focusing on extracting information from complex documents, including scanned documents and tables.
- **FEVEROUS (2020)**: A fact verification QA dataset, expanding the FEVER dataset with additional documents and verification tasks.
- **[HotpotQA](https://hotpotqa.github.io/)**: Provides multi-hop reasoning questions and answers, requiring information retrieval from multiple documents to answer. 

# Detailed QA Dataset Descriptions

## Open-Domain QA Datasets

### [HybridQA](https://hybridqa.github.io/)

**Introduction**: [HybridQA](https://hybridqa.github.io/), proposed by Chen et al. in "HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data," addresses the challenge of reasoning over heterogeneous information. Unlike existing QA datasets that primarily focus on homogeneous information, HybridQA requires reasoning over both tables and text. Each question is associated with a Wikipedia table and multiple paragraphs related to the table's entities. The questions are designed to require synthesizing both types of information, making the absence of either type insufficient for answering. Experiments show that while baseline models using only tables or text perform poorly, a hybrid model that integrates both types achieves significantly better results, though still trailing behind human performance. This dataset serves as a challenging benchmark for studying QA tasks involving heterogeneous information.

**Dataset and Code**: Available publicly at [GitHub - HybridQA](https://github.com/wenhuchen/HybridQA).

### TAT-QA

**Introduction**: TAT-QA, introduced by Zhu et al. in "TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance," is a large-scale QA dataset aimed at advancing research on complex financial data that combines tables and text. It involves questions requiring numerical reasoning over financial reports.

**Features**:
- Context includes a semi-structured table and at least two related paragraphs describing or supplementing the table content.
- Questions are generated by experts with rich financial knowledge, most being real-world application questions.
- Answers vary in format, including single span, multiple spans, and free-form responses.
- Answering often requires diverse numerical reasoning abilities such as addition, subtraction, multiplication, division, counting, comparison, sorting, and combinations thereof.
- Along with real answers, the dataset includes the corresponding reasoning processes and scales, when applicable.

**Size**: TAT-QA contains 16,552 questions based on 2,757 mixed-context financial reports.

### FinQA

**Introduction**: FinQA, proposed by Chen et al. in "FinQA: A Dataset of Numerical Reasoning over Financial Data," focuses on deep QA over financial data, aiming to automate the analysis of extensive financial documents. Compared to general domain tasks, the financial domain involves complex numerical reasoning and understanding of heterogeneous representations.

**Features**:
- Provides QA pairs created by financial experts, with annotated gold reasoning programs to ensure full interpretability.
- Introduces baseline models and conducts extensive experiments, showing that popular large pre-trained models significantly lag behind experts in acquiring financial knowledge and performing complex multi-step numerical reasoning.
- Dataset and code are available at [GitHub - FinQA](https://github.com/czyssrs/FinQA).

### [MultiHiertt](https://github.com/psunlpgroup/MultiHiertt)

**Introduction**: [MultiHiertt](https://github.com/psunlpgroup/MultiHiertt), introduced by Wang et al. in "MultiHiertt: A Multi-Modal Dataset for Hierarchical Table-Based Numerical Reasoning," provides a dataset specifically for numerical reasoning over hierarchical tables. Unlike other table-based datasets that focus on flat or moderately complex tables, [MultiHiertt](https://github.com/psunlpgroup/MultiHiertt) features hierarchical tables that reflect the multi-level nature of real-world financial documents.

**Features**:
- Contains hierarchical tables with multiple levels of nesting, reflecting complex financial data structures.
- Focuses on multi-modal reasoning, requiring combining information from different hierarchical levels and types.

**Size**: [MultiHiertt](https://github.com/psunlpgroup/MultiHiertt) includes a range of hierarchical tables and associated questions to test different levels of numerical reasoning.


Certainly! Here’s the refined version of the content in academic English with Markdown formatting:

---

### [HotpotQA](https://hotpotqa.github.io/)

**Introduction**: [HotpotQA](https://hotpotqa.github.io/) was introduced by Yang et al. in "[HotpotQA](https://hotpotqa.github.io/): A Dataset for Diverse, Explainable Multi-hop Question Answering". This dataset, based on English Wikipedia, includes approximately 113,000 crowd-sourced questions that require information from two introductory paragraphs of Wikipedia articles to answer.

**Features**:
- Covers various reasoning strategies, including questions involving missing entities, intersection questions (e.g., "What satisfies both attribute A and attribute B?"), and comparison questions.
- Provides ten paragraphs, including golden paragraphs. In the open-domain full Wikipedia setting, models are given only the question and the entire Wikipedia.
- Evaluation metrics include answer accuracy (measured by Exact Match (EM) and word F1 score) and explainability (assessing the alignment of predicted supporting sentences with human-annotated sentences).

### [FEVER](https://fever.ai/)

**Introduction**: FEVER (Fact Extraction and VERification) was proposed by Thorne et al. in "FEVER: a large-scale dataset for Fact Extraction and VERification". This dataset is used for verifying facts from textual sources and includes 185,445 statements generated by altering sentences from Wikipedia, which are subsequently verified.

**Features**:
- Statements are classified as "support", "refute", or "not enough information".
- For the first two categories, annotators record the sentences required to make a judgment.
- Developed a pipeline approach and compared it with well-designed predictors, demonstrating that FEVER is a challenging test platform that advances research in statement verification from textual sources.

**Dataset and Code**: [FEVER](https://fever.ai/)

## Table-Based Question Answering Datasets

### [WikiTableQuestions (WTQ)](https://ppasupat.github.io/WikiTableQuestions/)

**Introduction**: [WikiTableQuestions (WTQ)](https://ppasupat.github.io/WikiTableQuestions/), introduced by Pasupat et al. in "Compositional Semantic Parsing on Semi-Structured Tables", is based on HTML tables and includes 22,033 question-answer pairs. The questions were crafted by Amazon Mechanical Turk workers, and the tables were sourced from Wikipedia, each containing at least 8 rows and 5 columns.

**Features**:
- Questions are manually crafted by users rather than through predefined templates, showcasing significant linguistic variability.
- Compared to previous knowledge base datasets, it covers nearly 4,000 unique column headers and involves more relationships than closed-domain datasets.
- Questions span a wide range of domains and require various operations, including table retrieval, aggregation, superlatives (e.g., maximum, minimum), arithmetic calculations, joining, and merging.

### [AIT-QA](https://github.com/IBM/AITQA)

**Introduction**: [AIT-QA](https://github.com/IBM/AITQA) (Airline Industry Table QA), proposed by Katsis et al. in "AIT-QA: Question Answering Dataset over Complex Tables in the Airline Industry", is a domain-specific table-based question answering dataset. It includes 515 questions crafted by human annotators based on 116 tables extracted from the annual reports of major airlines from the SEC filings (2017-2019). The dataset also contains annotations on the nature of questions, highlighting those requiring hierarchical headers, domain-specific terminology, and synonym variations.

**Features**:
- The table layouts are more complex, presenting greater challenges compared to traditional table-based QA datasets.
- Includes annotations indicating questions that require hierarchical headers, domain-specific terminology, and synonym variations.

**Dataset and Code**: Available publicly: [GitHub - AIT-QA](https://github.com/IBM/AITQA)

### [TabFact](https://tabfact.github.io/)

**Introduction**: [TabFact](https://tabfact.github.io/), introduced by Chen et al. in "TabFact: A Large-scale Dataset for Table-based Fact Verification", is a large-scale dataset containing 117,854 manually annotated statements involving 16,573 Wikipedia tables. The relationships in these statements are classified as "entailed" or "refuted". [TabFact](https://tabfact.github.io/) is the first dataset designed to evaluate language reasoning over structured data, involving both symbolic and linguistic reasoning skills.

**Features**:
- Provides a large-scale dataset for fact verification based on tables.
- Relationship classification as "entailed" or "refuted", challenging the model's language reasoning and structured data processing capabilities.

**Dataset and Code**: Available publicly: [TabFact](https://tabfact.github.io/)

### [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253) (SequentialQA)

**Introduction**: [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253), proposed by Iyyer et al. in "Search-based Neural Structured Learning for Sequential Question Answering", explores the task of answering a sequence of related questions over HTML tables. The dataset comprises 6,066 sequences, totaling 17,553 questions.

**Features**:
- Focuses on answering a series of related questions within HTML tables.
- Provides a rich set of sequential questions, covering various problem orders and interrelationships.

**Dataset and Code**: Available publicly: [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253)

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
