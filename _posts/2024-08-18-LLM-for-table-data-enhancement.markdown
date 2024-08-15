---
title: "LLM for table data enhancement"
layout: post
date: 2024-08-15 09:19
headerImage: false  
category: blog 
author: Pei Ma

---

With the widespread application of large language models (LLMs) in the field of natural language processing (NLP), these models have demonstrated remarkable capabilities across various tasks, including text generation, question answering systems, and sentiment analysis. However, despite their outstanding performance in handling unstructured data, LLMs still face numerous challenges when processing structured data, particularly tabular data. The structured nature of tabular data and its rich semantic information impose higher demands on LLMs, rendering traditional text processing methods often inapplicable to such data.

This article aims to summarize and discuss the key technologies and methodologies for processing tabular data. We conduct an in-depth analysis of significant literature and methods related to LLMs' handling of tabular data. These studies attempt to address the challenges encountered by LLMs in processing tabular data, including encoding, querying, and generating tabular data. Through a detailed exploration of technologies such as ColBERT, ColBERTv2, DPR, and RAFT, we present the major advancements and innovations in the field of tabular data processing by LLMs. These technologies not only enhance the understanding and retrieval capabilities of tabular data but also provide important references for future research.

## LLM for Table

[Large Language Model for Table Processing](https://arxiv.org/pdf/2402.05121)

### 1. Classification of LLM Methods for Table Processing

Research on processing tabular data primarily focuses on two major approaches: training-based methods and prompt-based methods, specifically including:

1. **Training-based Methods**:
   - **Task-specific fine-tuning**: Examples include TaPas and TaBERT, which enhance the performance of table-related tasks by adjusting the model architecture and training objectives.
   - **Instruction fine-tuning**: Techniques like TableLlama and Table-GPT improve the model's performance on unseen tasks through fine-tuning on multiple datasets.
   - **Retrieval-augmented methods**: ITR and LI-RAGE, for instance, split large tables into sub-tables and jointly train retrievers and readers.

2. **Prompt-based Methods**:
   - **Table serialization**: This involves converting tables into a linear text format to make them more accessible to LLMs.
   - **Example selection for few-shot learning**: Selecting examples most relevant to the target task to improve model performance.

3. **Agent-based Methods**:
   - **Decomposition of complex tasks**: Techniques like DIN-SQL enhance accuracy by breaking down complex tasks into smaller sub-tasks.
   - **Action definition**: Abstracting software tools' APIs into actions, facilitating LLMs in making calls.
   - **Reflection and correction**: Improving model accuracy by generating multiple reasoning paths and selecting the most consistent answer or through self-correction.
   - **Multi-task framework**: StructGPT, for example, can handle multiple table-related tasks.

### Summary of Specific Methods

1. **Task-specific Fine-tuning**:
   - **TaPas**: Extends the BERT model architecture with table pre-training and fine-tuning.
   - **TaBERT**: Encodes table content related to the input statement using a vertical attention mechanism.
   - **TURL**: Encodes information from table components as separate input embeddings and integrates them.

2. **Instruction Fine-tuning**:
   - **Table-GPT**: Constructs instruction fine-tuning datasets using a synthesis-enhanced approach.
   - **TableLlama**: Utilizes real data from existing datasets for instruction fine-tuning.
   - **Magicoder**: Collects open-source code snippets and generates programming problems and solutions for instruction fine-tuning.

3. **Retrieval-augmented Methods**:
   - **ITR**: Splits large tables into sub-tables and jointly trains retrievers and readers.
   - **DB-GPT**: Supports various functions such as retrieval-augmented generation, fine-tuning, and agents.

4. **Table Serialization**:
   - Linearizes table content and inserts column delimiters.
   - The table schema can be represented as plain text or through a CREATE TABLE statement.

5. **Example Selection for Few-shot Learning**:
   - Selects examples most relevant to the target task, balancing quality and quantity.

6. **Decomposition of Complex Tasks**:
   - **DIN-SQL**: Breaks down text-to-SQL tasks into sub-tasks, generating intermediate sub-queries.

7. **Action Definition**:
   - **SheetCopilot**: Models existing spreadsheet software APIs as atomic actions through embedding and clustering methods.
   - **ReAcTable**: Extends the ReAct framework, defining three actions: generating SQL queries, generating Python code, and directly answering questions.

8. **Reflection and Correction**:
   - Generates multiple reasoning paths and selects the most consistent answer.
   - Adopts a proposal and correction mechanism, reflecting and improving past actions.

9. **Multi-task Framework**:
   - **StructGPT**: Addresses multiple table tasks by developing three actions for web tables, databases, and knowledge graphs.

This review systematically summarizes the latest advancements and specific methods of LLMs in table processing tasks, providing references for future research and applications.

## LLM on Tabular Data (Retriever)

[Large Language Models (LLMs) on Tabular Data: Prediction, Generation, and Understanding](https://arxiv.org/pdf/2402.17944)

### 5.2 General Capabilities of Large Language Models in Question Answering Tasks

Table 8 lists papers that study the performance of large language models (LLMs) in question answering (QA) and reasoning tasks, along with the models explored. While GPT-3.5 and GPT-4 are the most popular LLMs, these models have not been specifically optimized for table tasks. However, when combined with certain prompt engineering techniques (e.g., Chain of Thought, CoT), they perform well in executing complex table reasoning tasks.

### Numerical QA

Numerical QA tasks involve mathematical reasoning, such as "What is the average payment per transaction by American Express?" This type of mathematical reasoning task is prevalent in many practical applications, such as processing financial documents and annual reports. Akhtar et al. (2023) found that FlanT5 and GPT-3.5 outperform other models in various numerical reasoning tasks. On the DOCMATH-EVAL (Zhao et al., 2023d) dataset, GPT-4 with CoT significantly outperforms other LLMs, while open-source LLMs (e.g., LLaMa-2, Vicuna, Mistral, Starcoder, MPT, Qwen, AquilaChat2, etc.) perform poorly.

### Text2SQL

Liu et al. (2023c) designed a question matcher that identifies three types of keywords: 1) column-related terms, 2) restriction-related phrases (e.g., "Top 10"), and 3) algorithm or module keywords. Once these keywords are identified, the module merges the specific restrictions associated with each column into a unified combination, then matches it with the SQL algorithm or module indicated by the third keyword type. Zhang et al. (2023d) chose a more straightforward approach, allowing LLaMa-2 to generate SQL statements based on the question and table schema. Sun et al. (2023b) fine-tuned PaLM-2 on the Text2SQL task, achieving remarkable results on the Spider dataset. OpenTab (Kong et al., 2024) developed an open-domain table QA framework based on LLMs, combining it with a SQL generation module. Today, top models on Spider include those by Dong et al. (2023), Gao et al. (2024), and Pourreza & Rafiei (2023), all of which build on OpenAI's GPT model. SQL generation is highly popular in the industry, with many open-source fine-tuned models available.

### The Impact of Model Size on Performance

Chen (2023) found that model size does indeed matter: on WebTableQuestions, comparing GPT-3 models of 6.7B and 175B, the smaller model achieved only half the score of the larger model. On TabFact, they found that the accuracy of smaller models (<=6.7B) was almost random.

### To Fine-tune or Not to Fine-tune?

Some larger models have been fine-tuned on various table tasks, including QA and fact verification tasks. Li et al. (2023d) found that fine-tuning always helps improve performance on various table tasks, especially in zero-shot settings where the improvement is most significant. Ye et al. (2023b) used the PASTA (Gu et al., 2022) model to achieve a higher score (93.00%) on TabFact compared to GPT-3 Codex (code-davinci-002) with a score of 85.60%. PASTA was pre-trained on a synthetic corpus of 1.2 million entries composed of Wikipedia tables for six types of sentence-table fill-in-the-blank tasks. This suggests that fine-tuning LLMs on table tasks still offers some advantages.

However, fine-tuning is less common compared to other methods working on prediction and generation tasks. This might be because LLMs (such as GPT-3.5, GPT-4) perform well in out-of-the-box QA tasks. In SQL generation on Spider, DIN-SQL (Pourreza & Rafiei, 2023) and DAIL-S

QL (Sun et al., 2023b) each achieve scores above 90% with limited fine-tuning, indicating that prompt engineering techniques and retrieval methods could replace fine-tuning to some extent.

### 5.3 Special Data Considerations

When selecting training and evaluation data, it's essential to account for specific table characteristics. Some datasets use manually annotated data, but others use synthetic or heuristic methods to generate labels. As a result, future research should pay attention to annotation quality, given that annotation standards vary across different domains and regions.

## LI-RAGE

[LI-RAGE: Late Interaction Retrieval Augmented Generation with Explicit Signals for Open-Domain Table Question Answering](https://aclanthology.org/2023.acl-short.133.pdf)

### Summary of the Article

The LI-RAGE framework is a novel approach to open-domain table question answering (TableQA). By combining a late interaction (LI) model with retrieval-augmented generation (RAGE) loss incorporating explicit signals, this method significantly improves the performance of table question answering. Compared to traditional retriever-reader pipelines, LI-RAGE offers enhanced accuracy and reliability through the following improvements:

1. **Late Interaction Model (LI):** Utilizes the ColBERT model to encode both the query and the table on a word-by-word basis, capturing more fine-grained interaction information and thereby improving table retrieval effectiveness.
2. **Joint Training with RAGE Loss:** Combines the signals from the retriever and reader in joint training to optimize the effectiveness of both table retrieval and answer generation.
3. **Binary Relevance Token:** Introduces a binary relevance token (yes/no) before generating the answer to indicate whether the table is relevant to the query, thereby enhancing the reliability of the generated answer.

### Example Question

Consider the question: "Which country has the largest population?"

### Table Dataset

Assume the following table data:

**Table 1:**

| Country | Population |
|---------|------------|
| China   | 1,411 million |
| India   | 1,366 million |
| USA     | 331 million  |

**Table 2:**

| Country | Area |
|---------|------|
| Russia  | 17 million km² |
| Canada  | 9.98 million km² |
| China   | 9.6 million km²  |

**Table 3:**

| City   | Population |
|--------|------------|
| New York | 8 million  |
| Tokyo    | 14 million |
| Shanghai | 24 million |

### End-to-End Example

#### 1. Table Retrieval (Retriever)

The retriever selects the table most relevant to the query from the table corpus. In this example, the retriever might select Table 1 since it contains information related to countries and their populations.

**Retrieval Result:**  
Selected Table 1:

| Country | Population |
|---------|------------|
| China   | 1,411 million |
| India   | 1,366 million |
| USA     | 331 million  |

#### 2. Answer Generation (Reader)

The answer generator model takes the query and the retrieved table as input and generates the answer. In this example, the answer generator combines the question "Which country has the largest population?" with Table 1, identifying the maximum value corresponding to the population and generating the answer: "China."

#### 3. Binary Relevance Token

To ensure that the selected table by the answer generator is reliable, a binary relevance token is added before generating the answer. During training, the system learns that answers generated from a golden table are prefixed with "yes," whereas those from a non-golden table are prefixed with "no." In this case, since the generated answer is derived from the golden table (Table 1), the answer is prefixed with "yes."

**Final Output:**  
The answer generator outputs: "yes China."

#### 4. Filtering and Final Answer Determination

During inference, if the answer generator's output is prefixed with "yes," the answer is deemed reliable. The system prioritizes answers marked with "yes"; if all candidate answers are prefixed with "no," the system selects the final answer based on the confidence score of the answer generator. In this case, the system identifies the "yes" prefix, confirms the answer's reliability, and outputs the final answer: "China."

### Conclusion

The above process demonstrates the complete workflow of an open-domain table question answering system from input query to final answer generation. Through the LI-RAGE framework, the system not only efficiently retrieves relevant information from vast table data but also ensures the reliability of the answer through binary relevance tokens.

## TAP4LLM

[TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning](https://arxiv.org/pdf/2312.09039)

### Detailed Summary of TAP4LLM

### Application of Large Language Models to Tabular Data

As large language models (LLMs) advance in natural language processing, researchers have begun exploring their application to other modalities, such as vision and speech. However, directly applying traditional LLMs to the tabular domain presents two main challenges:

1. **Global Table Understanding:** LLMs like GPT have a token length limitation, making it difficult to read and understand large tables comprehensively, thereby restricting their ability to grasp global information.
2. **Generalization to the Tabular Domain:** These models are primarily trained on natural language, resulting in weaker generalization when handling tabular data.

Despite some research attempting to combine natural language processing with table data analysis, the performance of LLMs in table question answering remains limited.

### Table Augmentation Techniques

Table augmentation techniques aim to improve the generalization performance and robustness of machine learning models. To enhance LLMs' performance in the tabular domain, researchers have explored various augmentation methods, including the integration of structured knowledge, common sense knowledge, and analytical knowledge. Studies have shown that leveraging domain-specific metadata or knowledge graphs can significantly improve LLMs' understanding of tabular data. For example:
- **Jena et al. (2022)** proposed semi-automatically converting existing table data to create diversified natural language inference instances, improving zero-shot performance.
- **He et al. (2023)** introduced a multi-task metadata model that accurately infers analytical metadata for tables using field distribution and knowledge graph information, demonstrating its application in intelligent data analysis products.

### Core Components of TAP4LLM

TAP4LLM addresses the main challenges of comprehensive table understanding through three core components:

1. **Table Sampling:** Selecting and extracting the most relevant rows and columns from the table based on the query.
   - **Rule-based Sampling:** Uses predefined criteria or rules, such as random sampling, uniform sampling, and content snapshot sampling.
   - **Embedding-based Sampling:** Selects rows and columns based on semantic and contextual representation, employing methods such as semantic-based sampling and centroid-based sampling.
   - **LLM-based Sampling:** Utilizes powerful LLMs to predict the indices of table rows and columns, though this approach is computationally expensive.
2. **Table Augmentation:** Enriching table information by adding external knowledge and metadata.
   - **Metadata-based Augmentation:** Includes the addition of information like dimensions/metrics, semantic field types, table size, statistical features, and header hierarchies.
   - **Retrieval-based Augmentation:** Acquires relevant content from external knowledge bases through a document retrieval system to reduce hallucination or factual errors.
   - **Self-consistency Augmentation:** Enhances the model's reasoning capability through iterative generation and refinement of queries and responses.
3. **Table Packing and Serialization:** Manages token allocation by packing tables and augmented information into sequences suitable for LLMs.
   - Empirical studies show that a sub-table length to augmentation information length ratio of 5:5 or 4:6 generally yields the best performance.
   - Supports multiple serialization formats, such as HTML, XML, JSON, CSV, NL+Sep, and Markdown.

### Conclusion

TAP4LLM addresses the main challenges of comprehensive table understanding through table sampling, table augmentation, and table packing and serialization, enhancing the effectiveness of LLMs in table reasoning tasks. This method is not only applicable to table modeling but can also play a significant role in fields such as finance and transportation, promoting research based on tabular data.

### Limitations

Code generation methods have been proposed to convert natural language queries into executable code or structured representations (Cheng et al., 2023; Gemmell and Dalton, 2023). This research direction is important, but due to space constraints, it is not explored in depth in this study. Current empirical research is primarily focused on English scenarios, with discussions on multilingual capabilities left for future research.

### Example: Using TAP4LLM for Table Data Analysis

Suppose there is a financial data table containing a company's quarterly financial reports over the past few years. The columns of the table include year, quarter, revenue, expenditure, net profit, and debt-to-equity ratio. The goal is to generate an accurate analysis based on the natural language query, "What is the trend of the company's quarterly net profit over the past five years?"

### 1. Table Sampling

**Initial Table (T):**

| Year | Quarter | Revenue | Expenditure | Net Profit | Debt-to-Equity Ratio |
|------|---------|---------|-------------|------------|----------------------|
| 2019 | Q1      | 1000    | 800         | 200        | 50%                  |
| 2019 | Q2      | 1100    | 850         | 250        | 48%                  |
| 2019 | Q3      | 1050    | 820         | 230        | 49%                  |
| 2019 | Q4      | 1200    | 900         | 300        | 47%                  |
| 2020 | Q1      | 1300    | 950         | 350        | 46%                  |
| ...  | ...     | ...     | ...         | ...        | ...                  |
| 2023 | Q4      | 1600    | 1200        | 400        | 45%                  |

To answer the query "What is the trend of the company's quarterly net profit over the past five years?" the

 model applies TAP4LLM's table sampling techniques. Using embedding-based sampling, it extracts the rows corresponding to the net profit column, which are most relevant to the query.

**Sampled Table:**

| Year | Quarter | Net Profit |
|------|---------|------------|
| 2019 | Q1      | 200        |
| 2019 | Q2      | 250        |
| 2019 | Q3      | 230        |
| 2019 | Q4      | 300        |
| 2020 | Q1      | 350        |
| ...  | ...     | ...        |
| 2023 | Q4      | 400        |

### 2. Table Augmentation

To enhance the model's reasoning capabilities, TAP4LLM applies self-consistency augmentation, iteratively generating and refining queries to ensure accurate analysis. This might involve integrating the financial performance trends for each quarter into a broader context, perhaps by incorporating external knowledge about the economic climate during these years.

### 3. Table Packing and Serialization

Finally, the augmented table information is packed into a format suitable for LLMs. In this case, the financial data might be serialized into a JSON or Markdown format, enabling efficient token management and allowing the LLM to process the data effectively.

**Packed and Serialized Table (in JSON format):**

```json
{
  "Year": "2023",
  "Quarter": "Q4",
  "Net Profit": "400",
  "Previous_Trend": [
    {"Year": "2023", "Quarter": "Q3", "Net Profit": "380"},
    {"Year": "2023", "Quarter": "Q2", "Net Profit": "370"},
    {"Year": "2023", "Quarter": "Q1", "Net Profit": "360"}
  ],
  "Contextual_Info": "Company's net profit has shown a consistent upward trend over the past five years."
}
```

### Conclusion

The final output provides a comprehensive analysis of the company's quarterly net profit trend, supported by augmented and packed table data. By employing TAP4LLM, LLMs can accurately and efficiently handle complex queries over large tables, ensuring reliable and insightful results in financial data analysis.

### Example of Processing with TAP4LLM

**Task:** Identify the upward or downward trend of the company's quarterly net profit.

**Query:** "What is the trend of the company's quarterly net profit over the past five years?"

**Processed Answer:** "The company's quarterly net profit has shown a consistent upward trend over the past five years, with the highest profit recorded in Q4 2023."


## [ColBERT](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075)

### Innovations

1. **Delayed Interaction Framework**: [ColBERT](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075) introduces a delayed interaction framework by decoupling the encoding processes of queries and documents. This allows document representations to be precomputed, thereby reducing the computation required during online queries.
2. **MaxSim Operation**: When assessing the relevance between queries and documents, [ColBERT](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075) employs the MaxSim operation, which computes the maximum cosine similarity or L2 distance between each query embedding and document embedding, summing these maximum similarity values. This approach is both simple and efficient.
3. **Shared BERT Encoder**: [ColBERT](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075) utilizes a single BERT model shared between the query and document encoders, distinguishing inputs with special tokens ([Q] and [D]) for queries and documents respectively. This method conserves computational resources while maintaining the model's contextual understanding.
4. **Segmentation and Filtering**: The document encoder filters out embeddings of punctuation to reduce computational load and storage space.
5. **Vector-based Retrieval**: By leveraging existing vector similarity search libraries (such as Faiss), [ColBERT](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075) achieves efficient end-to-end retrieval through pruning operations from large document collections.

### Advantages

1. **High Computational Efficiency**: By precomputing document representations and employing a delayed interaction mechanism, ColBERT significantly reduces computation during query processing, achieving two orders of magnitude speed improvement compared to other BERT-based models.
2. **Efficient Space Utilization**: Through normalization and dimensionality reduction of document embeddings, ColBERT markedly decreases storage requirements, making it more feasible for practical applications.
3. **Scalability**: The architecture of ColBERT allows for processing large-scale document collections without sacrificing accuracy, especially when using vector similarity search for pruning, greatly enhancing retrieval efficiency.
4. **End-to-End Retrieval Capability**: ColBERT can be used not only for re-ranking pre-retrieved document sets but also for direct end-to-end retrieval from large document collections, improving the overall recall and precision of the retrieval system.

### Problems Addressed

1. **High Computational Cost**: Traditional BERT-based ranking models are extremely time-consuming for query-document pairs. ColBERT reduces online computation costs through delayed interaction and precomputation mechanisms.
2. **Long Response Time**: High computational costs result in lengthy query response times, negatively impacting user experience. ColBERT significantly reduces query latency through more efficient computation and retrieval mechanisms.
3. **Large Storage Space**: Deep language models typically require substantial storage for document representations. ColBERT reduces storage requirements through normalization and dimensionality reduction.
4. **Trade-off Between Retrieval Accuracy and Efficiency**: Existing methods often sacrifice accuracy to improve retrieval efficiency. ColBERT enhances retrieval efficiency without compromising accuracy through efficient delayed interaction and vector similarity search.

### Detailed and Vivid Example of ColBERT Usage

### Background

Suppose you are using an academic paper database containing millions of papers. You are researching "the benefits of machine learning" and want to find the most relevant papers. This is where ColBERT can assist you.

### 1. Offline Preprocessing and Encoding of Documents

Before any queries are made, we preprocess and encode each paper in the database. This is an offline process, akin to cataloging and numbering all the books in a library.

1. **Segmentation**: Break down each paper into words. For instance, "Machine learning is a method of data analysis that can automatically build analytical models" is segmented into "Machine," "learning," "is," "a," "method," "of," "data," "analysis," "that," "can," "automatically," "build," "analytical," "models," etc.
2. **Adding Markers**: Add special markers at the beginning of each paper, such as "[D]," to indicate that it is a document.
3. **BERT Encoding**: Use the BERT model to encode each word, converting them into contextually meaningful vector representations. This is like generating a unique numerical signature for each word.
4. **Filtering Unrelated Information**: Remove punctuation and other irrelevant information to retain important words.
5. **Normalization and Dimensionality Reduction**: Normalize and reduce the dimensionality of these vectors to make their representation more compact and efficient, similar to compressing large files into smaller ones for easier storage and processing.
6. **Storing Embeddings**: Store the processed vectors in a database for future use.

### 2. Query Preprocessing and Encoding

When you enter the query "the benefits of machine learning," ColBERT processes this query immediately, which is an online operation.

1. **Segmentation**: Break down the query into words, such as "Machine," "learning," "benefits," "of."
2. **Adding Markers**: Add special markers at the beginning of the query, such as "[Q]," to indicate that it is a query.
3. **Padding and BERT Encoding**: Pad the query to a fixed length and input it into the BERT model to generate context vectors for each word. These vectors represent the meaning of each word in the query and their relationships.
4. **Normalization and Dimensionality Reduction**: Normalize and reduce the dimensionality of these vectors to match the format of document vectors.

### 3. Delayed Interaction and Similarity Computation

Next, ColBERT finds the most relevant papers using delayed interaction and similarity computation.

1. **Loading Document Embeddings**: Load all precomputed document vector representations from the database.
2. **MaxSim Calculation**: For each query word vector, find the maximum similarity with all word vectors in the document. This is like finding the best matching puzzle pieces.
3. **Summing Similarities**: Sum the maximum similarity values for each query word with document words to obtain an overall similarity score. This score represents how relevant the document is to the query.

### 4. Document Ranking and Retrieval

Finally, rank documents based on similarity scores and return the top k documents.

1. **Document Ranking**: Sort all candidate documents by similarity score, similar to ranking exam results from highest to lowest.
2. **Returning Results**: Return the top k documents with the highest scores, which are the most relevant papers to your query.

### Vivid Example

Imagine you are in a library looking for books related to "the benefits of machine learning." The librarian (ColBERT) has previously cataloged and tagged all the books in detail. When you make your request, the librarian quickly reviews the digital content of each book (query encoding and similarity computation), finds the most relevant ones, ranks them, and provides you with the best matches. This all happens very swiftly because the librarian has done a lot of preparation in advance.

In this way, ColBERT ensures efficient handling of large datasets while providing fast response times and high-quality results.

[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)

## ColBERT v2

[ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/pdf/2112.01488)

### Key Improvements and Optimizations in [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/pdf/2112.01488)

1. **Residual Compression Mechanism**: A major innovation in [ColBERTv2](https://arxiv.org/pdf/2112.01488) is the residual compression mechanism. By encoding each embedding vector as an index of its nearest centroid and quantization residual, ColBERTv2 significantly reduces storage requirements. This improvement lowers storage costs considerably without sacrificing model quality.
2. **Denoising Supervision**: [ColBERTv2](https://arxiv.org/pdf/2112.01488) introduces a new supervision strategy, including cross-encoder distillation and hard negative mining. This method selects challenging negative samples to avoid rewarding false positives or penalizing false negatives, thus enhancing training effectiveness and model quality.
3. **Efficient Indexing and Retrieval**:
    - **Centroid Selection**: During indexing, [ColBERTv2](https://arxiv.org/pdf/2112.01488) optimizes paragraph representations through centroid selection.
    - **Paragraph Encoding**: The BERT encoder is used to compress output embeddings, assigning each embedding to the nearest centroid and computing quantization residuals.
    - **Inverted Index**: To support fast nearest neighbor search, [ColBERTv2](https://arxiv.org/pdf/2112.01488) groups embedding IDs corresponding to each centroid and maintains inverted lists, enabling rapid retrieval of similar token-level embeddings.
4. **Optimized Retrieval Process**:
    - **Candidate Generation**: For each vector in the query, the nearest centroids are found, and inverted lists are used to identify paragraph embeddings close to these centroids. These embeddings are decompressed, and cosine similarity with query vectors is computed.
    - **Scoring and Maximization**: Scores are grouped by paragraph ID, and scores for the same paragraph are maximized and reduced. This is similar to finding the best matching pieces in a puzzle.

### Full Process Example of [ColBERT

v2](https://arxiv.org/pdf/2112.01488)

### Background

Consider a scenario where you are searching for recent research papers on "machine learning advancements" in a database. Here's how [ColBERTv2](https://arxiv.org/pdf/2112.01488) would enhance this process:

### 1. Offline Preprocessing and Encoding of Documents

1. **Preprocessing and Encoding**:
    - **Segmentation and Encoding**: Similar to [ColBERT](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075), segment and encode each paper, but with residual compression for space efficiency.
    - **Storing Quantized Embeddings**: Store embeddings efficiently using centroids and quantization residuals.

2. **Index Construction**:
    - **Optimized Indexing**: Create an inverted index grouping embeddings by their nearest centroids for fast retrieval.

### 2. Query Processing and Encoding

1. **Encoding**: Encode the query with BERT, and perform similar normalization and quantization as with documents.

### 3. Efficient Retrieval

1. **Centroid-based Candidate Generation**:
    - **Find Nearest Centroids**: Identify centroids closest to the query vectors.
    - **Retrieve Embeddings**: Use inverted lists to gather candidate embeddings, decompress, and score them based on similarity to the query.

2. **Scoring and Ranking**:
    - **Maximizing Scores**: Rank and maximize scores to retrieve the most relevant papers efficiently.

### Vivid Example

Imagine you are a researcher looking for the latest papers on "machine learning advancements." Instead of manually reviewing each paper, [ColBERTv2](https://arxiv.org/pdf/2112.01488) preprocesses and indexes the papers in advance, using efficient storage techniques. When you make a query, it quickly finds and retrieves the most relevant papers by leveraging optimized indexing and centroid-based retrieval.

[ColBERTv2](https://arxiv.org/pdf/2112.01488) further refines the retrieval process, improving both efficiency and accuracy while reducing storage requirements, thus providing a highly effective tool for handling extensive document collections.

## DPR

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906)

### Example Scenario

Suppose a user asks the system a question: "What is photosynthesis?" The following steps illustrate how DPR retrieves relevant information:

### Step 1: Query Encoding

- **Input**: User's question "What is photosynthesis?"
- **Processing**: The question is first fed into a pre-trained Transformer model (e.g., BERT). This model transforms the text into a high-dimensional vector (typically 768 dimensions or more, depending on the model architecture).
- **Output**: Dense vector representation of the question.

### Step 2: Document Encoding

- **Preprocessing**: Prior to this step, the system has already encoded potential answers or information sources (e.g., Wikipedia entries, textbook paragraphs) into vectors and stored them in a vector database.
- **Database**: Contains precomputed vector representations of numerous documents.

### Step 3: Vector Similarity Calculation

- **Comparison**: The system now compares the query vector with each document vector in the database. Comparison typically uses cosine similarity.
- **Ranking**: Documents are ranked based on similarity scores, from highest to lowest.

### Step 4: Selecting Top Documents

- **Selection**: The system typically selects the top N documents (e.g., the top 5 or 10) with the highest similarity scores, considering these documents to be the most relevant to the query.
- **Output**: The text content of these top documents is sent to a generation model for the next step in answer generation.

### Step 5: Answer Generation

- **Generation Model Input**: The selected document contents are used as context for a generation model (e.g., GPT).
- **Generating Answer**: The generation model synthesizes these textual inputs to produce a comprehensive and relevant answer.

### Step 6: Output Final Answer

- **User Reception**: The system outputs the answer to the user, for example: "Photosynthesis is the process by which plants, algae, and some bacteria use sunlight to convert water and carbon dioxide into oxygen and glucose."

This example demonstrates how DPR precisely retrieves relevant content from a vast amount of information in a RAG system and assists the generation model in providing accurate and useful answers.

[Tabular Embedding Model (TEM): Finetuning Embedding Models For Tabular RAG Applications](https://arxiv.org/abs/2405.01585)

> No code

## RAFT

[RAFT: Adapting Language Model to Domain-Specific RAG](https://arxiv.org/abs/2403.10131)

The paper introduces a method called **Retrieval Augmented Fine Tuning (RAFT)** designed to enhance pre-trained language models' retrieval-augmented generation (RAG) capabilities, particularly in domain-specific settings under the "open-book" scenario. This approach integrates fine-tuning with RAG to improve model performance on domain-specific question answering tasks.

**Data Preparation**: The RAFT method prepares a set of documents for each question, including "oracle" documents containing the answers and irrelevant distractor documents. For questions with correct documents, RAFT generates Chain-of-Thought (CoT) style answers, directly referencing relevant fragments of the documents to minimize hallucination issues during the generation process. This data structure trains the model to better recognize and utilize relevant information.

**Training Strategy**: During training, the model is fine-tuned to handle scenarios with distractor documents. The model is tasked with accurately extracting useful information and generating answers amidst these distractors. Additionally, some training questions contain only distractor documents to encourage the model to rely on learned domain knowledge for answering. This strategy not only strengthens the model's domain-specific knowledge but also enhances its ability to respond effectively in noisy contexts.

**Integration of Fine-Tuning and RAG**: RAFT improves the model's effectiveness in open-book settings by fine-tuning it to efficiently handle domain-specific documents. The model learns to ignore irrelevant information and accurately cite relevant document content when generating answers. Unlike traditional RAG methods, RAFT focuses on domain-specific applications, further enhancing retrieval and generation capabilities.

Experimental results show that RAFT significantly outperforms other baseline models on several datasets (e.g., PubMed, HotpotQA, Gorilla API Bench), demonstrating its strong potential for domain-specific question answering tasks. The paper provides an effective training strategy for achieving high performance in domain-specific QA tasks and showcases the benefits of combining fine-tuning with RAG to enhance model performance.

## Reference

1. Lu, W., Zhang, J., Zhang, J. and Chen, Y., 2024. Large language model for table processing: A survey. arXiv preprint arXiv:2402.05121.
2. Fang, X., Xu, W., Tan, F.A., Zhang, J., Hu, Z., Qi, Y., Nickleach, S., Socolinsky, D., Sengamedu, S. and Faloutsos, C., 2024. Large Language Models on Tabular Data--A Survey. arXiv preprint arXiv:2402.17944.
3. Zhao, Y., Long, Y., Liu, H., Nan, L., Chen, L., Kamoi, R., Liu, Y., Tang, X., Zhang, R. and Cohan, A., 2023. Docmath-eval: Evaluating numerical reasoning capabilities of llms in understanding long documents with tabular data. arXiv preprint arXiv:2311.09805.
4. Sui, Y., Zou, J., Zhou, M., He, X., Du, L., Han, S. and Zhang, D., 2023. Tap4llm: Table provider on sampling, augmenting, and packing semi-structured data for large language model reasoning. arXiv preprint arXiv:2312.09039.
5. Dong, X., Zhang, C., Ge, Y., Mao, Y., Gao, Y., Lin, J. and Lou, D., 2023. C3: Zero-shot text-to-sql with chatgpt. arXiv preprint arXiv:2307.07306.
6. Sundar, A.S. and Heck, L., 2023. cTBLS: Augmenting large language models with conversational tables. arXiv preprint arXiv:2303.12024.
7. Gao, D., Wang, H., Li, Y., Sun, X., Qian, Y., Ding, B. and Zhou, J., 2023. Text-to-sql empowered by large language models: A benchmark evaluation. arXiv preprint arXiv:2308.15363.
8. Lin, W., Blloshmi, R., Byrne, B., de Gispert, A. and Iglesias, G., 2023, July. LI-RAGE: Late Interaction Retrieval Augmented Generation with Explicit Signals for Open-Domain Table Question Answering. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 1557-1566).
9. Khattab, O. and Zaharia, M., 2020, July. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval (pp. 39-48).
10. Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C. and Zaharia, M., 2021. Colbertv2: Effective and efficient retrieval via lightweight late interaction. arXiv preprint arXiv:2112.01488.
11. Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D. and Yih, W.T., 2020. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.
12. Khanna, S. and Subedi, S., 2024. Tabular Embedding Model (TEM): Finetuning Embedding Models For Tabular RAG Applications. arXiv preprint arXiv:2405.01585.
13. Zhang, T., Patil, S.G., Jain, N., Shen, S., Zaharia, M., Stoica, I. and Gonzalez, J.E., 2024. Raft: Adapting language model to domain specific rag. arXiv preprint arXiv:2403.10131.