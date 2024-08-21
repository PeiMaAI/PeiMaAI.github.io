---
title: "TableRAG(CN)"
layout: post
date: 2024-08-21 10:53
headerImage: false  
category: blog 
author: Pei Ma
---

# Background

In today's information-driven society, the volume of unstructured data is growing at an unprecedented rate, becoming the primary form of information storage and transmission. According to relevant studies, more than 80% of global data exists in an unstructured format, with textual data occupying a significant portion. This textual data is ubiquitous across various industries, from legal documents and academic papers to corporate financial reports and social media content. However, despite the richness and diversity of unstructured text data providing vast opportunities for information retrieval and knowledge discovery, effectively processing and utilizing this data remains an unresolved challenge.

In our previous work, we introduced the **TableRAG** method for processing structured data, achieving significant advancements in this domain. TableRAG optimized the process of retrieving and generating information from structured tabular data through enhanced Retrieval-Augmented Generation (RAG) technology, significantly improving data utilization efficiency. If you are not yet familiar with this method, we recommend reviewing our related research on TableRAG to gain a comprehensive understanding of our technical approach to data processing optimization.

However, as data shifts from structured to unstructured formats, traditional RAG methods face increasing challenges in handling unstructured textual data. Compared to structured data, unstructured text data has more complex inherent semantics, more diverse content structures, and stronger contextual associations. This complexity often makes simple retrieval and generation methods less effective, particularly when faced with complex semantic queries. Traditional RAG models have significant shortcomings in terms of recall rates and generation quality for relevant documents.

Facing these challenges, we realized that constructing a RAG model capable of handling only simple text retrieval and generation is no longer sufficient to meet practical application needs. Instead, we need a more advanced technology that can deeply understand the semantic structure of text, intelligently process contextual information, and enhance the quality of text data retrieval and generation across multiple dimensions.

It is within this context that we developed **TextRAG**. TextRAG is not an independent system but rather a methodology aimed at optimizing the core capabilities of RAG models to perform better in processing unstructured text data. By introducing key technologies such as semantic slicing, automatic context generation, and relevant fragment extraction, TextRAG effectively enhances the model's understanding and generation capabilities for complex texts, significantly improving retrieval precision and generation quality.

In the following sections, we will explore the core technical components of TextRAG in detail and demonstrate how these innovative methods advance the processing of unstructured text data.

# Overall

### Core Framework Introduction

In constructing TextRAG, we focused on addressing several key challenges encountered in processing unstructured text data and developed a set of innovative technologies around these challenges. The overall architecture of TextRAG consists of three main components: Context Segmentation, Context Generation, and Chunks Integration. Below is a flowchart of TextRAG's overall architecture:

![image.png](/insert_images/image.png)

This flowchart illustrates the overall design of TextRAG when processing complex text data. Our architecture includes the following core components:

1. **Context Segmentation**
    
    Context Segmentation is the foundational module of TextRAG, responsible for dividing long texts into multiple sections based on semantics and contextual logic. This segmentation process ensures that each section is coherent and meaningful, allowing subsequent processing steps to operate within more precise semantic ranges.
    
2. **Context Generation**
    
    The Context Generation module generates corresponding document-level and section-level contextual information for each segmented text portion. By adding this contextual information to text blocks, TextRAG can generate more accurate and semantically rich embedded vectors, thereby improving the effectiveness of text retrieval and generation.
    
3. **Rerank**
    
    During text retrieval, we use models from Cohere or Voyage to rerank the retrieved text blocks. The Rerank module sorts the text blocks by relevance, ensuring that the returned results are more closely aligned with the query, thereby enhancing retrieval precision.
    
4. **Chunks Integration**
    
    ![image.png](/insert_images//image%201.png)
    
    After reranking, the Chunks Integration module intelligently combines relevant text blocks, integrating them into more complete paragraphs. The overall architecture of Chunks Integration is shown above. This process ensures that the returned query results are not only highly relevant but also coherent and complete, particularly for handling complex queries.
    

In the following sections, we will explore the working principles of these core components in detail and demonstrate their application in unstructured text data processing. Through these detailed explanations, you will gain a deeper understanding of how TextRAG provides excellent performance and results in complex text processing tasks.

# Part 1: Context Segmentation

In processing unstructured text data, dividing documents into semantically clear and structurally defined parts is the foundation for improving overall processing efficiency and effectiveness. **Context Segmentation** achieves this by dividing long documents into multiple coherent, topic-focused sections, providing a solid semantic foundation for subsequent context generation (Context Generation) and chunks integration (Chunks Integration).

## Detailed Steps of Context Segmentation

1. **Line-Level Splitting and Numbering of Documents**
    
    To ensure precision during the segmentation process, documents are first split by lines, with each line numbered. This approach allows semantic analysis to operate based on specific line numbers, ensuring the accuracy of the segmentation process.
    
    For example, consider the following text fragment:
    
    ```
    [1] Climate change is one of the most pressing issues of our time.
    [2] The Earth’s temperature has risen significantly over the past century.
    [3] Scientists attribute this rise to the increased concentration of greenhouse gases.
    ```
    
    During processing, the text is split into three lines and numbered as [0], [1], and [2], respectively. This numbering method facilitates subsequent segmentation and processing and provides an accurate reference for semantic analysis.
    
2. **Structured Segmentation Based on Large Language Models (LLMs)**
    
    After the document is split and numbered by line, a large language model (LLM) is used to perform structured segmentation of the document. The LLM divides the document into several parts based on the actual semantics and logical structure, with each part focusing on a specific topic or concept.
    
    During this process, the LLM not only analyzes the text's semantics but also generates the start and end positions for each part based on line number information. This structured segmentation process relies on specific prompt design, ensuring that the segmentation results are highly consistent with the document's natural logic. Specifically, the prompt we used is as follows:
    
    ```python
    system_prompt = """
    
    **Task: Structured Document Creation**
    
    You are provided with a document that includes line numbers. Your task is to create a StructuredDocument object by dividing the document into several sections. Each section should focus on a specific topic or concept. Please follow the instructions below carefully:
    
    **Document Division:**
    - Divide the document into several sections based on its content. Each section should discuss a single topic or concept.
    - The sections should align as closely as possible with the document's natural divisions, such as "Introduction," "Background," "Discussion," "Conclusion," etc.
    
    **Line Number Annotation:**
    - For each section, note the starting and ending line numbers.
    - The start_index should be the line number of the first line in that section, and the end_index should be the line number of the last line in that section. For example, if a section begins at line 5 and ends at line 10, the start_index should be 5, and the end_index should be 10.
    
    **Complete Coverage of the Document:**
    - Ensure that all sections together cover the entire document without any gaps or overlaps.
    - The first section should begin at the document's first line, and the last section should end at the document's last line. No lines should be left out, and no sections should overlap.
    
    **Section Titles:**
    - Create a concise and descriptive title for each section, allowing the reader to grasp the main content of the section just by reading the title.
    
    **Judgment and Flexibility:**
    - Note that this document might be an excerpt from a larger document. Therefore, do not assume that the first line is necessarily an "Introduction" or the last line is a "Conclusion."
    - You must make a reasonable division based on the actual content and assign an appropriate title to each section.
    
    The final goal is to produce a StructuredDocument where each section focuses on a specific topic, with accurately marked line number ranges, to clearly and comprehensively reflect the document's content structure.
    
    """
    ```
    
3. **Iterative Processing of Long Documents**
    
    For lengthy documents, it is usually impossible to complete the entire segmentation process in one pass. Therefore, an iterative processing method is employed. By calling the LLM multiple times, processing one part of the document at a time, the entire document is gradually segmented. This approach ensures that even when dealing with ultra-long documents, the system can effectively perform semantic segmentation, ensuring the integrity and coherence of each part.
    
4. **Verification and Correction of Segmentation Results**
    
    After completing all segmentation operations, the segmentation results must be verified to ensure that they seamlessly cover the entire document. Any overlapping, omitted, or discontinuous parts will be adjusted during the correction process to generate the final, structured document segmentation. This step ensures the accuracy and reliability of the segmentation results, laying a solid foundation for subsequent processing.


## Example

---

**Document Content:**

```
[1] Climate change is one of the most pressing issues of our time.
[2] The Earth’s temperature has risen significantly over the past century.
[3] Scientists attribute this rise to the increased concentration of greenhouse gases.
[4] Human activities, such as deforestation and the burning of fossil fuels, are major contributors.
[5] These activities release large amounts of carbon dioxide and other greenhouse gases into the atmosphere.
[6] The effects of climate change are already being felt across the globe.
[7] Rising sea levels, more frequent extreme weather events, and shifting wildlife populations are some examples.
[8] Mitigation efforts are crucial to limit the impact of climate change.
[9] Reducing carbon emissions and transitioning to renewable energy sources are essential strategies.
[10] International cooperation is necessary to tackle this global challenge.
[11] The Paris Agreement is a landmark international accord aimed at limiting global warming.
[12] Nations around the world have committed to reducing their greenhouse gas emissions.
[13] The agreement also includes mechanisms for monitoring progress and holding countries accountable.
[14] However, achieving the targets set by the Paris Agreement requires significant effort and commitment.
[15] Governments, businesses, and individuals all have a role to play in combating climate change.
```

**Structured Document Example:**

```json
{
  "sections": [
    {
      "title": "Overview of Climate Change",
      "start_index": 1,
      "end_index": 3,
      "content_summary": "Introduction to the issue of climate change and its primary causes."
    },
    {
      "title": "Human Contributions to Climate Change",
      "start_index": 4,
      "end_index": 5,
      "content_summary": "Discussion on how human activities contribute to climate change, particularly through greenhouse gas emissions."
    },
    {
      "title": "Effects of Climate Change",
      "start_index": 6,
      "end_index": 7,
      "content_summary": "Description of the ongoing and anticipated effects of climate change globally."
    },
    {
      "title": "Mitigation Strategies",
      "start_index": 8,
      "end_index": 9,
      "content_summary": "Key strategies for mitigating climate change, including reducing emissions and adopting renewable energy."
    },
    {
      "title": "International Cooperation and the Paris Agreement",
      "start_index": 10,
      "end_index": 13,
      "content_summary": "Explanation of the Paris Agreement and the importance of global cooperation in fighting climate change."
    },
    {
      "title": "Call to Action",
      "start_index": 14,
      "end_index": 15,
      "content_summary": "Emphasis on the collective effort required from all sectors of society to achieve climate goals."
    }
  ]
}
```

---

Overall, Context Segmentation plays an irreplaceable role in processing unstructured text data. By reasonably dividing document content into multiple semantically clear parts, Context Segmentation provides the necessary semantic framework for subsequent context generation and paragraph integration and significantly improves the precision and efficiency of text processing. Through this structured segmentation method, the system can more accurately understand the inherent logic of the text, thereby improving information retrieval and generation effectiveness.

# Part 2: Context Generation

In processing unstructured text data, in addition to effectively segmenting the document into semantic parts, it is also essential to generate precise contextual information for each text fragment to support the system's deep semantic understanding and efficient information retrieval. **Context Generation** achieves this by generating titles, summaries, and context blocks for each document section, helping the system better capture the core content and semantic relationships of the text, providing a solid semantic foundation for subsequent embedding and retrieval processes.

## Detailed Steps of Context Generation

1. **Document Title Generation**
    
    Generating an accurate and concise document title is crucial when processing unstructured text. The document title not only quickly conveys the document's theme but also provides a semantic anchor for subsequent context generation and paragraph integration. The system first generates a title based on the document content, a process that relies on the large language model's (LLM) analysis of the text content.
    
    We use the following prompt to generate the document title:
    
    ```python
    DOCUMENT_TITLE_PROMPT = """
    INSTRUCTIONS
    Please provide the title of the document based on the content provided below.
    
    IMPORTANT:
    - Your response must ONLY be the title of the document.
    - Do NOT include any additional text, explanations, or comments.
    
    {document_title_guidance}
    
    {truncation_message}
    
    DOCUMENT CONTENT:
    {document_text}
    """.strip()
    
    ```
    
    When generating the document title, the system first extracts the first few thousand words of the document and passes this content to the LLM. The LLM generates a title that summarizes the main theme of the document based on the provided text content. The generated title must be concise and clear, accurately reflecting the core content of the document.
    
2. **Document Summary Generation**
    
    The document summary is an important part of information retrieval and text generation tasks. By generating a document summary, the system can quickly grasp the core content of the document, thereby enhancing retrieval accuracy. The generation of the document summary also relies on LLM's semantic analysis.
    
    The prompt for generating the document summary is as follows:
    
    ```python
    DOCUMENT_SUMMARIZATION_PROMPT = """
    INSTRUCTIONS
    Please summarize the content of the document in a single, concise sentence.
    
    IMPORTANT:
    - Your summary should begin with: "This document is about: "
    - The sentence should be clear and to the point, covering the main topic of the document.
    - Do NOT add any additional information or context beyond the main topic.
    
    Example Summaries:
    - For a history book titled "A People's History of the United States," you might say: "This document is about: the history of the United States, covering the period from 1776 to the present day."
    - For the 2023 Form 10-K of Apple Inc., you might say: "This document is about: the financial performance and operations of Apple Inc. during the fiscal year 2023."
    
    {document_summarization_guidance}
    
    {truncation_message}
    
    DOCUMENT TITLE: {document_title}
    
    DOCUMENT CONTENT:
    {document_text}
    """.strip()
    
    ```
    
    Through specific prompts, the system guides the LLM to generate a concise document summary, typically in one sentence, directly pointing out the document's main content or theme. The document summary not only provides a global semantic context for the system but also offers necessary support for subsequent context generation.
    
3. **Section Summary Generation**
    
    For long documents divided into multiple parts, generating a summary for each part is especially important. Section summaries help the system understand the text's semantic structure at a finer granularity, thereby improving accuracy in retrieval and generation tasks. (Section titles have already been generated during Part 1 segmentation)
    
    The prompt we use for section summary generation is as follows:
    
    ```python
    SECTION_SUMMARIZATION_PROMPT = """
    INSTRUCTIONS
    Please summarize the content of the following section in a single, concise sentence.
    
    IMPORTANT:
    - Your summary should begin with: "This section is about: "
    - The sentence should clearly describe the main topic or purpose of this section.
    - Do NOT add any additional information or context beyond the main topic of the section.
    
    Example Summary:
    - For a balance sheet section of a financial report on Apple, you might say: "This section is about: the financial position of Apple as of the end of the fiscal year."
    - For a chapter on the Civil War in a history book, you might say: "This section is about: the causes and consequences of the American Civil War."
    
    {section_summarization_guidance}
    
    SECTION TITLE: {section_title}
    
    DOCUMENT TITLE: {document_title}
    
    SECTION CONTENT:
    {section_text}
    """.strip()
    
    ```
    
    Through this prompt, the system can generate a brief summary for each section, clearly indicating the main content and purpose of that section. This fine-grained semantic understanding is particularly important for processing long, complex documents.
    

## Flowchart

The flowchart developed during this process is as follows:

![autocontext.svg](/insert_images/autocontext.svg)

## Example

**Document Content:**
The document is an extensive report on the various effects of climate change on polar bear populations in the Arctic. The report is divided into several sections:

1. **Introduction**: This section provides an overview of the current status of polar bears and the key challenges they face due to climate change. It introduces the main themes of the report, such as habitat loss, changing prey availability, and the impact of human activities.
2. **Melting Sea Ice**: This section discusses the drastic reduction in sea ice in the Arctic, which serves as a critical habitat for polar bears. It explains how the loss of sea ice has affected polar bear hunting behaviors, forced them to migrate longer distances, and led to a decline in their overall population.
3. **Impact on Prey Availability**: This section examines how climate change has disrupted the availability of prey for polar bears, particularly seals. It discusses the cascading effects on the polar bear food chain and how changes in prey distribution are leading to malnutrition and decreased reproductive rates among polar bears.
4. **Human Activities and Pollution**: This section explores how human activities, including oil drilling, shipping, and pollution, are further exacerbating the challenges faced by polar bears. It discusses the potential long-term effects of pollutants on polar bear health and how industrial activities are encroaching on their habitats.
5

. **Conservation Efforts and Future Strategies**: This section reviews current conservation efforts aimed at protecting polar bears and their habitats. It also proposes future strategies that could mitigate the impact of climate change, such as reducing greenhouse gas emissions, protecting critical habitats, and developing new conservation policies.
6. **Conclusion**: The conclusion summarizes the key findings of the report and emphasizes the urgent need for coordinated global efforts to protect polar bears in the face of ongoing climate change.

### Context Generation

```python
# Step 1: Document Title Generation
Generated Title:
"Climate Change and Polar Bear Survival in the Arctic: Challenges and Conservation Strategies"

# Step 2: Document Summarization
Generated Summary:
"This document is about: the various challenges posed by climate change on polar bear survival in the Arctic, including habitat loss, prey availability, and the impact of human activities, along with a review of current and proposed conservation efforts."

# Step 3: Section Title Generation
Section 1 Title:
"Introduction to the Challenges Facing Polar Bears in a Warming Arctic"

Section 2 Title:
"The Role of Melting Sea Ice in the Decline of Polar Bear Populations"

Section 3 Title:
"Disruption of Prey Availability: A Critical Threat to Polar Bears"

Section 4 Title:
"Human Activities and Pollution: Additional Pressures on Polar Bear Survival"

Section 5 Title:
"Current Conservation Efforts and Future Strategies for Polar Bear Protection"

# Step 4: Section Summarization
Section 1 Summary:
"This section is about: the overall impact of climate change on polar bear populations, introducing the main themes of habitat loss, prey disruption, and human impact."

Section 2 Summary:
"This section is about: how the reduction of Arctic sea ice is affecting polar bear habitats, leading to altered hunting patterns and population declines."

Section 3 Summary:
"This section is about: the impact of climate change on the availability of prey for polar bears, resulting in nutritional stress and lower reproductive success."

Section 4 Summary:
"This section is about: the additional challenges posed by human activities and pollution, further threatening polar bear survival."

Section 5 Summary:
"This section is about: the current efforts to conserve polar bears and strategies proposed to address the challenges posed by climate change."

```

Overall, by generating document titles, document summaries, section summaries, as well as context blocks and paragraph headers, the system can understand the semantic structure of the text at multiple levels and provide the necessary contextual information for each text block. This process not only enhances the system's performance in information retrieval and text generation tasks but also lays a solid foundation for semantic processing across the entire system.

In the subsequent sections, we will further explore the technical details of Chunks Integration and analyze how these technologies play a role in complex text processing tasks. Through these analyses, you can gain a more comprehensive understanding of how TextRAG optimizes text processing through context generation, achieving efficient semantic understanding and information integration.

# Part 3: Reranking

In processing unstructured text data, initial search results are typically based on the similarity of embedding vectors, selecting text blocks (chunks) related to the query. However, this vector-based initial ranking may not fully capture the deep semantics of the query, leading to situations where some text blocks appear relevant on the surface but may deviate from the user's query in actual semantics. To further enhance the accuracy and relevance of retrieval results, the **Reranking** step becomes an indispensable part of the process.

### Reranking Details

In the initial stage of text block retrieval, TextRAG screens the most relevant `top-k` chunks from the database based on the similarity of embedding vectors. These initially selected chunks reflect preliminary similarity to the query, but may have limitations when dealing with complex semantics, especially when processing highly semantic text. To address these issues, the Reranking step re-analyzes and reranks the initially selected text blocks using more refined language models (e.g., Cohere or VoyageAI) to better capture the semantic relationship between the text blocks and the query.

During the Reranking process, TextRAG first uses an advanced language model to conduct a more in-depth evaluation of the `top-k` text blocks relative to the query. Each text block is reassigned a new relevance score and reordered based on the new score. These scores are then transformed using a specific transformation function to uniformly distribute them between 0 and 1. This transformation, achieved through a Beta distribution function, ensures the balance and sensitivity of the score distribution, allowing TextRAG to more precisely distinguish subtle differences between text blocks. After reordering, although the text blocks remain the same as the initially selected chunks, their order, scores, and ranking have been adjusted to better reflect their actual semantic relevance to the query.

### Why Transform the Score?

Transforming the relevance score into a uniform distribution between 0 and 1 is a key step in the Reranking process. This transformation, achieved through a Beta distribution function, aims to avoid overly extreme score distributions, thereby enhancing the reliability of the Reranking results. This uniform score distribution ensures that during the subsequent Chunks Integration process, the relevance scores of the various text blocks can be reasonably and effectively considered.

Specifically, during the block integration process, TextRAG needs to combine multiple reranked text blocks into coherent fragment combinations. The uniform distribution of scores ensures that these text blocks can be more precisely ranked and selected based on their actual semantic relevance. In this way, during paragraph integration, TextRAG can more effectively arrange and combine text blocks to form semantically coherent and information-dense paragraphs, further enhancing the overall performance of the system.

Reranking is not only an important step to improve the accuracy of text retrieval but also lays a solid foundation for subsequent block integration. By transforming relevance scores into a distribution between 0 and 1, TextRAG ensures that the scores of each text block are more distinctive, allowing for more precise arrangement and selection of text blocks during block integration. This process greatly enhances the system's ability to handle complex queries, ensuring that the final generated content is both coherent and highly relevant.

In the following Chunks Integration section, we will explore in detail how these reranked text blocks are integrated into more coherent and semantically rich fragment combinations, and how further supplementation of contextual information improves TextRAG's overall performance in text generation and information retrieval.

# Part 4: Chunks Integration

In processing unstructured text data, TextRAG divides documents into chunks, breaking them down into smaller textual units to enable more granular information retrieval and analysis. However, the chunking process itself presents certain challenges: if the chunks are too small, the text's context may be severed, leading to semantic loss; if the chunks are too large, they may contain too much irrelevant information, affecting retrieval accuracy. Particularly when dealing with complex, broad queries, a single chunk is often insufficient to provide a complete answer, which brings up the crucial step of **Chunks Integration**, aimed at ensuring the system returns results that are both complete and accurate by integrating multiple text blocks.

### Why Chunks Integration?

In text retrieval, chunking directly affects the quality of the final results. Both overly small and overly large chunks have their drawbacks:

- **Drawbacks of Overly Small Chunks**:
    1. **Semantic Fragmentation**: When the text is split too finely, the original coherent context is interrupted. For instance, when discussing complex theories or historical events, key information may be dispersed across multiple chunks, making a single chunk unable to provide a complete background and logic.
    2. **Incomplete Information**: Small chunks may only contain part of the relevant information, making it impossible to provide a complete answer. This forces the user to gather multiple chunks to understand the entire content, increasing the complexity of the query results.
    3. **Low Retrieval Efficiency**: Because information is dispersed across multiple small chunks, the system needs to spend more computational resources and time processing and integrating these chunks, reducing retrieval efficiency.
- **Drawbacks of Overly Large Chunks**:
    1. **Increased Noise**: Larger chunks often contain a lot of irrelevant information, which can interfere with the presentation of core content, reducing the accuracy of retrieval results.
    2. **Difficulty in Precise Matching**: When chunks are too large, the system may find it challenging to find content that precisely matches the query, possibly returning chunks with a lot of irrelevant content, making the results less accurate.
    3. **Information Redundancy**: Larger chunks may repeatedly contain content unrelated to the query, making the results appear lengthy and difficult to extract key information from, affecting user experience.

These issues are particularly pronounced when handling complex queries. For example, for a simple query like "What is Shakespeare's birth year?" a single text block can accurately answer because the information is concentrated and clear. However, for more complex and broad questions, a single text block may not provide a comprehensive answer.

Consider a question like "What are the passages in *Hamlet* that describe the tragic fate?" This question requires extracting multiple passages scattered across different scenes of the play. If the chunks are too small, the answer will be fragmented into multiple incoherent parts; if the chunks are too large, the answer may include too much irrelevant plot description, causing the answer to lose focus.

Similarly, consider a historical question like "What were the main factors leading to the outbreak of the Pearl Harbor incident during World War II?" Such a query involves a complex historical background, and the answer may need to draw information from different parts of the literature. For example, the causes may be located at the beginning of the document, while the consequences and summaries may appear at the end. In this case, relying solely on a single text block cannot provide a comprehensive perspective, leading to a partial or inaccurate answer.

For these reasons, Chunks Integration becomes particularly important. For complex queries, returning just one or two chunks may not provide a complete answer and may result in inaccurate or partial answers. By integrating related chunks,

 TextRAG can generate a semantically coherent and information-complete paragraph, ensuring that the answers to complex queries are both comprehensive and precise. This integration process ensures that the system, when faced with complex queries, can synthesize multiple relevant text blocks to provide a detailed and well-organized answer, thus enhancing the quality of information retrieval and user satisfaction.

## Implementation Details

**Chunks Integration** is a critical step in the TextRAG system, optimizing the algorithm to integrate multiple relevant text blocks into semantically coherent and information-complete paragraphs. Below are the detailed implementation steps and an overall architecture diagram:

![image.png](/insert_images/image%202.png)

### 1. Building the Meta-document

In the first step of Chunks Integration, the system extracts the start points, end points, and chunk IDs of each text block from the reranked results, and constructs a meta-document from this information. This meta-document does not contain the actual text content but is an index structure recording the positions of various related text blocks in the document. This index structure provides the basis for subsequent paragraph integration, ensuring that the system can accurately identify and combine related text blocks during processing.

In this way, the system establishes a meta-document containing all the relevant text blocks, clarifying each block's position and relative relationship in the entire document, thereby laying a solid foundation for subsequent paragraph integration.

### 2. Evaluating the Relevance of Text Blocks

After constructing the meta-document, the system needs to evaluate the relevance value of each text block. This step receives multiple input parameters, including all parameters in the meta-document, reranked results, irrelevant chunk penalties, and decay rates. These parameters collectively determine the matching degree of the text block with the query. The following describes this process in detail.

1. **Input Parameters**:
    - **Meta-document Parameters**: Including the start point, end point, and chunk ID of each text block.
    - **Reranked Results**: These results contain the rank and absolute relevance value of each text block.
    - **Irrelevant Chunk Penalty**: Used to control the length of the paragraph generated during the integration process. The larger the penalty, the shorter the generated paragraph.
    - **Decay Rate**: Determines the impact of the rank on the relevance value.
2. **Relevance Score Calculation**:
In calculating the relevance score of a text block, the system uses the `get_chunk_value` function. The specific steps are as follows:
    - First, retrieve the `rank` and `absolute_relevance_value` from `chunk_info`. If these values do not exist, the default `rank` is set to 1000, and the `absolute_relevance_value` is set to 0.0.
    - Then calculate the text block score `v`, with the formula:
    ![formula1](/insert_images/formula1.svg)
    
    - The higher the `rank`:
    ![formula2](/insert_images/formula2.svg)
    the smaller the value, meaning the score will decrease;
    - The larger the `absolute_relevance_value`, the higher the initial score of the text block;
    - Finally, subtract an `irrelevant_chunk_penalty` as a penalty term.
    
    Different penalty coefficients will affect the length of the generated paragraph:
    
    - **0.05**: Generates longer paragraphs, typically containing 20-50 blocks.
    - **0.1**: Generates long paragraphs with 10-20 blocks.
    - **0.2**: Generates medium-length paragraphs with 4-10 blocks.
    - **0.3**: Generates short paragraphs with 2-6 blocks.
    - **0.4**: Generates very short paragraphs with 1-3 blocks.
    
    In this way, the system can dynamically adjust the length and relevance of paragraphs, ensuring that the generated paragraphs contain sufficient information while not being overly lengthy.
    
3. **Length Adjustment**:
After calculating the initial relevance value, the system further adjusts the score of each text block based on its length. This step is implemented through the `adjust_relevance_values_for_chunk_length` function. The specific steps are as follows:
    - **Input Parameters**:
        - `relevance_values`: The initial relevance value list calculated by the `get_chunk_value` function.
        - `chunk_lengths`: The corresponding length list (in characters) of each text block.
        - `reference_length`: A reference length (default 700 characters) used to standardize text blocks of different lengths.
    - **Calculation Process**:
    The system traverses `relevance_values` and `chunk_lengths` and adjusts the score of each text block. The specific formula is:
    ![formula3](/insert_images/formula3.svg)

    - If a text block's length exceeds the `reference_length`, its score will be amplified;
    - If the text block's length is less than the `reference_length`, its score will be reduced.
    
    This adjustment ensures that longer text blocks can achieve higher scores during the integration process, while shorter text blocks receive lower scores, balancing the influence of text blocks of different lengths during the integration process.
    

The purpose of these adjustment processes is to allow the system to more accurately evaluate the value of each text block by considering its length and relevance, thereby selecting the most valuable combination of text blocks during the subsequent paragraph integration process. This method ensures that the final generated paragraph achieves the best results in terms of information density and semantic coherence.

### 3. Selecting the Optimal Fragment Combination: Application of Optimization Algorithms

After evaluating the relevance scores of all chunks, the TextRAG system applies optimization algorithms to generate multiple optimal fragment combinations (Segment Grouping). This process is crucial as it determines whether the final content returned to the user can accurately and comprehensively answer complex queries. Therefore, the system performs precise combination optimization based on multiple constraints. The detailed steps are as follows:

### Input Parameters

The input to the optimization algorithm includes the relevance scores calculated in the previous step and several key constraints. These parameters determine which chunks will be selected and how they will be combined into fragment combinations:

- **Relevance Scores**: These are the scores calculated from the previous steps, reflecting each chunk's matching degree with the query, serving as the core basis for the optimization algorithm to select chunks. Before this step, each chunk has been added with contextual information through the `get_chunk_header` function, including the document title, document summary, section title, and section summary, ensuring that chunks have a complete semantic background during the calculation and selection process.
- **Constraints**:
    - **Maximum Combination Length (max_length)**: Specifies the maximum number of chunks that can be included in a single fragment combination. This constraint prevents the generated fragment combination from being too long, ensuring that the information is focused and coherent.
    - **Overall Maximum Length (overall_max_length)**: Limits the total length of all fragment combinations, preventing the returned content from being overly lengthy.
    - **Minimum Relevance Threshold (minimum_value)**: Ensures that only chunks reaching or exceeding this score are considered for integration, thereby excluding irrelevant or low-relevance content.

### Optimizing the Combination of Chunks

Upon receiving the relevance scores and constraints, the system begins optimizing the combination of chunks, generating one or more fragment combinations (Segment Grouping):

- **Traversing Relevance Scores**: The system traverses all the relevance scores of the query, searching for the highest-scoring chunks and combining them. The combination of these chunks forms a fragment combination that can collectively provide the required information for the query.
- **Checking Constraints**: While generating fragment combinations, the system strictly adheres to the following constraints:
    - **Maximum Chunks Limit**: The number of chunks in each fragment combination must not exceed `max_length`. If a combination exceeds this limit, the system stops adding chunks to that combination and starts creating a new fragment combination.
    - **Document Source Limit**: To maintain the semantic coherence of the fragment combination, all chunks in a combination must come from the same document. If the current chunk comes from a different document than the selected chunks, the system starts creating a new fragment combination.
    - **Overall Length Limit**: The system accumulates the total number of chunks in all generated fragment combinations, ensuring that it does not exceed `overall_max_length`. Once this limit is reached, the system stops creating new fragment combinations, even if there are still relevant chunks that have not been selected.
- **Generating Multiple Fragment Combinations**: The system may generate multiple fragment combinations, each containing chunks that are tightly related and semantically coherent. These combinations will collectively provide a complete answer to the query. During this process, the generated fragment combinations will also be added with contextual information, such as document titles and summaries, using the `get_segment_header` function, to provide complete semantic backgrounds for subsequent processing.

### Parameter Selection and Adjustment

The selection of parameters has a significant impact on the generation of fragment combinations, and different application scenarios require different parameter settings:

- **Maximum Combination Length (max_length)**: If the query requires a broader context, a larger `max_length` can be chosen to include more chunks; for queries requiring precise answers, a smaller `max_length` is more appropriate.
- **Overall Maximum Length (overall_max_length)**: This parameter is usually set according to the expected response length. For short answers, `overall_max_length` should be set smaller; for queries requiring detailed answers, a larger value can be chosen.
- **Minimum Relevance Threshold (minimum_value)**: This is usually set at a higher level to exclude noisy content. However, for queries requiring broader information, this value can be appropriately lowered to include more content.

The selection of these parameters is typically determined through experimentation and adjustment to achieve the best balance. For example, when dealing with complex queries requiring the integration of large amounts of information, `max_length` and `overall_max_length` can be increased, and `minimum

_value` lowered to include more chunks. Conversely, for queries requiring precise and concise answers, the parameters can be adjusted in the opposite direction.

### Segment Grouping

After completing all possible combination calculations, the system returns the highest-scoring one or more fragment combinations (Segment Grouping). The chunks within these combinations satisfy all constraints and maximize the integration of relevant information, ensuring that the content is both coherent and complete.

Through this optimization algorithm, the TextRAG system can generate semantically coherent, information-dense fragment combinations for complex queries. This process resolves the potential issues of information fragmentation and semantic fragmentation that simple chunk retrieval may cause, ensuring that the final generated content can provide users with comprehensive and accurate answers, providing strong support for TextRAG's excellent performance in complex information retrieval tasks.

## Flowchart

The flowchart of this module developed during the process is as follows:

![semantics.svg](/insert_images/semantics.svg)

# Evaluation

This section aims to detail the evaluation experiment process of the **TextRAG** system. We selected the **FINANCEBENCH** dataset as the primary evaluation benchmark and validated **TextRAG**'s performance in handling complex financial information queries through rigorous experimental design.

### FINANCEBENCH

**FINANCEBENCH** is a benchmark dataset specifically designed to evaluate the performance of language models in financial question-answering tasks. The dataset covers 10,231 questions related to publicly listed companies, with content spanning across the following three main categories:

1. **Domain-Related Questions**: These questions primarily involve basic indicators in corporate financial analysis, such as revenue and net profit, testing the model's ability to handle standard financial data.
2. **Newly Generated Questions**: These questions are more complex in design, aiming to test the model's semantic understanding and reasoning abilities, especially when dealing with multi-layered information.
3. **Indicator Generation Questions**: These questions require the model to calculate and reason about financial indicators, testing its comprehensive analytical capabilities.

This benchmark aims to evaluate language models' abilities in information retrieval, semantic understanding, and reasoning through challenging financial question-answering tasks, thereby setting a minimum performance standard for models applied in the financial domain.

### Evaluation Experiment Design for TextRAG

To validate **TextRAG**'s effectiveness in handling complex financial information queries, we designed a series of experiments encompassing the entire process from knowledge base construction to final answer generation.

1. **Knowledge Base Construction**
    - **Document Collection and Preprocessing**: We collected a large number of documents (e.g., 10-K and 10-Q) from the financial reports of publicly listed companies. These documents were preprocessed, split into multiple chunks by line, and embedded into the knowledge base.
    - **Knowledge Base Generation**: Using the Cohere embedding model, we converted the processed document fragments into queryable data in the knowledge base. The knowledge base construction was carried out under a shared vector store configuration, allowing multiple queries to efficiently access the same vector data simultaneously, greatly improving retrieval efficiency and system scalability.
2. **Automatic Query Generation**
    - **Query Generation**: For each question, up to six specific search queries were automatically generated using a large language model (e.g., Claude 3). These queries were designed to be precise enough to locate the most relevant information fragments within the documents.
    - **Query Optimization**: The generated queries were further optimized to ensure their efficiency and relevance, allowing the system to retrieve the most accurate content.
3. **Retrieval and Response Generation**
    - **Contextual Retrieval**: The system executed the generated queries within the knowledge base, extracting text fragments relevant to the questions and integrating these fragments into a complete context. Through multi-layered retrieval and integration, the system could generate the necessary background information for complex questions.
    - **Final Response Generation**: Using the GPT-4-Turbo model, the final answer was generated based on the retrieved context. The generation process focused on the brevity and accuracy of the answer, ensuring that it fully addressed the user's query.
4. **Experimental Results and Analysis**
    - **Result Comparison**: The system-generated answers were compared with the standard answers provided by **FINANCEBENCH**, calculating the model's accuracy and recall. The experimental results showed that under the Shared Vector Store configuration, **TextRAG** achieved an answer accuracy of 79%, significantly higher than the baseline model's accuracy of 19%.
    - **Manual Evaluation**: A manual review was conducted on some complex queries to further verify the system's performance in multi-document retrieval and information integration.
    - **Performance Metrics Evaluation**: Precision, recall, and other performance metrics were calculated to quantify **TextRAG**'s performance in financial question-answering tasks and comprehensively compared with the baseline model.

### Experiment Summary and Future Work

This evaluation experiment demonstrated that **TextRAG** performed excellently in complex financial question-answering tasks under the Shared Vector Store configuration. Its answer accuracy reached 79%, significantly higher than the baseline model's 19%. These results prove **TextRAG**'s advantages in handling complex semantic queries and multi-document integration, especially in its practical application in the financial domain, showing significant value and potential.

In the future, we plan to conduct more precise evaluations of the system's recall rate, precision, and other performance metrics to further optimize **TextRAG**'s performance. Through these follow-up studies, **TextRAG** will be better equipped to handle diverse query needs, providing more comprehensive and reliable answers.

# **(To be continued...)**

## Reference

[A long waiting list …]
