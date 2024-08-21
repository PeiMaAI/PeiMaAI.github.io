---
title: "RAG Augmentation Methods survey"
layout: post
date: 2024-08-17 13:27
headerImage: false  
category: blog 
author: Pei Ma
---

# Table of Contents

1. [Basic Paradigms of RAG](#basic-paradigms-of-rag)
   - 1.1 [Sparse Retrieval Methods](#sparse-retrieval-methods)
   - 1.2 [Dense Retrieval Methods](#dense-retrieval-methods)
   - 1.3 [Retriever Design](#retriever-design)
   - 1.4 [Other Related Methods](#other-related-methods)
2. [Pre-retrieval and Post-retrieval Augmentation](#pre-retrieval-and-post-retrieval-augmentation)
   - 2.1 [Pre-retrieval Augmentation Methods](#pre-retrieval-augmentation-methods)
   - 2.2 [Post-retrieval Augmentation Methods](#post-retrieval-augmentation-methods)
3. [Generation and Generation Augmentation](#generation-and-generation-augmentation)
   - 3.1 [Types of Generators](#types-of-generators)
   - 3.2 [Generator Augmentation Methods](#generator-augmentation-methods)
   - 3.3 [Integrated Enhancement of Retrieval and Generation](#integrated-enhancement-of-retrieval-and-generation)
4. [Optimization of Retrieval and Generation Processes](#optimization-of-retrieval-and-generation-processes)
   - 4.1 [Necessity and Frequency of Retrieval](#necessity-and-frequency-of-retrieval)
   - 4.2 [Training-Free and Independent Training](#training-free-and-independent-training)
   - 4.3 [Sequential Training and Joint Training](#sequential-training-and-joint-training)
5. [Case Studies of RAG in Applications](#case-studies-of-rag-in-applications)
   - 5.1 [NLP Applications](#nlp-applications)
   - 5.2 [Downstream Task Applications](#downstream-task-applications)
   - 5.3 [Domain-Specific Applications](#domain-specific-applications)
6. [Discussion and Future Research Directions](#discussion-and-future-research-directions)
   - 6.1 [Limitations of RAG](#limitations-of-rag)
   - 6.2 [Potential Future Research Directions](#potential-future-research-directions)

---

## Introduction

In recent years, models based on Retrieval-Augmented Generation (RAG) technology have made significant advancements in the field of Natural Language Processing (NLP). RAG technology combines the strengths of information retrieval and text generation to effectively address the "hallucination" problem commonly seen in traditional generative models, where the generated content may not align with factual information. Additionally, RAG technology does not rely on extensive fine-tuning or pre-training processes, which not only reduces computational resource requirements but also significantly enhances the model’s flexibility and adaptability. More importantly, RAG technology can directly retrieve the required information from external corpora, without the need to pre-construct and maintain a large knowledge base. This "plug-and-play" feature makes RAG systems perform exceptionally well in many practical applications. However, while the implementation of RAG technology is not complex, building a high-performing and stable RAG system is challenging. To gain a deeper understanding and optimization of RAG technology, this paper systematically summarizes the design and enhancement methods of key components in current RAG systems and proposes several potential optimization strategies to provide references for future research and applications.

---

## 1. Basic Paradigms of RAG

The design of RAG systems encompasses various paradigms, achieving efficient synergy between information retrieval and text generation by organically combining sparse and dense retrieval methods. Optimizing retriever design and introducing pre-retrieval and post-retrieval augmentation techniques can significantly enhance the performance of the generator.

### 1.1 Sparse Retrieval Methods

- **TF-IDF**: Term Frequency-Inverse Document Frequency (TF-IDF) is a traditional text retrieval method based on term frequency and inverted indexing. It locates the most relevant documents by measuring the importance of terms in a document. In RAG systems, TF-IDF is often used for paragraph-level retrieval to enhance the quality of the generator's input. However, since this method relies on surface-level term frequency, it may not fully capture the semantic information of terms, potentially limiting its performance when dealing with complex queries.

- **BM25**: BM25 is another retrieval method based on term frequency and inverted indexing. It represents documents as a bag of words and ranks them based on term frequency and inverse document frequency. BM25 is particularly effective in large-scale text database queries and is one of the most widely used paragraph retrieval methods in RAG models. Compared to TF-IDF, BM25 introduces parameters to adjust the frequency effect of terms, thereby improving retrieval robustness to some extent.

### 1.2 Dense Retrieval Methods

- **DPR (Dense Passage Retriever)**: DPR is a dense retriever based on the BERT architecture, designed and pre-trained specifically for open-domain question-answering tasks. DPR embeds queries and documents into the same vector space and retrieves them based on the semantic similarity of embedding vectors, demonstrating strong pre-training capabilities. This method serves as a critical component in many RAG models, significantly enhancing their performance in handling complex semantic queries.

- **Contriever**: Contriever is a dense retriever based on contrastive learning, using a single encoder architecture pre-trained on large-scale unaligned documents. It has shown excellent performance in open-domain question-answering tasks. Contriever is particularly suited for integration with large models like InstructGPT, performing exceptionally well in diversified tasks.

- **Spider**: Similar to Contriever, Spider is a general-purpose pre-trained dense retriever, pre-trained through contrastive learning to adapt to various tasks and domains. The flexibility and efficiency of Spider make it widely used in many RAG methods, especially when dealing with large-scale corpora.

### 1.3 Retriever Design

- **Bi-Encoder**: The Bi-Encoder consists of two independent BERT-based encoders, one for processing queries and the other for documents. This design is often used for sentence embedding similarity retrieval and diverse example retrieval, enabling the extraction and application of general knowledge through parameter freezing or partial freezing.

- **One-Encoder**: The One-Encoder structure processes queries and documents using a single encoder, typically based on Transformer, BERT, or other sequence modeling architectures. Through pre-training on large-scale unaligned documents using contrastive learning, the One-Encoder demonstrates excellent adaptability and generalization ability, flexibly meeting the needs of various tasks.

### 1.4 Other Related Methods

- **Contrastive Learning**: Contrastive learning is a commonly used method for pre-training single-encoder structures. It learns embeddings of positive and negative sample pairs, enabling the model to effectively train on unaligned documents. This method excels in enhancing model adaptability and generalization, particularly when handling diverse and complex corpora.

- **Large-scale Specialized Pre-training**: Specialized pre-training on specific tasks (such as open-domain question-answering) can significantly enhance a model's performance in knowledge-intensive tasks. By pre-training on open-domain question-answering tasks, DPR greatly improves the accuracy of models in addressing domain-specific issues.

### Summary of Methods

Retrieval methods in RAG systems mainly fall into sparse retrieval and dense retrieval categories. Sparse retrieval methods, such as TF-IDF and BM25, rely on term frequency and inverted indexing and are suitable for general text retrieval tasks. However, their performance is limited by the quality of the database and the query. In contrast, dense retrieval methods, such as DPR, Contriever, and Spider, embed queries and documents into vector spaces and retrieve them based on semantic similarity, demonstrating greater flexibility and adaptability. Additionally, through retriever designs such as Bi-Encoder and One-Encoder, combined with contrastive learning and large-scale specialized pre-training, the model's performance in various tasks is further enhanced.

---

## 2. Pre-retrieval and Post-retrieval Augmentation

To further improve retrieval quality and optimize the input and output of the generator, pre-retrieval and post-retrieval augmentation strategies have been widely applied in RAG systems. These augmentation strategies not only improve the quality of retrieval results but also significantly enhance the generator's output.

### 2.1 Pre-retrieval Augmentation Methods

**Query2doc**:
- Wang et al. proposed a query expansion method that generates pseudo-documents using a few prompts from a large language model (LLM) to expand relevant information in the original query. This method effectively improves the query's disambiguation ability and provides a clearer retrieval target for the retriever. Experiments on temporary information retrieval datasets have shown that Query2doc significantly enhances the performance of both sparse and dense retrievers.

**Hypothetical Document Embedding (HyDE)**:
- Gao et al. introduced the HyDE method, which guides LLMs to generate hypothetical documents and uses these documents as new query embeddings to search for relevant neighbors, thereby improving the accuracy and relevance of retrieval. The HyDE method performs exceptionally well when handling complex queries, especially in cases where context is lacking, significantly enhancing retrieval effectiveness.

**Query Rewrite**:
- Ma et al. proposed the Rewrite-Retrieve-Read framework, where LLMs are prompted to generate retrieval queries and rewrite the original question to better match the retrieval needs. This method not only enhances the retriever's understanding of the input but also significantly improves the

 relevance and consistency of the generated output.

**Query Augmentation**:
- Yu et al. proposed the Query Augmentation method, which combines the original query with the initial generated output to form a new query for further retrieval of relevant information. This method excels at clarifying the relationship between the query and the generated content, helping to extract more relevant information from the corpus and improving the completeness and accuracy of the generated content.

### 2.2 Post-retrieval Augmentation Methods

**Pluggable Reward-driven Context Adapter (PRCA)**:
- Yang et al. proposed a post-retrieval augmentation method based on reinforcement learning. It adjusts retrieved documents and fine-tunes lightweight adapters to better align them with the generator, thereby improving the quality and consistency of the generated content. The PRCA method effectively reduces irrelevant information during the generation process and enhances the fluency of the output.

**Retrieve-Rerank-Generate (R2G)**:
- Glass et al. introduced the R2G method, which re-ranks documents obtained from different retrieval methods to improve the robustness of retrieval results and optimize the final generated answers. This method performs exceptionally well in multimodal retrieval tasks, significantly enhancing the accuracy and relevance of the generated results.

**BlendFilter**:
- Wang et al. proposed the BlendFilter method, which combines pre-retrieval query generation blending with post-retrieval knowledge filtering to better handle complex questions and noisy retrieved knowledge, thereby enhancing the accuracy of the generated content. BlendFilter is particularly effective when dealing with long-tail queries, significantly reducing errors in the generation process.

**Retrieve, Compress, Prepend (RECOMP)**:
- Xu et al. proposed the RECOMP method, which compresses the retrieved documents before context enhancement in the generation process, reducing redundant information and improving the relevance and conciseness of the generated content.

**Lightweight Version of the FiD (Fusion-in-Decoder) Model**:
- Hofstätter et al. introduced a lightweight version of the FiD model, which compresses the encoding vectors of each retrieved paragraph before concatenation and decoder processing and re-ranks the retrieval results to optimize the generator’s performance. This method effectively reduces computational costs and improves the efficiency of the generation process.

### Summary of Methods

Pre-retrieval and post-retrieval augmentation methods play a crucial role in improving the overall performance of RAG systems. Pre-retrieval augmentation methods, such as Query2doc, HyDE, Query Rewrite, and Query Augmentation, improve retriever performance in handling complex queries by enhancing query expression. Post-retrieval augmentation methods, such as PRCA, R2G, BlendFilter, RECOMP, and the lightweight version of FiD, further enhance the quality of the generator's output by optimizing the ranking and processing of retrieval results. These methods complement each other, jointly contributing to the improvement of RAG system generation effects.

---

## 3. Generation and Generation Augmentation

Generators play a crucial role in RAG systems. By optimizing the design and augmentation of generators, the accuracy and relevance of generated text can be significantly improved.

### 3.1 Types of Generators

Generators can be broadly categorized into parameter-accessible (white-box) and parameter-inaccessible (black-box) types.

**Parameter-Accessible Generators (White-Box)**:

- **Encoder-Decoder Architecture**: This architecture independently processes the input and the target and uses cross-attention components to link the input with the target tokens. Representative models include T5 and BART. BART is typically used as a generator in RAG systems, while FiD uses T5 as a generator to achieve higher generation quality.

- **Decoder-Only Architecture**: The decoder-only architecture connects the input and the target, allowing both representations to be constructed in parallel, enabling the model to flexibly adjust generation strategies during the generation process. This architecture achieves the generation process through a single decoder, offering high generation flexibility.

**Parameter-Inaccessible Generators (Black-Box)**:

- Representative models include the GPT series, Codex, and Claude. These generators only allow query input and response reception without access to their internal structures or parameters. Despite this, black-box generators still perform excellently in diverse tasks due to their extensive pre-training and strong generative capabilities.

- **Prompt Retriever**: Rubin et al. proposed a method for training a prompt retriever that uses data generated by language models to provide better examples for the generator for in-context learning, further enhancing generation quality.

- **Document Compression**: Xu et al. proposed a method that compresses retrieved documents before context integration to reduce computational costs and alleviate the burden on the generator when handling large-scale contexts.

### 3.2 Generator Augmentation Methods

The performance of the generator directly affects the final output quality of the RAG system. Through prompt engineering, decoding optimization, and generator fine-tuning, the performance of the generator can be further enhanced.

**Prompt Engineering**:

- **LLMLingua**: This method compresses query length through a small model, speeding up model inference, reducing the negative impact of irrelevant information on the model, and mitigating the "mid-generation loss" phenomenon.

- **ReMoDiffuse**: ChatGPT is used to break down complex descriptions into clear textual scripts, enhancing the accuracy and consistency of generated text.

- **ASAP**: Example tuples (including input code, function definitions, analysis results, and corresponding comments) are integrated into prompts to improve generation quality.

**Decoding Optimization**:

- **InferFix**: This method balances the diversity and quality of generated content by adjusting the decoder's temperature, ensuring that the generated content is both accurate and diverse.

- **SYNCHROMESH**: By restricting the decoder's output vocabulary, this method eliminates potential implementation errors, enhancing the generator's reliability and stability.

**Generator Fine-Tuning**:

- **RETRO**: The retriever's parameters are fixed, and a block cross-attention mechanism is used to combine queries and retrieved content to improve generation effects.

- **APICoder**: This method improves the accuracy and consistency of code generation tasks by fine-tuning the generator with a mix of API information and code blocks.

- **CARE**: By fine-tuning the decoder, this method reduces subtitle and concept detection losses while keeping the encoder and retriever parameters fixed, optimizing the performance of multimodal generation tasks.

### 3.3 Integrated Enhancement of Retrieval and Generation

The integration of retrieval and generation is a key component of RAG systems. By enhancing design at the input, output, and intermediate layers, the overall performance of the model can be significantly improved.

**Input Layer Integration**:

- Input layer integration methods combine retrieved documents with the original input or query and pass them to the generator. This method is widely used in models such as In-Context RALM, FiD, Atlas, and REPLUG, where concatenating input and retrieved documents enables the generator to better handle complex tasks.

**Output Layer Integration**:

- Output layer integration methods improve the quality of generated content by combining retrieval and generation results. In kNN-LM, two distributions of the next token are interpolated during prediction, one guided by the language model and the other by the nearest neighbors in the retrieval corpus, allowing for a more flexible generation process.

**Intermediate Layer Integration**:

- Intermediate layer integration methods introduce semi-parametric modules that incorporate retrieved information into the internal layers of the generation model, interacting with the intermediate representations during generation. Although this integration method increases the model's complexity, effective training can significantly enhance the generative model's capabilities. For example, RETRO processes retrieval blocks within the generator's blocks through a block cross-attention layer, while EAE and TOME integrate retrieved entities and mentions through entity memory and memory attention layers.

### Summary of Methods

The design and augmentation of generators are crucial in RAG systems. By selecting appropriate generator types, optimizing prompt engineering, tuning the decoding process, and fine-tuning generators, the performance of generators can be effectively improved. Furthermore, through integrated enhancement design at the input, output, and intermediate layers, the synergy between retrieval and generation is further optimized, improving the overall generation quality of the RAG system.

---

## 4. Optimization of Retrieval and Generation Processes

The retrieval and generation processes in RAG systems can be further optimized through design improvements to enhance model efficiency and accuracy. The following sections introduce methods for optimizing retrieval necessity and frequency, training-free and independent training, as well as sequential training and joint training.

### 4.1 Necessity and Frequency of Retrieval

In LLM-based generation processes, retrieval operations are often used to supplement knowledge, enhancing the accuracy of the generated content. However, retrieval is not always necessary, and excessive retrieval may introduce irrelevant information, leading to generation errors. Therefore, determining the necessity and frequency of retrieval is crucial for achieving a robust RAG model.

**Self-RAG**:
- Self-RAG introduces special markers to assess the necessity of retrieval and controls retrieval behavior as needed, thereby reducing unnecessary resource consumption. This method dynamically adjusts retrieval frequency during the generation process, effectively optimizing model efficiency.

**SKR (Self-Knowledge Guided Retrieval)**:
- SKR leverages LLMs' self-assessment abilities to decide whether to invoke the retriever, dynamically adjusting retrieval behavior to reduce unnecessary queries and computations.

**FLARE**:
- FLARE actively decides whether and when to search during the generation process based on probabilities, avoiding excessive retrieval and optimizing the use of computational resources.

**Design of Retrieval Frequency**:
- In the generation process, retrieval frequency determines the degree of reliance on retrieval results, impacting model efficiency and effectiveness. Common settings include one-time retrieval, retrieval every n tokens, and retrieval for each token. Different retrieval frequency strategies balance performance and computational cost, applicable to models like REALM, In-Context RALM, and RETRO.

### 4.2 Training-Free and Independent Training

**Training-Free Methods**:
- With the development of LLMs, many studies suggest enhancing LLMs through retrieval mechanisms without

 fine-tuning model parameters. Methods based on prompt engineering directly integrate retrieved knowledge into the original prompt, improving generation quality. For example, In-Context RAG enhances generation effects by combining retrieved documents with the original prompt.

**Independent Training Methods**:
- In independent training, the retriever and generator are trained as two completely independent modules, without interaction during the training process. DPR uses BERT for contrastive learning training, while CoG improves the accuracy of text generation through the training of prefix and phrase encoders.

### 4.3 Sequential Training and Joint Training

**Sequential Training**:
- Sequential training methods pre-train the retriever or generator independently, then fix the pre-trained module before training the other module. RETRO adopts a BERT model as a pre-trained retriever and integrates retrieval blocks into the model's predictions using an encoder-decoder structure.

**Joint Training**:
- Joint training methods adopt an end-to-end training paradigm, simultaneously optimizing the retriever and generator. RAG jointly trains the retriever and generator by minimizing the negative log-likelihood loss, enhancing the overall system performance. REALM uses a similar training paradigm and employs maximum inner product search techniques to locate the most relevant documents.

### Summary of Methods

By controlling the necessity and frequency of retrieval, selecting training-free and independent training strategies, and optimizing sequential training and joint training processes, the overall performance of RAG systems can be significantly improved. These process optimization methods not only enhance model generation effects but also achieve a good balance in computational efficiency and resource utilization.

---

## 5. Case Studies of RAG in Applications

RAG technology demonstrates strong adaptability and efficiency across a wide range of applications in NLP, downstream tasks, and specific domains. The following sections introduce specific application examples of RAG in these areas.

### 5.1 NLP Applications

**Question-Answering Systems**:
- Question-answering systems can significantly improve answer accuracy and contextual relevance by integrating RAG technology. REALM integrates knowledge retrievers during pre-training, fine-tuning, and inference processes, retrieving information from large corpora and significantly improving system response quality. Fusion-in-Decoder achieves higher accuracy by retrieving paragraphs from supporting documents and merging them with questions to generate answers.

**Chatbots**:
- Chatbots need to maintain natural conversations with users. By integrating RAG technology, chatbots can retrieve relevant information from static databases or the internet, enhancing the richness and coherence of conversations. For example, BlenderBot3 combines internet information and local conversation history, significantly improving the quality of dialogue generation.

**Fact-Checking**:
- Fact-checking tasks aim to verify the accuracy of information. RAG technology enhances fact-checking capabilities by retrieving external knowledge. Atlas demonstrates significant progress by verifying the performance of RAG technology in fact-checking under few-shot learning.

### 5.2 Downstream Task Applications

**Recommendation Systems**:
- RAG technology shows great potential in recommendation systems by integrating retrieval and generation processes to provide personalized and context-relevant recommendations. For example, the retrieval-augmented recommendation model proposed by Di Palma utilizes knowledge from movie or book datasets to make recommendations, significantly improving the system's recommendation accuracy.

**Software Engineering**:
- RAG technology is applied in software engineering for tasks such as code generation and program repair. For example, APICoder improves the accuracy and efficiency of code generation by retrieving relevant code snippets from code repositories and aggregating them with inputs. Additionally, RAG technology demonstrates great potential in table data processing and Text-to-SQL semantic parsing.

### 5.3 Domain-Specific Applications

**Healthcare**:
- In the healthcare field, RAG technology is applied in drug discovery and molecular analysis. By integrating multiple data modalities (such as images, text, molecular structures, etc.), RAG technology can provide more comprehensive analysis and diagnostic support.

**Legal**:
- RAG technology is applied in the legal field mainly in the retrieval of legal documents and the generation of case summaries. By combining legal knowledge bases and case databases, RAG systems can provide accurate legal advice and references for legal practitioners.

### Summary of Methods

The application of RAG technology in NLP, downstream tasks, and specific domains demonstrates its broad adaptability and strong processing capabilities. By seamlessly integrating retrieval and generation processes, RAG systems can provide efficient and accurate support across various tasks.

---

## 6. Discussion and Future Research Directions

### 6.1 Limitations of RAG

Despite the outstanding performance of RAG technology in many fields, it also has certain limitations:

**Noise in Retrieval Results**:
- The information retrieval process may introduce noise, which in some cases may interfere with the generation process. However, some studies suggest that in certain circumstances, noisy retrieval results may actually help improve generation quality.

**Additional Overheads**:
- The retrieval process in RAG systems incurs additional computational overhead, especially when frequent retrieval of large-scale data is required. This overhead can affect the system's real-time performance and efficiency in resource utilization.

**Discrepancy Between Retriever and Generator**:
- Since the objectives of the retriever and generator are not entirely aligned, optimizing their interaction requires careful consideration. While some methods reduce this discrepancy through joint training, this increases system complexity.

**Increased System Complexity**:
- Introducing retrieval functionality in RAG systems increases overall system complexity, particularly when adjustments and optimizations are needed, requiring higher technical knowledge and experience.

**Increase in Context Length**:
- Query-based RAG systems significantly increase context length, which may negatively impact the generator’s performance and slow down the generation process.

### 6.2 Potential Future Research Directions

To further enhance the application value of RAG technology, future research can explore the following directions:

**Design of New Augmentation Methods**:
- Existing research has explored various interaction modes between retrievers and generators. Future research can explore more advanced augmentation methods to fully unlock the potential of RAG systems.

**Flexible RAG Processes**:
- Future RAG systems can achieve more refined task processing and performance improvements through recursive, adaptive, and iterative RAG processes.

**Broader Applications**:
- Although RAG technology has been applied in multiple domains, future efforts can focus on expanding its application to more scenarios, particularly in underexplored areas.

**Efficient Deployment and Processing**:
- Future research should focus on developing more plug-and-play RAG solutions to optimize system-level deployment efficiency and processing performance.

**Combining Long-Tail and Real-Time Knowledge**:
- To improve personalized information services, RAG systems can be designed with continuously updating knowledge bases and adapting to real-time information.

**Integration with Other Technologies**:
- Future research can explore the combination of RAG technology with other methods that enhance AI-generated content (AIGC) effectiveness, such as fine-tuning, reinforcement learning, and chain-of-thought approaches, to further improve generation outcomes.

---

This paper provides a comprehensive technical reference for researchers and developers by analyzing the basic paradigms, augmentation methods, generation and generation augmentation, process optimization, and application examples of RAG systems. It also identifies potential future research directions. The further optimization and development of RAG technology will bring broader application prospects and technological breakthroughs to natural language processing and related fields.

# (To be continued...)

## Acknowledgments

I would like to express my deep gratitude to the authors of several key surveys that have been instrumental in the development of this work. Their comprehensive analyses and insights into Retrieval-Augmented Generation (RAG) technology have provided a robust foundation for my understanding and exploration of this rapidly evolving field.

In particular, the surveys on "[Retrieval-Augmented Text Generation for Large Language Models" (arXiv:2404.10981)]((https://arxiv.org/abs/2404.10981)), "[Retrieval-Augmented Generation for AI-Generated Content" (arXiv:2402.19473)]((https://arxiv.org/abs/2402.19473)), and "[RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models" (arXiv:2405.06211)](https://arxiv.org/abs/2405.06211) have been invaluable resources. These works have significantly shaped my understanding of the current state of RAG technologies, their challenges, and their potential applications. I am deeply indebted to the authors for their contributions, which have greatly informed and guided the research presented in this paper.

## Reference
