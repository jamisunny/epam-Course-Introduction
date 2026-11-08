# Generative AI Applications in Business

Generative Artificial Intelligence (GenAI) is radically transforming familiar workflows, science, and everyday life. It has elevated AI from a mere automation tool into a system that can reason, adapt, and respond to customers in natural language. A function does not exist in a vacuum, and GenAI acts as the connective tissue between disparate data, products, and user experiences. However, while these technologies (large language models, diffusion neural networks, and code generators) successfully bridge digital ecosystems and automate routine tasks, they also pose unprecedented challenges for global society.

This report first explains how the technology works, then analyses three enterprise use cases. For each, it covers business value, technical challenges, weaknesses, and existing implementations.


## 1. How the Technology Works

### 1.1 Core concepts

Generative AI starts with ML (machine learning) and NLP (natural language processing). Today's systems are built on the **Transformer** model, a neural architecture that weighs how every part of a text relates to every other part. Scaled up, it becomes an **LLM** (large language model). This architecture works in two regimes: **Zero-Shot** and **Few-Shot**. Zero-shot Learning (ZSL) is a model's ability to perform tasks without any training examples. It achieves this through generalized knowledge acquired during pre-training. Few-shot Learning (FSL) is a method in which a model is given only a few examples (typically 1 to 5) to better understand the structure of a task.


* **Tokens.** An LLM doesn't read words directly. It splits text into tokens, which also set the price and speed of every call. A token is a basic chunk of text. Depending on the language and word complexity, one token equals roughly 4 characters or about ¾ of an English word (100 tokens ≈ 75 words). How it works: Common short words like "bank" or "risk" are usually single tokens. Longer or complex words like "compliance" might be split into two tokens (compli + ance). Spaces, punctuation marks, and numbers also count as tokens. AI cloud providers bill businesses strictly per million input and output tokens. Generating responses takes time proportional to the number of tokens produced.
* **Embeddings.** It turns meaning into numbers called embeddings, so similar ideas sit close together. Computers cannot understand the dictionary definition of words; they only process numbers. Embeddings bridge the gap between human language and computer math. An embedding is a long list of numbers (a mathematical vector, often containing 768 to 3,072 numbers) that represents the semantic meaning of a piece of text. How it works: Specialized AI models read a sentence and plot it as a point in a massive, multi-dimensional map of concepts. Sentences with similar meanings are placed physically close together on this map, even if they use completely different words. Example: The phrases "interest rate hike" and "central bank increases borrowing costs" share zero exact words, yet their vector embeddings will sit right next to each other because they mean the same thing. This approach allow software to search databases by intent and meaning (semantic search) rather than relying on rigid, old-fashioned keyword matching.
* **Context window.** It can only consider a limited span of text at once that mean the maximum number of tokens an LLM can hold and process at one single moment (measured in thousands or millions of tokens, e.g., 128k or 1M tokens). How it works: Everything you send to the model—your system rules, the chat history, pasted PDF pages, and your final question—plus the answer the model generates must fit inside this window. Once the conversation exceeds the limit, the oldest tokens fall out of the window and the model completely forgets them.
* **RAG (Retrieval-Augmented Generation).** Instead of relying only on what the model learned in training, the system first searches a document database for relevant snippets and passes only those to the model. By default, an LLM is like a student taking a closed-book exam: it only knows what it memorized during training months or years ago, and if it forgets a fact, it might guess (hallucinate). RAG turns that into an open-book exam.

1. **Preparation:** Your company breaks thousands of legal pages into small paragraphs, converts them into Embeddings, and stores them in a Vector Database.

2. **Retrieval:** When an employee asks a question, the system turns the question into an Embedding, searches the database for the closest matching paragraphs, and pulls out the top 3–5   relevant snippets.

3. **Augmentation:** The system pastes those exact snippets alongside the user's question into the LLM's Context Window (measured in Tokens).

4. **Generation:** The LLM reads the provided snippets and writes an accurate answer backed by direct citations.


### 1.2 Model catalogs

Previously, to use AI, companies had to build a direct relationship with a single developer (for example, going directly to OpenAI for ChatGPT). Today, hyperscale cloud providers have built the equivalent of an AI "supermarket" inside their existing cloud infrastructure, called a **Model Catalog**. The cloud provider handles the engineering, hosting, hardware, and security, and enterprises simply select and run the model they need from a curated list. Two platforms dominate the market:

* **Amazon Bedrock:** Amazon Web Services' (AWS) managed AI catalog. It gives access to foundation models such as Claude (Anthropic), Llama (Meta), and Mistral (Mistral AI).
* **Vertex AI Model Garden:** Google Cloud's AI repository. Its cornerstone is Google's Gemini family of models, and it also includes partner systems and leading open-source models.

**Why it matters:** companies aren't locked into one vendor. If a bank builds its whole pipeline around a single proprietary model, it is exposed to downtime, sudden price changes, or performance degradation. With a model catalog, the model behind a RAG pipeline or chatbot can be swapped from Claude to Llama or Gemini with minimal code changes, which preserves agility and data independence.


### 1.3 Common technical limitations (apply to all use cases)

* **Hallucinations.** The Transformer predicts the next most likely token instead of retrieving facts, so it can produce convincing but false output. RAG reduces hallucinations but doesn't eliminate them; failures shift to the retrieval layer (poor chunking, stale document versions, weak ranking). Common practice is 256–512 tokens per chunk with 10–20% overlap: chunks that are too small lose context, chunks that are too large dilute relevance.
* **Context window limits**. Context Window Limits: Every model processes input based on a fixed Context Window (measured in tokens—chunks of text/code). If enterprise documents exceed this capacity, historical context or crucial details risk being truncated or subject to Lossy Compression during multi-step reasoning. The context window is a limit on attention as well as capacity. Research shows LLMs pay less attention to information in the middle of their context window. This refers to a well-known phenomenon in the AI industry called the "Lost in the Middle" effect. If you upload a massive document or book into an LLM and ask about a specific fact located right in the middle of the text, the model is highly likely to overlook or ignore it. Models excel at recalling information from the very beginning (prompts/instructions) and the very end (latest lines of text), but they tend to become "blind" to the middle sections.
* **Larger windows don't replace RAG**, for three reasons:

  1. **Prefill latency.** The model must process every input token before writing the first word, so passing a 500-page book causes long delays. RAG passes only 2–3 relevant pages.
  2. **Frozen knowledge.** A model's knowledge stops at the end of training. RAG injects up-to-date documents without retraining.
  3. **Citations.** Regulated industries need every claim to be auditable, and RAG can return the exact source document and passage.
* **Prompt fragility.** Changing one word, reordering instructions, or adding a trailing space can change the output. A prompt that passed testing can fail in production, for example by returning invalid JSON.
* **Cost of fine-tuning and RLHF.** RLHF (Reinforcement Learning from Human Feedback) needs thousands of hours of expert ratings and runs several models at once (base, reward, reference, and policy models), which uses a lot of GPU time. Models can also learn "reward hacking": answers that score well but are hollow or sycophantic.


## 2. Use Case: Financial Services – Knowledge Search, Compliance, and Credit Decisions

### Business Value

* Compliance officers and risk analysts can query thousands of pages of policies and regulations in natural language and get answers with citations instead of searching manually.
* Advisors get instant access to the firm's research and procedures, which shortens client response times.
* In lending, AI models help institutions make faster and more consistent credit and fraud decisions.


### Technical Challenges

* **Retrieval quality:** answers are only as good as the chunking, embeddings, and ranking of the document store. A stale version of a regulation produces a confident but wrong answer.
* **Citation accuracy:** every answer must point to the exact source document, page, and paragraph to pass an audit.
* **Data security and residency:** client and internal data can't leave approved environments, so models from a catalog must be vetted for compliance and data residency.
* **Guardrails:** layers such as NeMo Guardrails or Llama Guard must filter queries and outputs so the model never leaks confidential data.


### Weaknesses

* Hallucinations carry direct legal and financial risk; a human expert must still verify the final answer.
* Credit decisions must be explainable to regulators and customers, which is hard for complex models.
* The model's knowledge is frozen, so the system depends on the document database being constantly updated.


### Existing Implementations

* [**AI @ Morgan Stanley Assistant**](https://openai.com/customer-stories/morgan-stanley)
An internal assistant for Morgan Stanley's wealth management advisors, built on OpenAI's GPT-4. It gives advisors instant answers drawn from the firm's knowledge base of over 100,000 research reports and documents, and it was tested with evaluations against the firm's own experts before rollout.
* [**Zest AI**](https://www.zest.ai/)
An AI-powered credit risk platform that helps financial institutions make better lending decisions. Its suite covers credit underwriting, fraud detection, and agentic AI for lenders.


## 3. Use Case: Software Engineering – AI Coding Assistants

### Business Value

* Developers write boilerplate code, documentation, and unit tests much faster, freeing time for design and review.
* New team members get up to speed faster because the assistant explains unfamiliar code.
* Shorter development cycles and instant prototyping help companies reach the market sooner.


### Technical Challenges

* **Context window limits:** large codebases don't fit in the window, so the assistant may not "see" related files and can suggest code that conflicts with the rest of the project.
* **Prompt fragility:** small changes in comments or instructions can produce very different code.
* **Integration:** the tool must plug into IDEs, repositories, and security policies without sending private code outside the company.


### Weaknesses

* Generated code can contain bugs or security vulnerabilities that look correct, so every suggestion still needs human review.
* Models trained on public repositories raise copyright and licensing questions about generated code.
* Over-reliance can weaken junior developers' learning and puts pressure on entry-level programming jobs.


### Existing Implementations

* [**GitHub Copilot**](https://github.com/features/copilot)
An AI pair programmer integrated into IDEs such as VS Code, JetBrains, and Visual Studio. It uses LLMs trained on public code to autocomplete lines, turn natural-language comments into code, help find vulnerabilities, and generate unit tests. GitHub reports developers completing tasks up to 55% faster with Copilot.


## 4. Use Case: Customer Operations – AI Support Agents

### Business Value

* Automates a large share of routine (Tier-1) requests such as billing questions, returns, and refunds.
* Lowers average handle time (AHT) and support costs and offers support 24/7 in many languages.
* Lets human agents focus on complex or sensitive cases.


### Technical Challenges

* **Multi-intent messages:** a single customer message can contain several requests that the agent must split and resolve correctly.
* **Real-time data access:** the agent must safely read and act on backend systems (orders, CRM, payments).
* **Jailbreaking and prompt attacks:** users can try roleplay, token obfuscation, or hypothetical scenarios to make the agent break its rules or reveal data.
* **Human escalation:** the system must recognise when it can't help and hand over to a person smoothly.


### Weaknesses

* AI agents struggle with emotional, unusual, or complicated cases, and pushing too much to automation can lower service quality.
* A wrong refund or policy statement given to a customer creates financial and reputational risk.
* Customer data in prompts raises privacy and governance concerns.


### Existing Implementations

* [**Salesforce Agentforce**](https://www.salesforce.com/agentforce/) (formerly Einstein Copilot)
An enterprise AI agent platform built into the Salesforce CRM (Sales, Service, and Marketing Clouds). It connects generative models to the company's internal data (Data Cloud) and carries out actions such as summarising client histories, drafting personalised emails from live data, prioritising high-risk accounts, and resolving service cases end to end.
* [**Klarna AI Assistant**](https://www.klarna.com/)
A multilingual support agent built into the Klarna app for payments, disputes, and "Buy Now, Pay Later" questions. In 2024 Klarna said it handled about two-thirds of customer chats and did the work of around 700 full-time agents. In 2025 the CEO admitted the cost-cutting push had gone too far and hurt quality, and Klarna started hiring human agents again in a hybrid model. This makes it a key case study on balancing automation with human support.


## 5. Major Problems Created by GenAI

The rapid commercialisation of GenAI has outpaced regulation and ethical frameworks, creating risks in five main areas:

* **Deepfakes and misinformation.** Hyper-realistic fake audio, video, and images are used for political propaganda, financial fraud, and extortion.
* **AI hallucinations.** Models fabricate "facts," non-existent legal precedents, or flawed medical advice, which erodes trust.
* **Copyright and intellectual property.** Models are trained on large volumes of copyrighted works (art, books, code) without consent or payment to creators, leading to major lawsuits.
* **Job displacement.** GenAI creates new roles but puts pressure on copywriters, translators, entry-level programmers, and support staff.
* **Environmental footprint.** Training and running large models requires huge amounts of electricity and water for data-centre cooling.


## 6. Summary

|Use case|Business value|Key challenges / weaknesses|Example|
|-|-|-|-|
|Financial services|Fast, cited answers from policies and research; better credit decisions|Hallucination risk, retrieval quality, data security, explainability|AI @ Morgan Stanley, Zest AI|
|Software engineering|Faster coding, testing, onboarding|Limited codebase context, insecure code, licensing|GitHub Copilot|
|Customer operations|Lower support costs, 24/7 multilingual service|Multi-intent requests, jailbreaking, quality drop without humans|Salesforce Agentforce, Klarna|

**Core technologies:** Transformer models, embeddings, RAG, RLHF, model catalogs, and AI-native tools.