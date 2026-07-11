# Fine-Tuning LLMs — Clear Explanation
### LoRA · QLoRA · Non-Instruction FT · SFT · DPO

> **Author:** Basazin (Bassa) Belhu · AI World with Bassa

---

## 1 — Why Full Fine-Tuning Is Expensive

When you fully fine-tune a model, you update **every single parameter** — all 7 billion of them for a model like LLaMA-2 7B. The problem is not just storing the weights. Training requires storing **four things per parameter**: the weight itself, its gradient, and two optimizer states (momentum and velocity for Adam). That adds up to 16 bytes per parameter.

For a 7B model, that is **112 GB of memory** just to run one training step — before you even count the activations, the data, or anything else. A single A100 GPU has 80 GB. You physically cannot fit it on one machine without using 2–4 high-end GPUs.

Beyond memory, there are two more problems:

- **Catastrophic forgetting** — updating every weight at once causes the model to lose its general knowledge while learning your task. It becomes good at your domain but forgets everything else.
- **Cost** — renting the GPUs needed costs thousands of dollars per training run. For a startup or individual, this is completely impractical.

> **The core issue is simple:** training requires storing way more than just the model weights, and for large models that total is enormous.

---

## 2 — What LoRA Does

LoRA (Low-Rank Adaptation) is built on one key observation: when a model adapts to a new task, the **changes to its weights are low-rank**. This means the update does not need to use the full complexity of a giant weight matrix — it can be approximated by two much smaller matrices multiplied together.

Instead of updating a large weight matrix directly, LoRA inserts **two small trainable matrices** (called A and B) alongside it. The original weight matrix is completely frozen — it never changes. Only A and B are trained. When you need the updated weight, you simply add the product of A and B to the frozen original.

In practice, for a weight matrix that has 16 million parameters, LoRA might replace it with two matrices that together have only 65,000 parameters — about 256 times fewer. Across the whole model, this means instead of training billions of parameters, you train roughly 4 million. That is about 0.06% of the total.

The memory savings are dramatic. Because A and B are tiny, their gradients and optimizer states are tiny too. The frozen weights need no gradient at all. Total memory drops from around 112 GB to around 28 GB for a 7B model — a 75% reduction.

> **The mental model:** The original model is a library of frozen knowledge. LoRA adds a small sticky-note booklet on top. You only write in the booklet, but when someone reads the book, they automatically see both.

---

## 3 — What QLoRA Does

QLoRA takes LoRA one step further by also compressing the frozen base model using **4-bit quantization**. It combines two ideas:

**First:** The frozen base model is stored in 4-bit precision instead of 16-bit or 32-bit. This is called NF4 (NormalFloat4). Neural network weights follow a predictable distribution — most of them cluster near zero. NF4 takes advantage of this by placing quantization levels where the weights actually are, rather than spacing them out uniformly. The result is that the compression is far more accurate than naive 4-bit rounding would be.

**Second:** The LoRA adapters (the only parts that actually learn) are kept in full 16-bit precision. Nothing about the training changes — only the storage of the frozen background model is compressed.

A 4-bit model takes up about a quarter of the space of a 16-bit model. Combined with LoRA, a 7B model that would need 28 GB in standard LoRA now needs only about 6–8 GB in QLoRA. A consumer GPU like an RTX 3090 can handle it.

QLoRA also introduces **paged optimizers** — a technique that temporarily moves optimizer states from GPU memory to CPU memory when the GPU gets full, then moves them back when needed. Think of it as virtual memory for your GPU. This prevents out-of-memory crashes during training.

> **QLoRA = LoRA (small adapters learn) + NF4 (frozen base model compressed to 4-bit) + paged optimizers (don't crash on consumer GPUs)**

---

## 4 — Why QLoRA Is Useful on Limited GPU

The numbers tell the story:

| Method | Memory Needed | What GPU |
|--------|--------------|----------|
| Full fine-tuning, 7B | ~112 GB | 4+ A100 GPUs |
| LoRA, 7B | ~28 GB | A100 40GB (barely) |
| QLoRA, 7B | ~6–8 GB | RTX 3090 / RTX 4090 |
| QLoRA, 70B | ~40 GB | A single A100 80GB |

QLoRA makes it possible to fine-tune a 70B model on hardware that full fine-tuning cannot even load a 7B model onto.

The reason quality stays high despite aggressive compression comes down to **what is compressed vs what is trained.** The frozen base model — the part stored in 4-bit — is never updated. It just sits there providing background context during the forward pass. For that read-only role, 4-bit is accurate enough. The adapters that actually learn remain in full 16-bit precision throughout. No learning quality is sacrificed.

In practice, QLoRA-trained models consistently reach 90–95% of the quality of full fine-tuning. For most real-world tasks — adapting a model to a legal domain, a coding style, or a company's documents — the difference is negligible.

> **QLoRA democratised fine-tuning.** Before it, fine-tuning large models required expensive cloud infrastructure. After it, a researcher with a single gaming GPU could do competitive fine-tuning.

---

## 5 — What Is Non-Instruction Fine-Tuning?

Non-instruction fine-tuning (also called continued pre-training or domain adaptive pre-training) is the simplest form of fine-tuning. You take a base model and train it to **continue predicting the next word** on a large collection of domain-specific text — exactly the same task the model was originally pre-trained on, just with your data instead of the internet.

There are no questions, no answers, no instructions. You simply feed the model raw text from your domain — legal documents, medical papers, scientific articles, internal company reports — and let it absorb the patterns.

The goal is **vocabulary and concept absorption**. A general-purpose model trained on internet text might not know that "I-290B" is a specific immigration form, or that "en banc" has a precise legal meaning, or that certain chemical notation means something specific. After non-instruction fine-tuning on domain text, it does.

This step is usually done **before** instruction fine-tuning. Think of it as making the model read your entire document library before you start teaching it how to answer questions about that library. It builds the knowledge foundation that instruction fine-tuning will later teach the model to apply.

> **Non-instruction FT answers the question:** "What should the model know?"  
> **Instruction FT (SFT) answers the question:** "How should the model behave?"

---

## 6 — What Is Instruction Fine-Tuning (SFT)?

Instruction fine-tuning, formally called Supervised Fine-Tuning (SFT), is what turns a raw language model into an assistant. A base model that has only done next-word prediction will continue a text you give it — it does not know how to respond to a question or follow a command. SFT teaches it that skill.

The training data is a collection of **instruction-response pairs**: a question or task on one side, and a good answer on the other. During training, the model sees both the instruction and the response, but it is only trained to predict the response tokens. It learns: "when I see an instruction like this, I should produce a response like that."

The result is a model that reads a question and answers it directly, maintains a helpful conversational tone, and follows the format of the training examples. The base model knows facts — the SFT model knows how to be helpful with those facts.

SFT is what produces models like Alpaca (LLaMA + instructions) or Vicuna (LLaMA + conversations). Without SFT, even a brilliantly knowledgeable model would be almost useless to a regular user — it would just continue their prompt rather than help them.

> **The analogy:** Non-instruction FT is teaching someone everything about immigration law. SFT is teaching them how to explain it to a client when asked.

---

## 7 — What Is DPO?

DPO (Direct Preference Optimization) is an alignment technique that teaches the model not just to answer, but to prefer **better answers over worse ones**. After SFT, a model can follow instructions — but given two valid responses to the same question, it has no sense of which is more accurate, more helpful, or more appropriate. DPO gives it that sense.

The training data for DPO consists of **preference triples**: a prompt, a chosen response (the better one), and a rejected response (the worse one). During training, the model is pushed to increase its probability of generating the chosen response and decrease its probability of generating the rejected response — simultaneously.

Crucially, DPO does this **without a reinforcement learning loop**. Earlier alignment methods (like RLHF, used by ChatGPT) required training a separate reward model and then using PPO — a complex, unstable reinforcement learning algorithm — to update the main model. DPO skips all of that. It achieves the same effect with a single mathematical loss function applied directly to preference pairs. The result is simpler to implement, cheaper to run, and more stable to train.

DPO is now the dominant alignment technique for open-source models. LLaMA-3-Instruct, Mistral-Instruct, and Zephyr all use variants of it.

> **OPRO** (from Google DeepMind) is a related but different idea — it optimises the text of prompts rather than model weights, and is closer to automated prompt engineering than fine-tuning.

---

## 8 — Difference Between SFT and DPO

SFT and DPO are not alternatives — they are **sequential stages** in the same pipeline. You always do SFT first, then DPO on top.

They solve different problems:

**SFT** asks: *Can the model answer at all?*  
It teaches the model to be an assistant — to read instructions and respond in a helpful, structured way. Without SFT, the model cannot do the basic task.

**DPO** asks: *Of all the ways it could answer, does it choose the best one?*  
It teaches the model to discriminate between good and bad responses, to prefer accuracy over vagueness, to choose safe over harmful, to pick detailed over unhelpful.

The data is fundamentally different. SFT needs `(instruction, correct response)` pairs — one good answer per question. DPO needs `(prompt, chosen response, rejected response)` triples — two contrasting answers per question, one clearly better.

The training signal is also different. SFT is purely imitative — "predict these exact tokens." DPO is contrastive — "this one should be more likely than that one." SFT teaches ability. DPO teaches judgment.

A practical illustration: after SFT, a model asked to "write a cover letter exaggerating my qualifications" might comply, because SFT only taught it to follow instructions. DPO would have been trained on examples where the honest cover letter was chosen and the exaggerated one was rejected — so after DPO, the model has learned to prefer honesty even when not explicitly told to.

| | SFT | DPO |
|--|-----|-----|
| **Stage** | Step 2 — before DPO | Step 3 — after SFT |
| **Data** | (instruction, response) | (prompt, chosen, rejected) |
| **Teaches** | How to answer | Which answer is better |
| **Signal** | Imitate the correct response | Prefer chosen over rejected |
| **Goal** | Ability | Judgment and alignment |

---

## 9 — Hyperparameter Values and Reasoning

### LoRA Parameters

**`rank = 8`**  
The rank controls how large the LoRA adapter matrices are. A rank of 8 is the most common starting point in production — it is large enough to capture meaningful task-specific patterns while keeping the adapter small enough to train quickly. For simple tasks like classification, a rank of 4 is sufficient. For complex reasoning tasks where the model needs to learn a lot of new behaviour, rank 16 or 32 is more appropriate.

**`alpha = 16`**  
Alpha is a scaling factor that controls how strongly the adapter's updates influence the final output. The effective scale applied is `alpha / rank`. With rank 8 and alpha 16, the scale is 2.0 — meaning the adapter's contribution is amplified 2×. The standard rule of thumb is to set alpha to twice the rank. Setting it too high risks overfitting; too low and the adapter barely affects the output.

**`dropout = 0.05`**  
Dropout randomly zeros some adapter values during training to prevent the small adapter from simply memorising the training examples. A value of 0.05 (5% dropout) is a safe default for most situations. For very small datasets under 1,000 examples, 0.10 provides more regularisation. For large datasets, dropout can be set to 0 since there is enough data variety already.

---

### Training Parameters

**`learning_rate = 2e-4` for SFT, `5e-5` for DPO**  
The learning rate for SFT is set relatively high because the adapter weights start from zero or random values and need to learn from scratch. The DPO learning rate is set much lower because DPO is making subtle preference adjustments to an already well-trained SFT model — a high learning rate here would overwrite the SFT quality rather than refine it.

**`batch size = 4 per GPU, gradient accumulation = 4 → effective batch of 16`**  
Modern transformer training needs a reasonably large effective batch size for stable, noise-free gradients. However, fitting 16 full examples simultaneously on one GPU is impossible for 7B models at typical sequence lengths. Gradient accumulation solves this by running 4 small batches of 4, accumulating their gradients, and then taking one combined update step — achieving the same effect as a batch of 16 without needing the memory.

**`epochs = 3 for SFT, 1 for DPO`**  
SFT typically needs 2–3 passes through the instruction dataset for the model to fully absorb the instruction-following patterns. DPO needs only 1 epoch — it is a fine-tuning of a fine-tuned model, making very targeted preference adjustments. More than 1 epoch of DPO risks the model memorising the specific preference pairs and losing the generality it gained from SFT.

**`scheduler = cosine with 3% warmup`**  
The cosine schedule gradually increases the learning rate from zero to the target value over the first 3% of training steps (the warmup), then smoothly decreases it following a cosine curve back toward zero by the end. This prevents the unstable, noisy updates that happen when you start training at full learning rate immediately, and gives better final performance than stopping abruptly or decaying linearly.

**`optimizer = paged_adamw_32bit`**  
The paged version of AdamW is used specifically because of QLoRA. During backward passes, the GPU can run out of memory when handling the optimizer states for even the tiny adapter matrices. Paged AdamW automatically moves these states to CPU RAM when needed and brings them back when the GPU has space — preventing out-of-memory crashes on consumer GPUs without meaningfully slowing down training.

---

### Priority Order — What to Tune First

When you have limited time to experiment, tune hyperparameters in this order:

1. **Learning rate first** — this has the biggest effect on whether training converges at all. If the loss spikes or diverges, lower it. If training is stable but very slow to improve, raise it slightly.

2. **Rank second** — if the model clearly underfits (validation loss remains high), increase the rank to give it more capacity. If you have a small dataset and the model overfits, decrease the rank.

3. **Effective batch size third** — if the training loss is very noisy and unstable, increase the effective batch size. This matters less than learning rate and rank but contributes to stability.

4. **Alpha last** — the `alpha = 2 × rank` rule of thumb is reliable enough that alpha rarely needs to be tuned independently. Change it only if the other adjustments have not helped.

---
