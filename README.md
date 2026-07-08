# 🩺 Healthcare AI Assistant using Unsloth

An end-to-end domain-specific Large Language Model (LLM) fine-tuning project that builds a Healthcare FAQ Assistant using Unsloth. The project follows a complete three-stage training pipeline: domain adaptation, supervised instruction fine-tuning, and preference alignment with DPO.

---

# Project Overview

This project demonstrates how to transform a general-purpose LLM into a specialized Healthcare AI Assistant capable of providing more accurate, structured, and domain-specific responses.

The workflow follows an industry-style LLM fine-tuning pipeline:

```
Base LLM
      │
      ▼
Stage 1: Non-Instruction Fine-Tuning
      │
      ▼
Stage 2: Instruction Fine-Tuning (SFT)
      │
      ▼
Stage 3: DPO Preference Alignment
      │
      ▼
Healthcare AI Assistant
```

---

# Domain

Healthcare FAQ Assistant

The assistant answers healthcare-related questions involving:

- Diabetes
- Hypertension
- Liver Disease
- Cardiovascular Disease
- Cancer
- Infection Prevention
- Vaccination
- Asthma
- Chronic Kidney Disease
- Public Health
- Patient Safety
- Preventive Medicine

---

# Business Problem

Healthcare professionals and patients often require reliable, domain-specific answers rather than generic responses produced by foundation models.

The objective is to fine-tune an open-source LLM so it:

- Understands medical terminology
- Produces more structured responses
- Provides safer and more informative healthcare answers
- Reduces generic or incomplete responses

---

# Project Structure

```
healthcare-ai-assistant/
│
├── assets/
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
│
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
│
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   ├── base_answers.json
│   ├── sft_answers.json
│   └── dpo_answers.json
│
├── saved_models/
│   ├── non_instruction_adapter/
│   ├── final_medical_assistant/
│   └── final_medical_assistant_dpo/
│
└── README.md
```

---

# Dataset

## Stage 1 — Non-Instruction Dataset

Source:

- Curated healthcare educational content
- Public medical knowledge
- Healthcare reference material
- AI-generated content reviewed and verified

Contains:

- 50+ healthcare knowledge paragraphs

Purpose:

- Learn healthcare terminology
- Adapt language style
- Improve domain understanding

---

## Stage 2 — Instruction Dataset

Source:

Medical Meadow Medical Flashcards (MedAlpaca)

Contains:

- 100+ instruction-response examples

Each record includes:

- Instruction
- Response

Purpose:

- Teach the model how to answer healthcare questions.

---

## Stage 3 — Preference Dataset

Custom preference dataset containing:

- Prompt
- Preferred (Chosen) response
- Rejected response

Contains:

- 50 healthcare preference examples

Purpose:

- Improve response quality using Direct Preference Optimization (DPO).

---

# Base Model

- Meta Llama 3.1 8B
- Loaded using Unsloth
- 4-bit QLoRA quantization

---

# Fine-Tuning Pipeline

## Stage 1 — Non-Instruction Fine-Tuning

Objective

Adapt the base model to healthcare language without question-answer supervision.

Process

- Load raw healthcare text
- Clean and prepare dataset
- Apply LoRA adapters
- Train with QLoRA
- Save Stage 1 adapter

Output

```
saved_models/non_instruction_adapter/
```

---

## Stage 2 — Instruction Fine-Tuning

Objective

Teach the model to answer healthcare questions.

Process

- Load Stage 1 model
- Format instruction dataset
- Apply supervised fine-tuning (SFT)
- Save instruction-tuned model

Output

```
saved_models/final_medical_assistant/
```

---

## Stage 3 — DPO Preference Alignment

Objective

Improve response quality by learning preferred answers.

Process

- Load Stage 2 model
- Train using prompt, chosen, and rejected responses
- Align the model with preferred response behavior

Output

```
saved_models/final_medical_assistant_dpo/
```

---

# LoRA / QLoRA Configuration

| Parameter | Value |
|-----------|------:|
| Rank (r) | 16 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Quantization | 4-bit QLoRA |
| Optimizer | AdamW 8-bit |
| Max Sequence Length | 2048 |
| Learning Rate | 2e-4 (SFT), 5e-5 (DPO) |

---

# Training

Training performed using:

- Unsloth
- Hugging Face Transformers
- TRL
- PEFT
- PyTorch

Include screenshots of:

- Stage 1 training logs
- Stage 2 SFT logs
- Stage 3 DPO logs

under the `assets/` folder.

---

# Model Evaluation

Three models were evaluated:

- Base Model
- Instruction Fine-Tuned Model (SFT)
- DPO-Aligned Model

Evaluation criteria:

- Correctness
- Domain Accuracy
- Helpfulness
- Safety
- Clarity
- Professional Tone
- Hallucination Reduction

Detailed reports are available in:

```
reports/
```

---

# Results

The DPO-aligned model consistently produced:

- More complete responses
- Better medical terminology
- Improved structure
- Safer recommendations
- Less generic answers
- Stronger domain-specific behavior

---

# Challenges

- Managing GPU memory during multi-stage fine-tuning
- Running multiple 8B models within Colab GPU limits
- Organizing intermediate model checkpoints
- Building high-quality preference pairs for DPO
- Maintaining consistent prompt formatting across training stages

---

# Future Improvements

- Expand the instruction dataset with more clinical scenarios
- Add Retrieval-Augmented Generation (RAG)
- Integrate medical knowledge bases
- Deploy using FastAPI and Docker
- Add automated LLM evaluation benchmarks
- Support multi-turn healthcare conversations
- Implement guardrails and hallucination detection

---

# Technologies

- Python
- Unsloth
- PyTorch
- Hugging Face Transformers
- TRL
- PEFT
- LoRA
- QLoRA
- DPO
- Pandas
- Google Colab

---

# Author

**Bassa Belhu**

Healthcare AI Assistant — End-to-End LLM Fine-Tuning with Unsloth