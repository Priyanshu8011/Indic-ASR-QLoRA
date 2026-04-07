# Advancing ASR for Diverse Indian Accents using QLoRA

## 📌 Project Overview
State-of-the-art Automatic Speech Recognition (ASR) foundation models achieve near-human transcription accuracy on global languages but suffer severe "acoustic mismatch" when exposed to regional Indian accents. Traditional full-network fine-tuning to correct this mismatch is computationally prohibitive, requiring enterprise-grade hardware. 

This repository contains the complete engineering pipeline for adapting OpenAI's foundational `whisper-base` model to diverse Indian accents. By implementing **Quantized Low-Rank Adaptation (QLoRA)**, this project successfully achieves competitive, corpus-level dialect adaptation using strictly consumer-grade hardware (a single NVIDIA T4 16GB GPU).

## 🚀 Architecture & Stack
* **Foundation Model:** `openai/whisper-base`
* **Compression:** 4-bit NormalFloat (NF4) Quantization via `bitsandbytes`
* **Adaptation:** LoRA matrices injected into attention modules (`q_proj`, `v_proj`) via `peft`
* **Feature Extraction:** 16,000 Hz Log-Mel Spectrograms
* **Evaluation Metric:** Corpus-Level Word Error Rate (WER) via `jiwer`

## 📊 Dataset
The model was fine-tuned and evaluated on the open-source **ai4bharat/Svarah** dataset. 
* **Size:** 9.6 hours of transcribed audio.
* **Scope:** 117 speakers spanning 65 different geographic locations across India.
* **Preprocessing:** Engineered dynamic lambda functions to bypass low-level `torchcodec` driver faults by dropping corrupted audio arrays during ingestion.

## 📈 Ablation Study & Results
A rigorous mathematical ablation study was conducted to evaluate the performance trade-offs between different LoRA adapter ranks against transcription accuracy and computational overhead. 

The established baseline for the un-tuned Whisper-Base model on the Svarah dataset is a **13.6% WER** (Javed et al., 2023).

| LoRA Rank | Trainable Parameters | Word Error Rate (WER) |
| :--- | :--- | :--- |
| **Baseline** | *Zero-shot (None)* | 13.60% |
| **Rank 16** | ~0.1% | [6.72%] |
| **Rank 32** | ~0.2% | [4.88%] |
| **Rank 64** | ~0.4% | [4.51%] |

*(Note: The final WER is a micro-averaged Corpus-Level metric, calculated as the absolute sum of all insertions, deletions, and substitutions across the test split to natively account for the proportional representation of each dialect).*
