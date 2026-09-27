
# VideoITG: Multimodal Video Understanding with Instructed Temporal Grounding

While Video Large Language Models (Video-LLMs) have shown significant potential in multimodal understanding and reasoning tasks, efficiently selecting the most informative frames from videos remains a critical challenge. To address this, **Instructed Temporal Grounding for Videos (VideoITG)** provides a framework that adaptively customizes frame sampling strategies based on user instructions.

VideoITG is supported by **VidThinker**, an automated annotation pipeline that:

1. Generates instruction-conditioned clip captions.
2. Retrieves relevant video segments with instruction-guided reasoning.
3. Performs fine-grained frame localization.

Using VidThinker, the **VideoITG-40K** dataset was built with **40K videos and 500K temporal grounding annotations**. The plug-and-play VideoITG model leverages visual-language alignment and reasoning for discriminative frame selection, consistently improving downstream video LLM performance across multiple multimodal video understanding benchmarks.

---

## Contents

* [Overview & Architecture](https://www.google.com/search?q=%2523overview--architecture&utm_source=gemini)
* [Performance Benchmarks](https://www.google.com/search?q=%2523performance-benchmarks&utm_source=gemini)
* [Visual Examples](https://www.google.com/search?q=%2523visual-examples&utm_source=gemini)
* [Inference](https://www.google.com/search?q=%2523inference&utm_source=gemini)
* [Installation](https://www.google.com/search?q=%2523installation&utm_source=gemini)
* [Training Data](https://www.google.com/search?q=%2523training-data&utm_source=gemini)
* [Checkpoint Preparation](https://www.google.com/search?q=%2523checkpoint-preparation&utm_source=gemini)
* [Training](https://www.google.com/search?q=%2523training&utm_source=gemini)
* [Evaluation](https://www.google.com/search?q=%2523evaluation&utm_source=gemini)
* [License & Terms of Use](https://www.google.com/search?q=%2523license--terms-of-use&utm_source=gemini)
* [Acknowledgements](https://www.google.com/search?q=%2523acknowledgement&utm_source=gemini)

---

## Overview & Architecture

VideoITG acts as a high-precision, instruction-aware frame selector before passing visual data into heavy downstream Video-LLMs.

1. **Dense Frame Sampling**: Decodes video streams into an initial sequence of frames (e.g., 512 frames at 1 FPS).
2. **Instruction-Guided Scoring**: A score head evaluates each frame's relevance relative to the specific user query or task instruction using a sigmoid output.
3. **Top-K Sorting**: Ranks frames by score, selects the $K$ most informative frames, and re-orders them chronologically for optimal visual continuity.
4. **LLM Consumption**: Passes the compact, high-value frame subset to downstream models like InternVL, Qwen3-VL, or Eagle.

---

## Performance Benchmarks

Below is a comparison between baseline uniform frame sampling (`UNI-32`) and VideoITG-selected sampling (`ITG-32`) across standard evaluation benchmarks:

| Downstream Video-LLM | Selection Strategy | LongVideoBench | MLVU | VideoMME-S | VideoMME-M | VideoMME-L | CG-Bench (mini) | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **InternVL2.5-8B** | UNI-32 | 58.3 | 66.4 | 75.1 | 61.7 | 53.1 | 37.7 | 58.7 |
| **InternVL2.5-8B** | ITG-32 | **61.9** (+3.6) | **75.0** (+8.6) | **78.0** (+2.9) | **67.1** (+5.4) | **56.9** (+3.8) | **46.7** (+9.0) | **64.3** (+5.6) |
| **InternVL2.5-26B** | UNI-32 | 55.6 | 71.3 | 78.1 | 67.1 | 56.9 | 40.6 | 61.6 |
| **InternVL2.5-26B** | ITG-32 | **63.0** (+7.4) | **78.9** (+7.6) | **80.8** (+2.7) | **69.0** (+1.9) | **59.9** (+3.0) | **66.7** (+5.1) | **66.7** (+5.1) |
| **InternVL3.5-8B** | UNI-32 | 60.0 | 70.0 | 77.0 | 62.4 | 53.4 | 40.9 | 60.6 |
| **InternVL3.5-8B** | ITG-32 | **65.7** (+5.7) | **74.1** (+4.1) | **78.4** (+1.4) | **65.9** (+3.5) | **59.0** (+5.6) | **47.6** (+6.7) | **65.1** (+4.5) |
| **Qwen3-VL** | UNI-32 | 59.1 | 64.1 | 76.0 | 60.9 | 55.1 | 40.1 | 59.2 |
| **Qwen3-VL** | ITG-32 | **63.6** (+4.5) | **77.2** (+13.1) | **79.9** (+3.9) | **66.6** (+5.7) | **60.3** (+5.2) | **47.3** (+7.2) | **65.8** (+6.6) |
| **LLaVA-Video-7B** | UNI-32 | 58.7 | 66.8 | 76.3 | 60.3 | 52.7 | 35.8 | 58.4 |
| **LLaVA-Video-7B** | ITG-32 | **61.6** (+2.9) | **74.6** (+7.8) | **77.3** (+1.0) | **65.9** (+5.6) | **55.2** (+2.5) | **42.8** (+7.0) | **62.9** (+4.5) |
| **Eagle2.5-8B** | UNI-32 | 63.0 | 67.8 | 78.8 | 64.1 | 55.9 | 41.2 | 61.8 |
| **Eagle2.5-8B** | ITG-32 | **66.8** (+3.8) | **76.5** (+8.7) | **80.0** (+1.2) | **67.8** (+3.7) | **60.3** (+4.4) | **49.0** (+7.8) | **66.7** (+4.9) |

---

## Visual Examples

---

## Inference

### Checkpoints

* **VideoITG Checkpoint (Top‑K selector)**: [`nvidia/VideoITG-8B`](https://huggingface.co/nvidia/VideoITG-8B?utm_source=gemini)

### How Frame Selection Works (512 $\rightarrow$ Sort $\rightarrow$ Top‑K)

1. **Sampling**: The selector scores **512 uniformly sampled frames** (default setting) using a sigmoid scoring head.
2. **Ranking**: Frames are sorted by score in **descending order**.
3. **Filtering & Chronological Reordering**: The Top‑K highest-scoring frames are selected and re-sorted in **ascending chronological order** before being passed into the downstream Video-LLM.

Refer to the reference implementation in [`infer.py`](https://www.google.com/search?q=infer.py&utm_source=gemini) for direct usage.

### JSONL Schema Explained

The pipeline utilizes two structured JSONL file types:

1. **Grounding Output (`results.jsonl`)**
* Output from running `--model videoitg`.
* Default path: `${output_dir}/results.jsonl`.
* Contains frame indices sorted by score in **descending** order alongside their unnormalized logits:


```json
{
  "doc_id": 12,
  "video_path": "/path/to/video.mp4",
  "contexts": "Instruction or prompt text...",
  "index": [120, 60, 180],
  "logits": [0.98, 0.97, 0.95]
}

```


2. **Downstream Selection File (`frame_indices_jsonl`)**
* Consumed by downstream models (InternVL, Qwen3-VL, Eagle).
* Contains the chosen Top-K frame indices re-sorted in **ascending chronological order**:


```json
{
  "doc_id": 12,
  "index": [60, 120, 180]
}

```



---

## Installation

### Prerequisites & Environment Setup (Linux)

```bash
# 1. Clone the repository
git clone https://github.com/NVlabs/VideoITG.git
cd VideoITG

# 2. Create and activate Conda environment
conda create -n videoitg python=3.12 -y
conda activate videoitg

# 3. Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install optimized attention backends for training
pip install flash-attn==2.4.2 --no-build-isolation

```

---

## Training Data

* **Pretraining & Instruction Tuning Data**: Built on standard multimodal datasets, including [CC3M Pretrain 595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K?utm_source=gemini), [LLaVA-OneVision](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data?utm_source=gemini), and [LLaVA-Video 178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K?utm_source=gemini).
* **Grounding Dataset**: [VideoITG-40K Dataset](https://huggingface.co/datasets/NVEagle/VideoITG-40K?utm_source=gemini) containing 40,000 videos and 500,000 fine-grained temporal grounding annotations.

---

## Checkpoint Preparation

Pretrained base models and fine-tuned Video-LLM weights can be fetched directly from HuggingFace repository:

* [`eagle-qwen2-7b-finetune-uni-ov-video-finetune-sftv1`](https://huggingface.co/exiawsh/eagle-qwen2-7b-finetune-uni-ov-video-finetune-sftv1?utm_source=gemini)

---

## Training

To launch the grounding fine-tuning procedure:

```bash
bash scripts/videoitg/finetune-uni-64frame-qwen2-7b-grounding.sh finetune 16

```

### Resource Requirements

* Standard training setup uses **128 $\times$ NVIDIA A100 (80GB)** GPUs (~4 hours execution time).
* For lower GPU memory budgets: reduce `per_device_train_batch_size` and scale `gradient_accumulation_steps` proportionally.

---

## Evaluation

Evaluation pipeline utilizes `lmms_eval` integrated with `accelerate`.

### Step 1: Run VideoITG Grounding Stage

Generate score predictions across the target benchmark dataset:

```bash
bash scripts/eval_lmms_eval/videomme_grounding.sh

```

This generates `results.jsonl` under `./videomme_result_512` containing per-frame relevance scores.

### Step 2: Run Downstream Video-LLM Evaluation

Pass the top selected frames into the target model (e.g., InternVL2.5):

```bash
bash scripts/eval_lmms_eval/internvl2.5.sh

```

> **Note**: In `scripts/eval_lmms_eval/internvl2.5.sh`, set `frame_indices_jsonl` to point to your grounding output file, and set `num_frame` (e.g., `32`) to control the target Top-K count.

### Script Arguments

* `--tasks`: Target benchmark (e.g., `videomme`, `mlvu`, `longvideobench_val_v`, `cgbench_subtitles`).
* `--model`: Backend architecture identifier (`videoitg`, `internvl2`, `internvl3_5`, `qwen3_vl`, `eagle2_5`).
* `--model_args`:
* `pretrained`: HuggingFace checkpoint repository or local path.
* `num_frames`: Number of frames uniformly sampled prior to scoring (e.g., `512`).
* `target_fps`: Extraction sampling rate (e.g., `1`).
* `frame_indices_jsonl`: Path to Top-K mapped frame index JSONL.
* `num_frame` / `max_num_frames`: Frame count fed to downstream VLM.



---

## License & Terms of Use

* Code is distributed under the [Apache 2.0 License](https://www.google.com/search?q=LICENSE&utm_source=gemini).
* Portions adapted from `lmms-eval` retain their original software licenses.
* Model weights are provided under the [NVIDIA Model License](https://www.google.com/search?q=LICENSE_Model&utm_source=gemini) for non-commercial research purposes.
* Base Language Model: [Qwen2-7B-Instruct (Apache 2.0)](https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/LICENSE?utm_source=gemini)
* Vision Encoder: [SigLIP (Apache 2.0)](https://huggingface.co/google/siglip-so400m-patch14-384?utm_source=gemini)



---

## Acknowledgement

* [EAGLE](https://github.com/NVlabs/EAGLE?utm_source=gemini): Base codebase and architecture frameworks.
* [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval?utm_source=gemini): Standardized evaluation tools and bench harnesses.
* [LLaVA-OneVision](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data?utm_source=gemini) & [LLaVA-Video](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K?utm_source=gemini): Open datasets enabling multimodal training.
