---
title: "LAION-BVD: A 10-Million-Hour Open Video Dataset for Multimodal Pre-training"
authors: ["Hochlehnert et al."]
year: 2026
arxiv: "2608.24845"
url: https://arxiv.org/abs/2608.24845
priority: Good-To-Read
read_on: 2026-09-04
tags: [paper, llm, vision, theory]
---
## The Core Idea

Open image-text data is huge (LAION-5B: 2.3B English pairs). Open video-text data is not — the biggest before this, InternVid, has 7.1M videos and 760k hours. The bottleneck was never that videos are scarce. It is that downloading and processing them is a brutal engineering job: videos live behind platform gates, they are big, and you need to split, decode, and caption them.

LAION-BVD is the result of just doing that job at scale. Starting from CommonCrawl, the authors extracted **1.3B video URLs** from YouTube, Vimeo and Dailymotion, and actually downloaded **80M videos = 10 million hours** of footage. That is roughly **13× the total duration of InternVid**.

The second, less obvious idea: one video is three datasets. The same download gives you (a) video clips with captions, (b) the audio track with captions, and (c) still frames with captions. So one crawl feeds video-text, audio-text, *and* image-text pre-training.

The third idea is the most interesting empirical claim. Video frames are **not** the same kind of picture as web images. Computing FID (a distance between two image distributions) over CLIP-ViT-B/32 embeddings, two random 100k samples of Re-LAION give FID ≈ **0.16** (same distribution), but Re-LAION vs. LAION-BVD frames gives **33.92**. Video frames are a genuinely different visual distribution — and that shows up sharply in what models trained on them are good at (retrieval: yes; ImageNet classification: no).

> [!NOTE] Fréchet Inception Distance (FID)
> A number measuring how far apart two sets of images are, by fitting a Gaussian to their embeddings and comparing means and covariances. Zero means "same distribution". ^fid

## The Methodology

**Crawl → URLs.** They parse CommonCrawl WAT files (metadata + hyperlinks for archived pages) with `cc2dataset` on an Apache Spark cluster. All dumps up to March 2024 → 4.7B candidate URLs. Keep only links that `yt-dlp` has an extractor for on the three big platforms → 1.3B URLs.

**URLs → videos.** 2,000 virtual servers coordinated by Celery, running `yt-dlp` behind a residential proxy network. They attempted 130M downloads with a **~60% success rate** → 80M videos, 10M hours. No extra safety filter was applied; they lean on the platforms' own moderation.

**Videos → clips.** Randomly sample 2.4M videos. Drop anything shorter than 10s or longer than 30min. Run PySceneDetect (content-aware, threshold 30) to cut at scene boundaries. Then estimate frame-to-frame motion on a downscaled copy and throw away clips with essentially no visual change (static slides, frozen frames). Result: **55M clips, 56k hours** (BVD-V-55M / BVD-A-*).

**Videos → frames.** Separately, pull keyframes with `ffmpeg`, drop black frames, keep scene-change frames (ffmpeg scene threshold 0.1). Result: **BVD-I-300M**, 300M frames. Note the long tail: 51% of videos contribute 9 or more frames.

**Captioning.** Three different small models, one per modality:

| Modality | Captioner | Prompt |
|---|---|---|
| Video clip | Qwen3-VL-2B-Instruct (served with vLLM), 32 frames sampled uniformly | "Describe the video in 20 words or less." |
| Audio | Audio Flamingo 3 | "Describe the audio sounds in 10 words or less." |
| Frame | DeepSeek-VL2-tiny | "Provide a very coarse brief single line of caption for the image." |

The audio prompt length is deliberate: AudioCaps and Clotho captions average ~8 and ~11 words, so matching that keeps the style in distribution.

The frame captioner was picked by a bake-off. Seven VLMs under 4B params were scored with CLIPScore (a reference-free caption-quality proxy: cosine similarity between CLIP image and text embeddings) on 100k DataComp images. DeepSeek-VL2-tiny won on quality (0.62) and came second on throughput (8.20 img/s), beating Qwen2.5-VL-3B (0.55, 2.46 img/s) and SmolVLM (0.50, 2.71 img/s). Captioning 300M frames took **256 A100s and 12,200 A100-hours** — roughly 6× the compute of all the CLIP training runs combined (2,187 A100-hours). Captioning, not training, is the cost centre.

**Validation training.** Three [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|InfoNCE]]-style contrastive models:

- **ViCLIP** (video-text). Two towers, both initialised from a DataComp-1B-pretrained CLIP. The vision tower is a [[An Image is Worth 16x16 Words (ViT)|ViT]] modified to eat patches from *all 8 sampled frames* at once, with both spatial and temporal position embeddings, and full attention across frames. Optimiser [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], $\beta_1=0.9$, $\beta_2=0.98$, weight decay 0.2, grad clip 1.0, cosine schedule with 100 warmup steps, global batch 32k, LR 1e-4 (for 10M samples seen) or 4e-5 (50M). LR swept over {1e-6 … 4e-4}. 100 NVIDIA GH200 GPUs; the biggest run (L-14, 50M samples) is 6.3e19 FLOPs / 455 GPU-hours.
- **CLAP** (audio-text). Audio resampled to 48 kHz, turned into log-Mel spectrograms, HTSAT audio encoder + RoBERTa text encoder, symmetric contrastive loss.
- **CLIP** (image-text) via OpenCLIP, ViT-B/32 and B/16, matched recipe across datasets, batch sizes 2k → 16k as samples-seen grows.

## Ablation Studies and Experiments

### Video (ViCLIP)

Metric is an "overall average": mean of (3 zero-shot classification top-1 scores: K400, UCF-101, HMDB51) and (4 retrieval R@1 scores: MSR-VTT and MSVD, both directions).

| Data | Samples seen | Avg |
|---|---|---|
| DataComp-1B (images only, frame embeddings averaged) | — | 53.2 |
| InternVid-10M-FLT (their reproduction) | 10M | 58.0 |
| BVD-V-10M | 10M | 61.3 |
| BVD-V-50M | 10M | 61.5 |
| BVD-V-50M | 50M | **62.0** |

So minimally-filtered BVD beats the *heavily filtered* InternVid subset by ~4 points. Model scaling is monotonic: B-32 → B-16 → L-14 at 50M samples gives 52.7 → 56.8 → 62.0.

**The baseline story is worth its own paragraph.** Their reproduction of InternVid-10M-FLT scored well below the published number. Digging through a GitHub issue, they found the original authors had used **WiSE-FT** at test time — averaging the fine-tuned weights with the original CLIP checkpoint. Re-running with $\alpha = 0.5$ lifts InternVid from 58.0 to **60.2** and closes most of the gap. Applying the same trick to BVD (with $\alpha$ tuned on a validation split) gives 62.6. This is a clean instance of the problem in [[On the Difficulty of Evaluating Baselines]]: an undocumented evaluation trick silently inflates a published number for years.

**Noise floor.** Five seeds at 50M samples: going from BVD-V-10M to BVD-V-50M gives clear gains on HMDB51 (+1.98, CI [+1.54, +2.42]) and the overall average (+0.58, CI [+0.40, +0.76]), but K400 (+0.06) and UCF-101 (+0.14) are inside the noise. More unique data mainly helps the harder tasks.

**Contamination check.** Overlap of BVD YouTube IDs with test sets: K400 0.26%, MSR-VTT 5.5%, MSVD 6.3%. Re-evaluating on decontaminated test sets moves numbers by ≤0.4 points. Not a problem. (Compare [[Towards Quantifying Benchmark Optimization in ASR Models]].)

**Human audit** of 134 clips: 79.1% accurate, 18.7% minor error (wrong action / object / colour), 2.2% major error. So roughly one caption in five is somewhat wrong, and the models still train fine.

### Audio (CLAP)

Average of UrbanSound8K zero-shot classification + AudioCaps/Clotho retrieval R@5.

At 30M samples seen, single-source training:

| Model size | LAION-Audio | LA + AudioSet | BVD-A-1.7M | BVD-A-10M |
|---|---|---|---|---|
| 158M | 42.1 | **56.4** | 42.4 | 42.9 |
| 431M | 44.8 | **56.6** | 44.5 | **46.8** |

BVD audio matches or beats LAION-Audio, but **curated AudioSet crushes both** by ~10 points. In mixed training (AudioCaps + Clotho + X), swapping LAION-Audio for BVD-A-1.7M loses ~1.5 points (62.3 vs 63.7 at 431M), and adding AudioSet still wins (65.7). Honest conclusion: web-video audio is a fine bulk source but does not replace curated audio.

**What did not work: pushing samples-seen on small models.** Going from 30M to 110M samples seen on BVD-A-10M *hurts* the small models (158M: 42.9 → 40.0; 200M: 44.8 → 41.3) and only helps the big ones (431M: 46.8 → **48.7**). Two effects mixed together — small models bottleneck out, and 110M samples over 10M unique clips is 11 epochs of repetition.

**A contamination scare that turned out fine.** Audio Flamingo 3 (the captioner) had seen 356 of 975 AudioCaps *test clips* during its own training — though paired with AudioSet captions, not the AudioCaps reference captions used for evaluation. Re-evaluating 2,118 checkpoints on an AF3-clean subset (against a size-matched random control, because shrinking the retrieval gallery alone adds ~8.3 pp to R@5) shows BVD-trained models are *less* affected (pooled −1.20 aR@5) than non-BVD models (−2.56). The gain is not contamination.

### Images (frame-based CLIP)

ViT-B/16 at 300M samples seen:

| Dataset | ImageNet-1k | COCO T2I R@5 | COCO I2T R@5 | ImNet-Sketch |
|---|---|---|---|---|
| DataComp-1B | **0.58** | 0.52 | 0.70 | **0.44** |
| Re-LAION | 0.51 | 0.53 | 0.71 | 0.38 |
| BVD-I-300M | 0.28 | **0.63** | **0.80** | 0.19 |

A huge split. Video frames are *better* than web images for retrieval and *much worse* for ImageNet classification. The scaling fits make it worse for classification:

$$\text{ImNet error} \propto C^{-0.19}\ \text{(DataComp)},\qquad C^{-0.07}\ \text{(BVD)}$$
$$\text{COCO retrieval error} \propto C^{-0.15}\ \text{(DataComp)},\qquad C^{-0.20}\ \text{(BVD)}$$

BVD's ImageNet curve is nearly flat — more data will not fix it.

**Why?** They dug in. Counting ImageNet-1k class names (via synsets) in 300M captions from each source: **21M mentions in BVD captions vs 141M in DataComp**. DataComp and Re-LAION were themselves filtered with CLIP scoring against WordNet/ImageNet-style concepts, so they are pre-baked to match ImageNet's vocabulary. BVD's synthetic captions are longer, narrower in length distribution, and COCO-descriptive in style. The benchmark, not the data, is doing half the work — an echo of [[Do ImageNet Classifiers Generalize to ImageNet]].

**A control that separates data from captions:** recaptioning DataComp with the *same* DeepSeek-VL2-tiny pipeline drops ImageNet from 0.40 → 0.23 at 128M samples (BVD gets 0.19) while COCO retrieval rises 0.33 → 0.42. So most of the classification loss is the captioner, not the video frames.

**What did not work: shorter captions.** They tried a prompt forcing ~7-word captions ("1–3 words … very high-level"). It gave a tiny classification bump at 30M samples (0.17 vs 0.15) but cost retrieval (0.28 vs 0.31), so they dropped it.

## Worth Remembering

- **The dataset composition.** 94% YouTube, 4% Vimeo, 2% Dailymotion. 57% English, then Russian 9%, Spanish 8%. Topics: People & Blogs 20%, Music 17%, Entertainment 10%. Mean video 7.7 min, median 3.7 min, ~6% over 30 min. Uploads spanning 2006 to early 2024.
- **Only a slice is captioned.** The 10M-hour headline is the *raw* corpus. The captioned parts are 55M clips from 2.4M videos (56k hours) and 300M frames. Everything else is URLs plus raw video you have to process yourself.
- **Access is gated.** URLs and captions are on HuggingFace; the raw video requires being a research institution and accepting terms of use. Practically, expect to re-download from a link list, and link rot will bite (they already lost 40%).
- **Limitations the authors state plainly.** Captions are short and machine-made, from small models, with their own biases. No generative-model evaluation — everything is contrastive, so nothing tells you if this is good for video diffusion or video LLMs. No joint audio-visual training, so the "the audio and video are already aligned" advantage is untested. No safety filtering beyond platform moderation.
- **The practical lesson for anyone curating data:** captioning is where the compute goes (12,200 vs 2,187 A100-hours here), and the *style* of the caption determines which benchmark you win. If you want ImageNet-style zero-shot classification, you need class-centric vocabulary in your captions — fluency and length are not enough.
- **Open question.** Does the flat ImageNet scaling exponent (0.07) mean video frames are fundamentally worse for object-centric classification, or would a captioner instructed to name concrete objects close the gap? The DataComp-recaption control suggests the latter, but nobody ran it.
- Relates nicely to [[The Bitter Lesson (essay)]]: the entire contribution is scale plus a boring pipeline, and it beats a smaller, more carefully filtered dataset.

## Links

Related: [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Shortcut Learning in Deep Neural Networks]] · [[Denoising Diffusion Probabilistic Models]]

New topics worth writing: CLIP (Learning Transferable Visual Models from Natural Language Supervision), ViCLIP and video-text contrastive encoders, CLAP and audio-text contrastive learning, WiSE-FT weight-space checkpoint merging, Fréchet Inception Distance, CLIPScore as a reference-free caption metric, CommonCrawl and web-scale data pipelines, InternVid, DataComp benchmark, LAION-5B / Re-LAION, synthetic recaptioning of web data, PySceneDetect and shot-boundary detection
