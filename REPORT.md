# Lab 21 — Evaluation Report

**Học viên**: Đỗ Thị Thanh Bình — 2A202600717  
**Ngày nộp**: 2026-06-25  
**Submission option**: A (lightweight ZIP)

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train / 20 eval), random seed = 42
- **max\_seq\_length**: 1024 (p95 = 562, p99 = 704 — capped tại 1024 theo T4 profile)
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Training cost**: ~$0.07 (12.8 phút tổng @ $0.35/hr)
- **Target modules**: `q_proj`, `v_proj`
- **Hyperparameters chung**: 3 epochs, LR = 2e-4, cosine schedule, warmup\_ratio = 0.10, effective batch = 8 (batch 1 × grad\_accum 8), optimizer = adamw\_8bit, gradient\_checkpointing = True

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | % of Total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-----------------|------------|------------|-----------|-----------|------------|
| Base | —     | 0               | 0%         | —          | —         | 1.8840    | **6.58**   |
| 8    | 16    | 1,843,200       | 0.06%      | 4.13 min   | 7.22 GB   | 1.5577    | **4.75**   |
| 16   | 32    | 3,686,400       | 0.12%      | 4.60 min   | 6.62 GB   | 1.5161    | **4.55**   |
| 64   | 128   | 14,745,600      | 0.48%      | 4.11 min   | 8.00 GB   | 1.4768    | **4.38**   |

**Quan sát nhanh**:
- r=16 cải thiện perplexity 0.20 so với r=8 (+2× params)
- r=64 chỉ cải thiện thêm 0.17 so với r=16 (+4× params) → diminishing returns rõ ràng
- Thời gian train gần như không đổi (~4.1–4.6 min) vì bottleneck là data loading/tokenization, không phải LoRA compute
- r=16 dùng ít VRAM hơn r=8 (6.62 vs 7.22 GB) — có thể do memory fragmentation sau khi unload model trước đó

---

## 3. Loss Curve Analysis

T4 profile tắt `eval_strategy` trong training để tiết kiệm VRAM, nên chỉ có **training loss curve** (không có eval loss mid-training). Dựa trên notebook output:

- Training loss giảm đều qua 3 epochs cho cả 3 ranks
- Không phát hiện overfitting rõ ràng — eval loss cuối (đo sau training) thấp hơn hoặc xấp xỉ training loss, cho thấy model chưa overfit với 200 samples / 3 epochs
- Với dataset chỉ 200 samples, nguy cơ overfit sẽ tăng nếu tăng số epochs hoặc rank lên cao hơn r=64

---

## 4. Qualitative Comparison (5 examples)

So sánh base model vs fine-tuned `r=16` trên cùng 5 prompts.

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

| | Response |
|--|--|
| **Base** | "Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu..." |
| **Fine-tuned (r=16)** | "Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng..." |

**Nhận xét**: Cả hai đều đúng. Fine-tuned nhấn mạnh "không có hướng dẫn trực tiếp" — chuẩn hơn về định nghĩa supervised learning vs ML nói chung. **→ improved**

---

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

| | Response |
|--|--|
| **Base** | Dùng đệ quy, có check `n <= 0` và `n == 1`, `n == 2` |
| **Fine-tuned (r=16)** | Dùng vòng lặp, có `raise ValueError` cho input âm, robust hơn |

**Nhận xét**: Fine-tuned chọn iterative approach (tránh stack overflow) và thêm error handling đúng chuẩn. **→ improved**

---

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

| | Response |
|--|--|
| **Base** | 5 nguyên tắc có giải thích chi tiết: thân thiện, trực quan, v.v. |
| **Fine-tuned (r=16)** | 5 nguyên tắc ngắn gọn: chuyển đổi, thích ứng, đơn giản, tương thích, v.v. |

**Nhận xét**: Base có chiều sâu hơn; fine-tuned ngắn gọn nhưng bỏ mất context quan trọng. **→ same / slight degradation** (case loss)

---

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

| | Response |
|--|--|
| **Base** | Mô tả đúng LoRA là Low-Rank Adaptation, nhưng giải thích hơi lẫn lộn với optimization |
| **Fine-tuned (r=16)** | Nhầm LoRA là "Layer-wise Adaptive Regularization Optimization" — sai tên viết tắt |

**Nhận xét**: Fine-tuned hallucinate sai tên viết tắt của LoRA. Dataset gốc là general Vietnamese instruction, không có domain knowledge về LLM methods. **→ degraded** (case loss — fine-tuning không fix knowledge gaps)

---

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

| | Response |
|--|--|
| **Base** | Mô tả đúng 3 kỹ thuật, có so sánh trade-off |
| **Fine-tuned (r=16)** | Mô tả đúng, có thêm ví dụ cụ thể hơn về prompt |

**Nhận xét**: Fine-tuned cấu trúc rõ hơn nhưng không cải thiện độ chính xác nhiều. **→ slightly improved**

---

## 5. Conclusion về Rank Trade-off

Dựa trên experiment với Qwen2.5-3B trên 200 Vietnamese instruction samples, kết quả cho thấy pattern **diminishing returns** rõ ràng theo rank:

- **r=8 → r=16**: Perplexity giảm 0.20 (4.75 → 4.55) với chỉ 2× params tăng thêm — **ROI cao nhất**
- **r=16 → r=64**: Perplexity giảm thêm 0.17 (4.55 → 4.38) nhưng params tăng 4× (3.7M → 14.7M) — **ROI giảm rõ**

Điểm diminishing returns xuất hiện ngay sau r=16: cải thiện perplexity per additional parameter giảm gần 5× khi chuyển từ khoảng r=8→16 sang r=16→64.

Thời gian training không phân biệt được giữa các ranks (~4.1–4.6 min) vì bottleneck là data I/O, không phải LoRA matrix computation. Điều này đặc biệt đúng với dataset nhỏ (200 samples).

**Recommendation cho production**: Với dataset 200 samples và task general instruction-following tiếng Việt, **r=16 là lựa chọn tốt nhất**. r=8 tiết kiệm params nhưng perplexity kém hơn đáng kể; r=64 cải thiện perplexity không đủ để bù lại việc tốn thêm 3× VRAM và 4× checkpoint size. Nếu dataset lớn hơn (≥2000 samples) và domain-specific knowledge cần được học sâu, r=32 hoặc r=64 mới có thêm giá trị. Ngoài ra, qualitative evaluation cho thấy fine-tuning trên general dataset không giúp model học được domain knowledge mới (ví dụ LoRA definition) — nhất quán với nguyên tắc "fine-tune for style, RAG for knowledge".

---

## 6. What I Learned

- **Fine-tuning không thay thế knowledge**: Example 4 chứng minh trực tiếp — fine-tuned model hallucinate sai tên LoRA vì dataset training không có nội dung đó. RAG mới là giải pháp cho knowledge gaps, fine-tuning chỉ cải thiện format và style.
- **Rank selection cần đánh giá đồng thời nhiều chiều**: Nhìn riêng perplexity, r=64 "tốt nhất". Nhưng khi xét VRAM, checkpoint size, và ROI per parameter, r=16 là practical choice cho dataset nhỏ.
- **Dataset quality quan trọng hơn rank**: Với 200 samples, sự khác biệt perplexity giữa các ranks nhỏ (0.37 giữa r=8 và r=64). Tăng dataset lên 1000+ samples chất lượng cao có thể cải thiện nhiều hơn so với tăng rank.
