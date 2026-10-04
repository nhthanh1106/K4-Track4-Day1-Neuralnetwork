# Báo cáo Lab Day 1 — MSSV 2A202602813

## 1. Thiết lập

- Dữ liệu Covertype, 54 đặc trưng, 7 lớp; dùng đúng split_metadata.csv: 464,809 train / 116,203 eval. Validation là 20% train, phân tầng, seed 42: 371,847 train-fit / 92,962 validation.
- Chuẩn hóa 10 đặc trưng liên tục bằng thống kê train-fit; giữ 44 cột one-hot nguyên trạng.
- Môi trường: NVIDIA GeForce RTX 4050 Laptop GPU; PyTorch 2.7.1+cu118.
- Baseline baseline_s42: M-base 54→256→128→7, 47,879 tham số; CE, sgd_momentum, lr=0.05, batch=512, 20 epoch, khởi tạo he.
- Accuracy của chiến lược đoán lớp đa số trên train-fit: 0.4876. Cấu hình cuối được chọn bằng validation; eval chỉ dùng để chấm sau khi đã khóa lựa chọn.
- Chủ đề đã thử: loss, optimizer, hyper-parameter, dropout, gradient clipping, mixed precision, initialization.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47,879 / (B, 7) |
| Loss bước 0 baseline_s42 | 2.3776 (ln 7 = 1.9459; gap=+0.4317) |
| Quá khớp 20 mẫu | accuracy 100.0%, 11 bước, loss cuối 0.0107 |
| Gradient | PASS: mọi tham số có gradient hữu hạn, khác 0 |
| Baseline seed / val accuracy TB ± σ | 2 / 0.8991 ± 0.0028 |
| Baseline val macro-F1 TB ± σ | 0.8351 ± 0.0013 |

Ngưỡng nhiễu dùng để đọc các khác biệt trên validation: 2σ = 0.0025. Chỉ có hai seed baseline nên đây là ước lượng thô.
Loss bước 0 cao hơn ln 7 khoảng +0.4317, nên chưa gần phân phối logit đều. Shape, gradient và phép thử overfit 20 mẫu đều đạt; ghi nhận đây là giới hạn của khởi tạo hiện tại thay vì xem con số loss như bằng chứng pipeline hỏng.

## 3. Kết quả theo chủ đề

Điểm trong bảng là macro-F1 tại epoch có val loss thấp nhất. So sánh dùng cùng baseline_s42; kết luận cải thiện chỉ khi chênh lệch vượt 2σ.

| exp_id | nhóm | cấu hình chính | val macro-F1 | Δ vs baseline_s42 | vượt 2σ? | diverged? |
|---|---|---|---:|---:|---|---|
| baseline_s42 | baseline | sgd_momentum, ce, lr=0.05 | 0.8342 | 0.0000 | không | no |
| baseline_s123 | baseline | sgd_momentum, ce, lr=0.05 | 0.8360 | 0.0018 | không | no |
| loss_mse | loss | sgd_momentum, mse, lr=0.05 | 0.7029 | -0.1313 | có | no |
| adam_1e3 | optimizer | adam, ce, lr=0.001 | 0.8521 | 0.0179 | có | no |
| adam_3e4 | optimizer | adam, ce, lr=0.0003 | 0.7917 | -0.0425 | có | no |
| adam_3e3 | optimizer | adam, ce, lr=0.003 | 0.8757 | 0.0415 | có | no |
| sgd_1e3 | optimizer | sgd_momentum, ce, lr=0.001 | 0.5936 | -0.2406 | có | no |
| sgd_1e2 | optimizer | sgd_momentum, ce, lr=0.01 | 0.7449 | -0.0893 | có | no |
| sgd_1e1 | optimizer | sgd_momentum, ce, lr=0.1 | 0.8453 | 0.0112 | có | no |
| width_512_256 | hparam | sgd_momentum, ce, lr=0.05 | 0.8637 | 0.0295 | có | no |
| dropout_03 | dropout | sgd_momentum, ce, lr=0.05 | 0.7606 | -0.0735 | có | no |
| baseline_clip1 | clipping | sgd_momentum, ce, lr=0.05 | 0.8399 | 0.0057 | có | no |
| highlr_no_clip | clipping | sgd_momentum, ce, lr=1.0 | 0.7544 | -0.0798 | có | no |
| highlr_clip1 | clipping | sgd_momentum, ce, lr=1.0 | 0.7882 | -0.0460 | có | no |
| init_xavier | init | sgd_momentum, ce, lr=0.05 | 0.8489 | 0.0147 | có | no |
| amp_fp16 | amp | sgd_momentum, ce, lr=0.05 | 0.8375 | 0.0034 | có | no |

### 3.1 Hàm mất mát — CE và MSE

Dự đoán: CE phù hợp hơn cho phân loại vì tối ưu log-likelihood của lớp đúng; MSE trên softmax và one-hot có gradient khác và có thể bão hòa. Không so độ lớn CE loss với MSE loss.

- loss_mse: val macro-F1 0.7029, Δ -0.1313 so với baseline_s42; vượt 2σ=0.0025. Đây là một lần chạy với MSE, so với CE baseline. Đường cong: [figures/loss_mse.png](figures/loss_mse.png).

### 3.2 Bộ tối ưu hóa

| exp_id | optimizer | lr | val macro-F1 | epoch val loss thấp nhất |
|---|---|---:|---:|---:|
| baseline_s42 | sgd_momentum | 0.05 | 0.8342 | 18 |
| sgd_1e3 | sgd_momentum | 0.001 | 0.5936 | 20 |
| sgd_1e2 | sgd_momentum | 0.01 | 0.7449 | 20 |
| sgd_1e1 | sgd_momentum | 0.1 | 0.8453 | 18 |
| adam_3e4 | adam | 0.0003 | 0.7917 | 20 |
| adam_1e3 | adam | 0.001 | 0.8521 | 20 |
| adam_3e3 | adam | 0.003 | 0.8757 | 20 |

Lưới đã thử: SGD+momentum tại 0.001/0.01/0.05/0.1 và Adam tại 0.0003/0.001/0.003. Tại lr chung 0.001, Adam=0.8521 so với SGD+momentum=0.5936; Δ Adam−SGD=+0.2585 (vượt ngưỡng nhiễu thô 2σ=0.0025). Khi chọn lr tốt nhất trong mỗi lưới, SGD+momentum chọn sgd_1e1 (lr=0.1, F1=0.8453), Adam chọn adam_3e3 (lr=0.003, F1=0.8757); Δ Adam−SGD=+0.0304. Đây là lưới nhỏ, một seed mỗi cấu hình, không chứng minh tối ưu toàn cục. Adam chuẩn hóa bước theo moment bậc nhất/hai nên lr hữu hiệu khác SGD+momentum; vì vậy cùng lr số học không đồng nghĩa cùng độ lớn cập nhật. Đồ thị nhóm: [compare_optimizer.png](figures/compare_optimizer.png).

### 3.3 Hyper-parameter — độ rộng

width_512_256 đổi hidden từ 256-128 thành 512-256; số tham số tăng từ 47,879 lên 161,287. Thời gian trung bình 1.45 giây/epoch. Độ rộng và số tham số cùng đổi, nên không tách được riêng ảnh hưởng của từng yếu tố. Đồ thị: [figures/width_512_256.png](figures/width_512_256.png).

### 3.4 Dropout

Final val loss − train loss: dropout_03=0.0050, baseline=0.0178. Dropout 0.3 chỉ được ủng hộ nếu giảm gap mà vẫn giữ/tăng macro-F1; áp dụng kết luận chỉ cho cấu hình đã chạy.
- dropout_03: val macro-F1 0.7606, Δ -0.0735 so với baseline_s42; vượt 2σ=0.0025. Đồ thị train/val cho thấy ảnh hưởng lên gap và tốc độ học. Đường cong: [figures/dropout_03.png](figures/dropout_03.png).

### 3.5 Gradient clipping

Ở lr baseline=0.05, baseline_clip1 có norm lớn nhất 3.34 và clip kích hoạt trên 14.0% minibatch. Cặp cùng seed với baseline_s42: val macro-F1 0.8342 → 0.8399 (Δ=+0.0057, vượt 2σ=0.0025 (ước lượng thô)). Ở lr=1.0, highlr_clip1 có norm lớn nhất 5.21 và clip kích hoạt trên 0.1% minibatch. Cặp lr=1.0: highlr_no_clip macro-F1=0.7544, highlr_clip1=0.7882 (Δ=+0.0338); grad_norm trung bình lớn nhất theo epoch lần lượt 0.35 và 0.31. Vì clip thực sự kích hoạt, cặp cùng lr cho phép quan sát tác động trong run này. Clip cũng kích hoạt ở lr baseline trong cặp có seed khớp; kết quả vẫn chỉ là một seed cho biến thể clipping. Ở lr cao, clip tác động lên rất ít minibatch, nên không suy rộng thành cách xử lý mọi kiểu mất ổn định. Đồ thị: [figures/compare_clipping.png](figures/compare_clipping.png).

### 3.6 Mixed precision

amp_fp16: 3.16 giây/epoch, peak VRAM 159.8 MB, val macro-F1=0.8375 (Δ=+0.0034 so baseline); 1.76× thời gian baseline (chậm hơn); peak VRAM FP32=159.8 MB, FP16=159.8 MB; hoàn tất 20 epoch. FP16 autocast dùng GradScaler để hạn chế underflow/overflow; trọng số vẫn FP32. Không đo BF16; kết luận chỉ áp dụng cho GPU và batch đã chạy. Đồ thị: [figures/amp_fp16.png](figures/amp_fp16.png).

### 3.7 Khởi tạo tham số

- baseline_s42 (he): activation std sau các lớp ReLU = [0.4140, 0.3787], step-0 loss=2.3776; [đồ thị](figures/baseline_s42.png).
- init_xavier (xavier): activation std sau các lớp ReLU = [0.1728, 0.1291], step-0 loss=2.0429; [đồ thị](figures/init_xavier.png).

init_xavier val macro-F1=0.8489. He dùng phương sai xấp xỉ 2/n_in cho ReLU; Xavier dùng 2/(n_in+n_out). Thí nghiệm không chạy zeros hoặc normal; phần giải thích zeros bên dưới là lý thuyết.

## 4. Đánh giá cuối trên eval

| cấu hình | seed | val macro-F1 | eval macro-F1 | eval accuracy |
|---|---:|---:|---:|---:|
| Baseline baseline_s42 | 42 | 0.8342 | 0.8355 | 0.9000 |
| Cuối cùng adam_3e3 | 42 | 0.8757 | 0.8810 | 0.9168 |

Cấu hình cuối chọn theo validation: adam_3e3 (adam, ce, lr=0.003, hidden=(256, 128), init=he); dùng epoch 20 có val loss thấp nhất.
Δ eval macro-F1 cuối − baseline=+0.0455. Mỗi cấu hình có một lần chấm eval; không có nhiều seed eval để ước lượng nhiễu eval và không dùng kết quả eval để đổi cấu hình.
Chênh lệch val–eval macro-F1 của cấu hình cuối=+0.0053.

### 4.1 Phân tích lỗi theo lớp

| lớp | support | precision | recall | F1 |
|---:|---:|---:|---:|---:|
| 0 | 42,368 | 0.9008 | 0.9263 | 0.9134 |
| 1 | 56,661 | 0.9359 | 0.9191 | 0.9274 |
| 2 | 7,151 | 0.9213 | 0.9150 | 0.9181 |
| 3 | 549 | 0.8373 | 0.8342 | 0.8358 |
| 4 | 1,899 | 0.7910 | 0.7973 | 0.7941 |
| 5 | 3,473 | 0.8526 | 0.8477 | 0.8501 |
| 6 | 4,102 | 0.9424 | 0.9137 | 0.9278 |

Lớp có F1 thấp nhất là **4** (F1=0.7941, support=1,899); 305 mẫu thật lớp 4 bị dự đoán thành lớp **1** (16.1%). Trong 10 đặc trưng liên tục đã chuẩn hóa, ba chênh lệch trung bình nhỏ nhất là Vertical_Distance_To_Hydrology (|Δ mean|=0.123451), Aspect (|Δ mean|=0.149342), Horizontal_Distance_To_Hydrology (|Δ mean|=0.302818). So sánh trung bình không chứng minh nguyên nhân hoặc độ chồng lấp toàn bộ phân phối. Support nhỏ làm F1 nhạy với một số lỗi; mất cân bằng có số liệu hỗ trợ, còn giải thích địa hình cụ thể vẫn là giả thuyết.

Ma trận nhầm lẫn (hàng = nhãn thật, cột = nhãn dự đoán):

| thật / dự đoán | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 39247 | 2850 | 1 | 0 | 57 | 8 | 205 |
| 1 | 3961 | 52077 | 175 | 1 | 308 | 116 | 23 |
| 2 | 1 | 172 | 6543 | 66 | 31 | 338 | 0 |
| 3 | 0 | 0 | 51 | 458 | 0 | 40 | 0 |
| 4 | 51 | 305 | 21 | 0 | 1514 | 7 | 1 |
| 5 | 7 | 185 | 311 | 22 | 4 | 2944 | 0 |
| 6 | 301 | 53 | 0 | 0 | 0 | 0 | 3748 |

## 5. Trả lời các câu hỏi dẫn dắt

1. **Optimizer:** lưới nhỏ tốt nhất là sgd_1e1 cho SGD+momentum và adam_3e3 cho Adam; tại lr chung 0.001, Adam−SGD Δ=+0.2585. Kết quả khác nhau giữa tuned grid và common-lr comparison; chưa chứng minh lr tối ưu toàn cục.
2. **Dropout:** dropout_03 giảm gap val−train từ 0.0178 xuống 0.0050 nhưng val macro-F1 0.7606 thấp hơn baseline 0.8342. Gap baseline vốn nhỏ; q=0.3 gây regularization quá mạnh trong run này. Chỉ nên giữ dropout khi validation cho thấy giảm overfit mà không làm giảm điểm.
3. **Clipping:** ở c=1, norm baseline cao nhất vượt 1 và khoảng 14.0% minibatch bị clip; paired val F1 Δ=+0.0057. Ở lr=1.0, highlr_clip1 chỉ kích hoạt 0.1% minibatch. Norm cần đo trước clip; clipping chỉ thay đổi update ở minibatch vượt ngưỡng.
4. **Mixed precision:** FP16 hoàn tất 20 epoch; thời gian là 1.76× thời gian baseline (chậm hơn); peak VRAM FP32=159.8 MB, FP16=159.8 MB. Val macro-F1 Δ=+0.0034 so baseline. BF16 chưa được thử; phép đo này không ủng hộ FP16 nếu mục tiêu chỉ là tăng tốc.
5. **Khởi tạo:** He giữ phương sai kích hoạt phù hợp ReLU hơn theo công thức. Khởi tạo zeros làm các nơ-ron cùng lớp đối xứng và nhận gradient giống nhau nên không học vai trò khác nhau; đây là giải thích lý thuyết vì zeros không được chạy.
6. **Loss phẳng sau 2,000 bước:** kiểm tra (i) shape/nhãn 0..6 và loss bước 0 so với ln 7; (ii) overfit một lô 20 mẫu để kiểm tra pipeline/khả năng biểu diễn; (iii) gradient hữu hạn, khác 0 ở mọi tham số và grad_norm trước clip. Sau đó kiểm tra lr, chuẩn hóa chỉ trên train, dropout ở eval và mapping nhãn.

## 6. Hạn chế và điều bất ngờ

- Chỉ có 2 seed baseline nên 2σ là ước lượng thô; mỗi biến thể optimizer chỉ có một seed. Lưới lr của mỗi optimizer có ba hoặc bốn điểm, không bảo đảm tìm cực trị.
- Loss bước 0 của He không gần ln 7 dù các phép thử gradient và overfit mẫu nhỏ đạt; cần xem đây là cảnh báo về độ lớn logit ban đầu nếu đổi kiến trúc hoặc chuẩn hóa.
- Cùng số epoch không đảm bảo cùng tốc độ hội tụ. Độ rộng đổi cả số tham số lẫn chi phí tính toán; FP16 chỉ đo trên một GPU và không có đối chứng BF16.
- Phân tích centroid đặc trưng của lớp khó mô tả trung bình, không chứng minh quan hệ nhân quả.

## 7. Phụ lục

- Gồm REPORT.md, experiments.xlsx, predictions_eval.csv, eval_result.json, figures/, results/, code/.
- Tổng thời gian train: 9.9 phút.
- Baseline evaluation chi tiết: results/evaluation/baseline_eval_result.json; điểm cuối chính thức: eval_result.json.
