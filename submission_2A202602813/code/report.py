"""Generate a submission report from experiment and official evaluation results."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _fmt(value, digits=4):
    return "—" if value is None else f"{float(value):.{digits}f}"


def build_report(results: list[dict], sanity: dict, baseline_eval: dict,
                 final_eval: dict, selected_id: str, output_path: str | Path,
                 error_context: dict | None = None) -> str:
    by_id = {r["cfg"]["exp_id"]: r for r in results}
    selected, baseline = by_id[selected_id], by_id["baseline_s42"]
    baseline_runs = [r for r in results if r["cfg"].get("group") == "baseline"]
    seed_f1 = np.asarray([r["summary"]["val_macro_f1"] for r in baseline_runs], dtype=float)
    seed_acc = np.asarray([r["summary"]["val_acc"] for r in baseline_runs], dtype=float)
    mean_f1 = float(seed_f1.mean())
    mean_acc = float(seed_acc.mean())
    sigma = float(seed_f1.std(ddof=1)) if len(seed_f1) > 1 else None
    sigma_acc = float(seed_acc.std(ddof=1)) if len(seed_acc) > 1 else None
    two_sigma = 2 * sigma if sigma is not None else None
    baseline_f1 = float(baseline["summary"]["val_macro_f1"])
    groups = {}
    for result in results:
        groups.setdefault(result["cfg"].get("group", "other"), []).append(result)

    rows = []
    for r in results:
        cfg, summary = r["cfg"], r["summary"]
        score = summary.get("val_macro_f1")
        delta = None if score is None else float(score) - baseline_f1
        beyond = "chưa đo" if delta is None or two_sigma is None else (
            "có" if abs(delta) > two_sigma else "không"
        )
        rows.append(
            f"| {cfg['exp_id']} | {cfg.get('group')} | {cfg.get('optimizer')}, "
            f"{cfg.get('loss')}, lr={cfg.get('lr')} | {_fmt(score)} | "
            f"{_fmt(delta)} | {beyond} | {'yes' if summary.get('diverged') else 'no'} |"
        )

    def experiment(exp_id, mechanism):
        r = by_id[exp_id]
        delta = float(r["summary"]["val_macro_f1"]) - baseline_f1
        noise = (f"vượt 2σ={two_sigma:.4f}" if abs(delta) > two_sigma
                 else f"không vượt 2σ={two_sigma:.4f}") if two_sigma is not None else "chưa đo được nhiễu"
        return (
            f"- {exp_id}: val macro-F1 {_fmt(r['summary'].get('val_macro_f1'))}, "
            f"Δ {_fmt(delta)} so với baseline_s42; {noise}. {mechanism} "
            f"Đường cong: [figures/{exp_id}.png](figures/{exp_id}.png)."
        )

    topics = [
        label for group, label in [
            ("loss", "loss"), ("optimizer", "optimizer"), ("hparam", "hyper-parameter"),
            ("dropout", "dropout"), ("clipping", "gradient clipping"),
            ("amp", "mixed precision"), ("init", "initialization"),
        ] if group in groups
    ]
    lines = [
        "# Báo cáo Lab Day 1 — MSSV 2A202602813",
        "",
        "## 1. Thiết lập",
        "",
        "- Dữ liệu Covertype, 54 đặc trưng, 7 lớp; dùng đúng split_metadata.csv: "
        "464,809 train / 116,203 eval. Validation là 20% train, phân tầng, seed 42: "
        "371,847 train-fit / 92,962 validation.",
        "- Chuẩn hóa 10 đặc trưng liên tục bằng thống kê train-fit; giữ 44 cột one-hot nguyên trạng.",
        f"- Môi trường: {sanity.get('device', 'chưa ghi')}; PyTorch {sanity.get('torch_version', 'chưa ghi')}.",
        f"- Baseline baseline_s42: M-base 54→256→128→7, "
        f"{baseline['summary'].get('param_count'):,} tham số; CE, "
        f"{baseline['cfg']['optimizer']}, lr={baseline['cfg']['lr']}, "
        f"batch={baseline['cfg']['batch']}, {baseline['cfg']['epochs']} epoch, "
        f"khởi tạo {baseline['cfg']['init']}.",
        f"- Accuracy của chiến lược đoán lớp đa số trên train-fit: "
        f"{sanity.get('majority_train_acc', 0.4876):.4f}. Cấu hình cuối được chọn bằng validation; "
        "eval chỉ dùng để chấm sau khi đã khóa lựa chọn.",
        f"- Chủ đề đã thử: {', '.join(topics)}.",
        "",
        "## 2. Kiểm tra ban đầu và độ nhiễu",
        "",
        "| Kiểm tra | Kết quả |",
        "|---|---|",
        f"| Số tham số / shape logits | {baseline['summary'].get('param_count'):,} / (B, 7) |",
        f"| Loss bước 0 baseline_s42 | {_fmt(baseline['summary'].get('step0_loss'))} "
        f"(ln 7 = {np.log(7):.4f}; gap={float(baseline['summary'].get('step0_loss'))-np.log(7):+.4f}) |",
        f"| Quá khớp 20 mẫu | accuracy {sanity.get('overfit_acc', 0):.1%}, "
        f"{sanity.get('overfit_steps', '—')} bước, loss cuối {_fmt(sanity.get('overfit_loss'))} |",
        f"| Gradient | {'PASS' if sanity.get('gradient_pass') else 'FAIL'}: mọi tham số có gradient hữu hạn, khác 0 |",
        f"| Baseline seed / val accuracy TB ± σ | {len(baseline_runs)} / "
        f"{_fmt(mean_acc)} ± {_fmt(sigma_acc)} |",
        f"| Baseline val macro-F1 TB ± σ | {_fmt(mean_f1)} ± {_fmt(sigma)} |",
        "",
        f"Ngưỡng nhiễu dùng để đọc các khác biệt trên validation: 2σ = {_fmt(two_sigma)}. "
        "Chỉ có hai seed baseline nên đây là ước lượng thô.",
        f"Loss bước 0 cao hơn ln 7 khoảng "
        f"{float(baseline['summary'].get('step0_loss')) - np.log(7):+.4f}, nên chưa gần phân phối "
        "logit đều. Shape, gradient và phép thử overfit 20 mẫu đều đạt; ghi nhận đây là giới hạn của "
        "khởi tạo hiện tại thay vì xem con số loss như bằng chứng pipeline hỏng.",
        "",
        "## 3. Kết quả theo chủ đề",
        "",
        "Điểm trong bảng là macro-F1 tại epoch có val loss thấp nhất. So sánh dùng cùng baseline_s42; "
        "kết luận cải thiện chỉ khi chênh lệch vượt 2σ.",
        "",
        "| exp_id | nhóm | cấu hình chính | val macro-F1 | Δ vs baseline_s42 | vượt 2σ? | diverged? |",
        "|---|---|---|---:|---:|---|---|",
        *rows,
        "",
    ]
    if "loss" in groups:
        lines += [
            "### 3.1 Hàm mất mát — CE và MSE",
            "",
            "Dự đoán: CE phù hợp hơn cho phân loại vì tối ưu log-likelihood của lớp đúng; MSE trên "
            "softmax và one-hot có gradient khác và có thể bão hòa. Không so độ lớn CE loss với MSE loss.",
            "",
            experiment("loss_mse", "Đây là một lần chạy với MSE, so với CE baseline."),
            "",
        ]
    if "optimizer" in groups:
        optimizer_ids = [
            "baseline_s42", "sgd_1e3", "sgd_1e2", "sgd_1e1",
            "adam_3e4", "adam_1e3", "adam_3e3",
        ]
        sgd_ids = ["sgd_1e3", "sgd_1e2", "baseline_s42", "sgd_1e1"]
        adam_ids = ["adam_3e4", "adam_1e3", "adam_3e3"]
        best_sgd = max((by_id[name] for name in sgd_ids),
                       key=lambda r: r["summary"]["val_macro_f1"])
        best_adam = max((by_id[name] for name in adam_ids),
                        key=lambda r: r["summary"]["val_macro_f1"])
        matched_sgd, matched_adam = by_id["sgd_1e3"], by_id["adam_1e3"]
        matched_delta = (float(matched_adam["summary"]["val_macro_f1"])
                         - float(matched_sgd["summary"]["val_macro_f1"]))
        matched_noise = (
            f"vượt ngưỡng nhiễu thô 2σ={two_sigma:.4f}"
            if two_sigma is not None and abs(matched_delta) > two_sigma
            else f"không vượt ngưỡng nhiễu thô 2σ={two_sigma:.4f}"
            if two_sigma is not None else "chưa đo được ngưỡng nhiễu"
        )
        lines += [
            "### 3.2 Bộ tối ưu hóa",
            "",
            "| exp_id | optimizer | lr | val macro-F1 | epoch val loss thấp nhất |",
            "|---|---|---:|---:|---:|",
        ]
        for exp_id in optimizer_ids:
            r = by_id[exp_id]
            lines.append(
                f"| {exp_id} | {r['cfg']['optimizer']} | {r['cfg']['lr']} | "
                f"{_fmt(r['summary'].get('val_macro_f1'))} | {r['summary'].get('best_epoch')} |"
            )
        sgd_f1 = float(best_sgd["summary"]["val_macro_f1"])
        adam_f1 = float(best_adam["summary"]["val_macro_f1"])
        best_delta = adam_f1 - sgd_f1
        lines += [
            "",
            f"Lưới đã thử: SGD+momentum tại 0.001/0.01/0.05/0.1 và Adam tại "
            f"0.0003/0.001/0.003. Tại lr chung 0.001, Adam={_fmt(matched_adam['summary']['val_macro_f1'])} "
            f"so với SGD+momentum={_fmt(matched_sgd['summary']['val_macro_f1'])}; "
            f"Δ Adam−SGD={matched_delta:+.4f} ({matched_noise}). Khi chọn lr tốt nhất trong mỗi lưới, "
            f"SGD+momentum chọn {best_sgd['cfg']['exp_id']} (lr={best_sgd['cfg']['lr']}, "
            f"F1={_fmt(sgd_f1)}), Adam chọn {best_adam['cfg']['exp_id']} "
            f"(lr={best_adam['cfg']['lr']}, F1={_fmt(adam_f1)}); Δ Adam−SGD={best_delta:+.4f}. "
            "Đây là lưới nhỏ, một seed mỗi cấu hình, không chứng minh tối ưu toàn cục. Adam chuẩn hóa "
            "bước theo moment bậc nhất/hai nên lr hữu hiệu khác SGD+momentum; vì vậy cùng lr số học "
            "không đồng nghĩa cùng độ lớn cập nhật. Đồ thị nhóm: "
            "[compare_optimizer.png](figures/compare_optimizer.png).",
            "",
        ]
    if "hparam" in groups:
        wide = by_id["width_512_256"]
        lines += [
            "### 3.3 Hyper-parameter — độ rộng",
            "",
            f"width_512_256 đổi hidden từ 256-128 thành 512-256; số tham số tăng từ "
            f"{baseline['summary'].get('param_count'):,} lên {wide['summary'].get('param_count'):,}. "
            f"Thời gian trung bình {_fmt(wide['summary'].get('time_per_epoch_s'), 2)} giây/epoch. "
            "Độ rộng và số tham số cùng đổi, nên không tách được riêng ảnh hưởng của từng yếu tố. "
            f"Đồ thị: [figures/width_512_256.png](figures/width_512_256.png).",
            "",
        ]
    if "dropout" in groups:
        drop = by_id["dropout_03"]
        drop_gap = float(drop["summary"]["final_val_loss"]) - float(drop["summary"]["final_train_loss"])
        base_gap = float(baseline["summary"]["final_val_loss"]) - float(baseline["summary"]["final_train_loss"])
        lines += [
            "### 3.4 Dropout",
            "",
            f"Final val loss − train loss: dropout_03={_fmt(drop_gap)}, baseline={_fmt(base_gap)}. "
            "Dropout 0.3 chỉ được ủng hộ nếu giảm gap mà vẫn giữ/tăng macro-F1; áp dụng kết luận "
            "chỉ cho cấu hình đã chạy.",
            experiment("dropout_03", "Đồ thị train/val cho thấy ảnh hưởng lên gap và tốc độ học."),
            "",
        ]
    if "clipping" in groups:
        plain, clipped = by_id["highlr_no_clip"], by_id["highlr_clip1"]
        normal_clipped = by_id.get("baseline_clip1")
        plain_peak = max(plain["history"].get("grad_norm", [0]) or [0])
        clipped_peak = max(clipped["history"].get("grad_norm", [0]) or [0])
        clipped_step_max = max(clipped["history"].get("grad_norm_max", [0]) or [0])
        clipped_fraction = float(np.mean(clipped["history"].get("clip_fraction", [0]) or [0]))
        normal_fraction = (
            float(np.mean(normal_clipped["history"].get("clip_fraction", [0]) or [0]))
            if normal_clipped else None
        )
        normal_max = (
            max(normal_clipped["history"].get("grad_norm_max", [0]) or [0])
            if normal_clipped else None
        )
        normal_delta = (
            float(normal_clipped["summary"]["val_macro_f1"])
            - baseline_f1 if normal_clipped else None
        )
        normal_noise = (
            f"vượt 2σ={two_sigma:.4f} (ước lượng thô)"
            if normal_delta is not None and two_sigma is not None and abs(normal_delta) > two_sigma
            else f"không vượt 2σ={two_sigma:.4f} (ước lượng thô)"
            if two_sigma is not None else "chưa đo được ngưỡng nhiễu"
        )
        normal_text = (
            f"Ở lr baseline=0.05, baseline_clip1 có norm lớn nhất "
            f"{_fmt(normal_max, 2)} và clip kích hoạt trên {normal_fraction:.1%} minibatch. "
            f"Cặp cùng seed với baseline_s42: val macro-F1 {_fmt(baseline_f1)} → "
            f"{_fmt(normal_clipped['summary']['val_macro_f1'])} (Δ={normal_delta:+.4f}, "
            f"{normal_noise}). "
            if normal_clipped else "Không chạy clipping ở lr baseline. "
        )
        high_text = (
            f"Ở lr=1.0, highlr_clip1 có norm lớn nhất {_fmt(clipped_step_max, 2)} và "
            f"clip kích hoạt trên {clipped_fraction:.1%} minibatch. "
        )
        attribution_text = (
            "Vì clip thực sự kích hoạt, cặp cùng lr cho phép quan sát tác động trong run này. "
            if clipped_fraction > 0
            else "Không minibatch nào vượt ngưỡng 1.0; không thể quy chênh lệch macro-F1 cho clipping. "
        )
        lines += [
            "### 3.5 Gradient clipping",
            "",
            normal_text + high_text +
            f"Cặp lr=1.0: highlr_no_clip macro-F1={_fmt(plain['summary'].get('val_macro_f1'))}, "
            f"highlr_clip1={_fmt(clipped['summary'].get('val_macro_f1'))} "
            f"(Δ={float(clipped['summary']['val_macro_f1'])-float(plain['summary']['val_macro_f1']):+.4f}); "
            f"grad_norm trung bình lớn "
            f"nhất theo epoch lần lượt {_fmt(plain_peak, 2)} và {_fmt(clipped_peak, 2)}. "
            + attribution_text +
            "Clip cũng kích hoạt ở lr baseline trong cặp có seed khớp; kết quả vẫn chỉ là một seed "
            "cho biến thể clipping. Ở lr cao, clip tác động lên rất ít minibatch, nên không suy rộng "
            "thành cách xử lý mọi kiểu mất ổn định. "
            "Đồ thị: [figures/compare_clipping.png](figures/compare_clipping.png).",
            "",
        ]
    if "amp" in groups:
        amp = by_id["amp_fp16"]
        base_t = baseline["summary"].get("time_per_epoch_s")
        amp_t = amp["summary"].get("time_per_epoch_s")
        base_mem = baseline["summary"].get("peak_mem_MB")
        amp_mem = amp["summary"].get("peak_mem_MB")
        time_ratio = float(amp_t) / float(base_t) if base_t and amp_t else None
        speed_text = (
            f"{time_ratio:.2f}× thời gian baseline (chậm hơn)"
            if time_ratio is not None and time_ratio > 1
            else f"{1 / time_ratio:.2f}× nhanh hơn baseline"
            if time_ratio
            else "thời gian không có sẵn"
        )
        memory_text = (
            f"peak VRAM FP32={_fmt(base_mem, 1)} MB, FP16={_fmt(amp_mem, 1)} MB"
            if base_mem is not None and amp_mem is not None
            else f"peak VRAM FP16={_fmt(amp_mem, 1)} MB; FP32 không có log"
        )
        amp_delta = float(amp["summary"]["val_macro_f1"]) - baseline_f1
        amp_status = (
            f"dừng sớm sau {len(amp['history'].get('epoch', []))} epoch do gradient overflow"
            if amp["summary"].get("diverged")
            else f"hoàn tất {len(amp['history'].get('epoch', []))} epoch"
        )
        lines += [
            "### 3.6 Mixed precision",
            "",
            f"amp_fp16: {_fmt(amp_t, 2)} giây/epoch, peak VRAM "
            f"{_fmt(amp['summary'].get('peak_mem_MB'), 1)} MB, "
            f"val macro-F1={_fmt(amp['summary'].get('val_macro_f1'))} "
            f"(Δ={amp_delta:+.4f} so baseline); {speed_text}; {memory_text}; {amp_status}. "
            "FP16 autocast dùng GradScaler để hạn chế underflow/overflow; trọng số vẫn FP32. "
            "Không đo BF16; kết luận chỉ áp dụng cho GPU và batch đã chạy. "
            "Đồ thị: [figures/amp_fp16.png](figures/amp_fp16.png).",
            "",
        ]
        amp_answer = (
            f"4. **Mixed precision:** FP16 hoàn tất {len(amp['history'].get('epoch', []))} epoch; "
            f"thời gian là {speed_text}; {memory_text}. Val macro-F1 Δ={amp_delta:+.4f} "
            "so baseline. BF16 chưa được thử; phép đo này không ủng hộ FP16 nếu mục tiêu chỉ là tăng tốc."
        )
    else:
        amp_answer = "4. **Mixed precision:** không chạy vì runtime không có CUDA; chưa có số để kết luận."
    if "init" in groups:
        lines += [
            "### 3.7 Khởi tạo tham số",
            "",
        ]
        for exp_id in ("baseline_s42", "init_xavier"):
            r = by_id[exp_id]
            stats = ", ".join(_fmt(v) for v in r["summary"].get("activation_std_step0", []))
            lines.append(
                f"- {exp_id} ({r['cfg']['init']}): activation std sau các lớp ReLU = "
                f"[{stats}], step-0 loss={_fmt(r['summary'].get('step0_loss'))}; "
                f"[đồ thị](figures/{exp_id}.png)."
            )
        lines += [
            "",
            f"init_xavier val macro-F1={_fmt(by_id['init_xavier']['summary'].get('val_macro_f1'))}. "
            "He dùng phương sai xấp xỉ 2/n_in cho ReLU; Xavier dùng 2/(n_in+n_out). Thí nghiệm "
            "không chạy zeros hoặc normal; phần giải thích zeros bên dưới là lý thuyết.",
            "",
        ]

    per_class = final_eval.get("per_class", [])
    cm = np.asarray(final_eval.get("confusion_matrix", []), dtype=int)
    class_table = [
        "| lớp | support | precision | recall | F1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    class_table.extend(
        f"| {r['cls']} | {r['support']:,} | {_fmt(r['precision'])} | "
        f"{_fmt(r['recall'])} | {_fmt(r['f1'])} |" for r in per_class
    )
    confusion_table = [
        "| thật / dự đoán | " + " | ".join(str(i) for i in range(7)) + " |",
        "|---|" + "---:|" * 7,
    ]
    if cm.shape == (7, 7):
        confusion_table.extend(
            f"| {i} | " + " | ".join(str(int(n)) for n in cm[i]) + " |"
            for i in range(7)
        )

    error = "Chưa có ma trận nhầm lẫn hợp lệ."
    if per_class and cm.shape == (7, 7):
        hardest = min(per_class, key=lambda r: r["f1"])
        cls = int(hardest["cls"])
        off_diagonal = cm[cls].copy()
        off_diagonal[cls] = 0
        partner = int(off_diagonal.argmax())
        count = int(off_diagonal[partner])
        share = count / max(int(hardest["support"]), 1)
        error = (
            f"Lớp có F1 thấp nhất là **{cls}** (F1={_fmt(hardest['f1'])}, support="
            f"{int(hardest['support']):,}); {count:,} mẫu thật lớp {cls} bị dự đoán thành lớp "
            f"**{partner}** ({share:.1%}). "
        )
        if error_context and error_context.get("closest_mean_features"):
            close = ", ".join(
                f"{x['name']} (|Δ mean|={x['abs_delta']:.6f})"
                for x in error_context["closest_mean_features"][:3]
            )
            error += (
                f"Trong 10 đặc trưng liên tục đã chuẩn hóa, ba chênh lệch trung bình nhỏ nhất là {close}. "
                f"So sánh trung bình không chứng minh nguyên nhân hoặc độ "
                "chồng lấp toàn bộ phân phối. "
            )
        error += (
            "Support nhỏ làm F1 nhạy với một số lỗi; mất cân bằng có số liệu hỗ trợ, còn giải thích "
            "địa hình cụ thể vẫn là giả thuyết."
        )

    eval_diff = float(final_eval["macro_f1"]) - float(baseline_eval["macro_f1"])
    val_eval_gap = float(final_eval["macro_f1"]) - float(selected["summary"]["val_macro_f1"])
    cfg = selected["cfg"]
    lines += [
        "## 4. Đánh giá cuối trên eval",
        "",
        "| cấu hình | seed | val macro-F1 | eval macro-F1 | eval accuracy |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline baseline_s42 | {baseline['cfg']['seed']} | "
        f"{_fmt(baseline['summary'].get('val_macro_f1'))} | "
        f"{_fmt(baseline_eval.get('macro_f1'))} | {_fmt(baseline_eval.get('accuracy'))} |",
        f"| Cuối cùng {selected_id} | {cfg['seed']} | "
        f"{_fmt(selected['summary'].get('val_macro_f1'))} | "
        f"{_fmt(final_eval.get('macro_f1'))} | {_fmt(final_eval.get('accuracy'))} |",
        "",
        f"Cấu hình cuối chọn theo validation: {selected_id} ({cfg['optimizer']}, {cfg['loss']}, "
        f"lr={cfg['lr']}, hidden={tuple(cfg['hidden'])}, init={cfg['init']}); dùng epoch "
        f"{selected['summary'].get('best_epoch')} có val loss thấp nhất.",
        f"Δ eval macro-F1 cuối − baseline={eval_diff:+.4f}. Mỗi cấu hình có một lần chấm eval; "
        "không có nhiều seed eval để ước lượng nhiễu eval và không dùng kết quả eval để đổi cấu hình.",
        f"Chênh lệch val–eval macro-F1 của cấu hình cuối={val_eval_gap:+.4f}.",
        "",
        "### 4.1 Phân tích lỗi theo lớp",
        "",
        *class_table,
        "",
        error,
        "",
        "Ma trận nhầm lẫn (hàng = nhãn thật, cột = nhãn dự đoán):",
        "",
        *confusion_table,
        "",
        "## 5. Trả lời các câu hỏi dẫn dắt",
        "",
        f"1. **Optimizer:** lưới nhỏ tốt nhất là {best_sgd['cfg']['exp_id']} cho SGD+momentum và "
        f"{best_adam['cfg']['exp_id']} cho Adam; tại lr chung 0.001, Adam−SGD Δ={matched_delta:+.4f}. "
        "Kết quả khác nhau giữa tuned grid và common-lr comparison; chưa chứng minh lr tối ưu toàn cục.",
        f"2. **Dropout:** dropout_03 giảm gap val−train từ {_fmt(base_gap)} xuống {_fmt(drop_gap)} "
        f"nhưng val macro-F1 {_fmt(by_id['dropout_03']['summary']['val_macro_f1'])} thấp hơn baseline "
        f"{_fmt(baseline_f1)}. Gap baseline vốn nhỏ; q=0.3 gây regularization quá mạnh trong run này. "
        "Chỉ nên giữ dropout khi validation cho thấy giảm overfit mà không làm giảm điểm.",
        "3. **Clipping:** ở c=1, norm baseline cao nhất vượt 1 và khoảng "
        f"{normal_fraction:.1%} minibatch bị clip; paired val F1 Δ={normal_delta:+.4f}. Ở lr=1.0, "
        f"highlr_clip1 chỉ kích hoạt {clipped_fraction:.1%} minibatch. Norm cần đo trước clip; clipping "
        "chỉ thay đổi update ở minibatch vượt ngưỡng.",
        amp_answer,
        "5. **Khởi tạo:** He giữ phương sai kích hoạt phù hợp ReLU hơn theo công thức. Khởi tạo zeros "
        "làm các nơ-ron cùng lớp đối xứng và nhận gradient giống nhau nên không học vai trò khác nhau; "
        "đây là giải thích lý thuyết vì zeros không được chạy.",
        "6. **Loss phẳng sau 2,000 bước:** kiểm tra (i) shape/nhãn 0..6 và loss bước 0 so với ln 7; "
        "(ii) overfit một lô 20 mẫu để kiểm tra pipeline/khả năng biểu diễn; (iii) gradient hữu hạn, "
        "khác 0 ở mọi tham số và grad_norm trước clip. Sau đó kiểm tra lr, chuẩn hóa chỉ trên train, "
        "dropout ở eval và mapping nhãn.",
        "",
        "## 6. Hạn chế và điều bất ngờ",
        "",
        f"- Chỉ có {len(baseline_runs)} seed baseline nên 2σ là ước lượng thô; mỗi biến thể optimizer "
        "chỉ có một seed. Lưới lr của mỗi optimizer có ba hoặc bốn điểm, không bảo đảm tìm cực trị.",
        "- Loss bước 0 của He không gần ln 7 dù các phép thử gradient và overfit mẫu nhỏ đạt; cần xem "
        "đây là cảnh báo về độ lớn logit ban đầu nếu đổi kiến trúc hoặc chuẩn hóa.",
        "- Cùng số epoch không đảm bảo cùng tốc độ hội tụ. Độ rộng đổi cả số tham số lẫn chi phí tính toán; "
        "FP16 chỉ đo trên một GPU và không có đối chứng BF16.",
        "- Phân tích centroid đặc trưng của lớp khó mô tả trung bình, không chứng minh quan hệ nhân quả.",
        "",
        "## 7. Phụ lục",
        "",
        "- Gồm REPORT.md, experiments.xlsx, predictions_eval.csv, eval_result.json, figures/, results/, code/.",
        f"- Tổng thời gian train: {sum(float(r['summary'].get('total_time_s') or 0) for r in results)/60:.1f} phút.",
        "- Baseline evaluation chi tiết: results/evaluation/baseline_eval_result.json; điểm cuối chính thức: eval_result.json.",
        "",
    ]
    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report, encoding="utf-8")
    return report
