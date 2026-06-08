import gradio as gr
import torch
import esm
import joblib
import numpy as np
import pandas as pd
import json
import os
from functools import lru_cache

# ==================== 配置 ====================
MODEL_DIR = "saved_model"
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
MODEL_PATH = os.path.join(MODEL_DIR, "best_xgb_model.pkl")

# 加载模型指标（用于展示）
DEFAULT_METRICS = {
    "auc": 0.9446,
    "mcc": 0.8246,
    "f1": 0.9254,
    "recall": 0.9619,
    "precision": 0.8923
}

# 加载阈值
BEST_THRESHOLD = 0.5
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
            BEST_THRESHOLD = config.get("best_threshold", 0.5)
            # 如果配置中有指标，则更新
            if "metrics" in config:
                DEFAULT_METRICS.update(config["metrics"])
    except Exception as e:
        print(f"⚠️ 配置加载失败，使用默认阈值 0.5")

# ==================== 加载模型 ====================
print("正在加载 ESM-2 模型...")
esm_model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

print("正在加载 XGBoost 模型...")
try:
    xgb_model = joblib.load(MODEL_PATH)
    # 清理旧属性
    if hasattr(xgb_model, 'use_label_encoder'):
        delattr(xgb_model, 'use_label_encoder')
    print("✅ XGBoost 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# ==================== 特征提取（带缓存） ====================
@lru_cache(maxsize=1000)
def extract_esm2_feature_cached(seq):
    """缓存特征提取结果，避免重复计算"""
    data = [("protein", seq)]
    _, _, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6])
        token_reps = results["representations"][6]
    seq_emb = token_reps[0, 1: len(seq) + 1].max(dim=0)[0].cpu().numpy()
    return seq_emb

def extract_esm2_feature(seq):
    return extract_esm2_feature_cached(seq)

# ==================== 预测函数 ====================
def predict_sequence(seq, threshold):
    if len(seq) < 5 or len(seq) > 100:
        return "长度不符", None
    if not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq):
        return "含非法字符", None
    try:
        feature = extract_esm2_feature(seq).reshape(1, -1)
        prob = xgb_model.predict_proba(feature)[0][1]
        pred = "抗癌肽" if prob >= threshold else "非抗癌肽"
        return pred, prob
    except Exception as e:
        return f"预测出错: {str(e)}", None

# ==================== Gradio 界面 ====================
custom_css = """
<style>
    body { background: linear-gradient(to bottom, #f8f9fa, #e9ecef); }
    .main-card { background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.06); margin-bottom: 20px; }
    h1 { color: #1a202c; font-weight: 700; letter-spacing: -0.01em; }
    textarea { border-radius: 12px; border: 1px solid #e2e8f0; padding: 12px; font-size: 16px; }
    textarea:focus { border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,0.1); }
    .gr-button-primary { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; border-radius: 12px; border: none; padding: 0.5rem 2rem; font-weight: 500; transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(99,102,241,0.2); }
    .gr-button-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99,102,241,0.3); }
    .result-box { background: #f8fafc; border-radius: 16px; padding: 20px; border: 1px solid #e2e8f0; }
    .sidebar { background: white; border-right: 1px solid #e2e8f0; padding: 20px; }
</style>
"""

with gr.Blocks(theme=gr.themes.Soft(), title="🧬 抗癌肽智能预测系统", css=custom_css) as demo:
    # ==================== 侧边栏 ====================
    with gr.Row():
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### 🧬 ACPredictor")
            gr.Markdown("---")
            gr.Markdown("**模型信息**")
            gr.Markdown(f"- **特征提取:** ESM-2 t6 8M")
            gr.Markdown(f"- **分类器:** XGBoost")
            gr.Markdown(f"- **最佳阈值:** `{BEST_THRESHOLD:.4f}`")
            gr.Markdown(f"- **AUC:** {DEFAULT_METRICS['auc']:.4f}")
            gr.Markdown(f"- **MCC:** {DEFAULT_METRICS['mcc']:.4f}")
            gr.Markdown(f"- **F1:** {DEFAULT_METRICS['f1']:.4f}")
            gr.Markdown(f"- **Recall:** {DEFAULT_METRICS['recall']:.4f}")
            gr.Markdown(f"- **Precision:** {DEFAULT_METRICS['precision']:.4f}")
            gr.Markdown("---")
            gr.Markdown("✨ 基于 ESM-2 + XGBoost，仅供学术研究")

        # ==================== 主界面 ====================
        with gr.Column(scale=4):
            gr.Markdown("""
            <div class="main-card">
                <h1 style="margin-bottom: 0.2rem;">🧬 抗癌肽智能预测系统</h1>
                <p style="color: #6b7280; font-size: 1.1rem; margin-top: 0;">基于 ESM-2 预训练模型与 XGBoost 集成</p>
            </div>
            """)

            with gr.Tabs():
                # ---------- Tab 1: 单序列 ----------
                with gr.TabItem("🔤 单序列预测"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            seq_input = gr.Textbox(
                                label="输入氨基酸序列",
                                placeholder="请输入大写单字母氨基酸序列 (长度 5-100)",
                                lines=5
                            )
                            example_btn = gr.Button("📋 填入示例序列")
                            threshold_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=BEST_THRESHOLD,
                                step=0.01,
                                label="预测阈值（可调）",
                                info=f"默认使用最佳阈值 {BEST_THRESHOLD:.4f}"
                            )
                            predict_btn = gr.Button("🔬 开始预测", variant="primary")
                        with gr.Column(scale=2):
                            pred_output = gr.Textbox(label="预测类别", lines=2)
                            prob_output = gr.Textbox(label="置信度 (概率)", lines=2)
                            progress_html = gr.HTML("")

                    # 示例序列
                    def fill_example():
                        return "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
                    example_btn.click(fn=fill_example, outputs=seq_input)

                    # 预测函数（使用可调阈值）
                    def predict_and_display(seq, threshold):
                        seq_clean = seq.strip().upper()
                        pred, prob = predict_sequence(seq_clean, threshold)
                        if isinstance(pred, str) and ("长度" in pred or "字符" in pred or "出错" in pred):
                            return pred, "", ""
                        prob_val = prob if prob is not None else 0.0
                        color = "#6366f1" if pred == "抗癌肽" else "#10b981"
                        bar_html = f"""
                        <div style="background: #f3f4f6; border-radius: 10px; height: 8px; overflow: hidden; margin-top: 12px;">
                            <div style="width: {prob_val * 100:.1f}%; height: 100%; background: linear-gradient(90deg, {color}, #8b5cf6); border-radius: 10px;"></div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.85rem; color: #6b7280;">
                            阈值 = {threshold:.4f}
                        </div>
                        """
                        return pred, f"{prob_val:.4f}", bar_html

                    predict_btn.click(
                        fn=predict_and_display,
                        inputs=[seq_input, threshold_slider],
                        outputs=[pred_output, prob_output, progress_html]
                    )

                # ---------- Tab 2: 批量预测 ----------
                with gr.TabItem("📂 批量预测 (FASTA)"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            file_input = gr.File(label="上传 FASTA 文件", file_types=[".fasta", ".fa", ".txt"])
                            batch_threshold_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=BEST_THRESHOLD,
                                step=0.01,
                                label="批量预测阈值 (可调)"
                            )
                            batch_btn = gr.Button("🚀 开始批量预测", variant="primary")
                        with gr.Column(scale=2):
                            batch_status = gr.Textbox(label="状态", lines=2)
                    batch_output = gr.Dataframe(label="预测结果")
                    batch_stats = gr.Markdown("")  # 用于显示统计信息
                    download_file = gr.File(label="下载预测结果")  # 显示下载文件

                    # 批量预测函数（支持可调阈值）
                    def predict_batch(file_obj, threshold):
                        if file_obj is None:
                            return None, "请上传 FASTA 文件", None, None
                        content = file_obj.read().decode("utf-8")
                        sequences = []
                        current_id = "未知"
                        current_seq = ""
                        for line in content.splitlines():
                            line = line.strip()
                            if line.startswith(">"):
                                if current_seq:
                                    sequences.append((current_id, current_seq))
                                current_id = line[1:] if len(line) > 1 else "未知"
                                current_seq = ""
                            else:
                                current_seq += line.upper()
                        if current_seq:
                            sequences.append((current_id, current_seq))

                        if not sequences:
                            return None, "未解析到有效序列", None, None

                        results = []
                        for seq_id, seq in sequences:
                            pred, prob = predict_sequence(seq, threshold)
                            results.append({
                                "序列ID": seq_id,
                                "序列片段": seq[:50] + ("..." if len(seq) > 50 else ""),
                                "预测类别": pred,
                                "可信度": f"{prob:.4f}" if prob is not None else "N/A"
                            })
                        df = pd.DataFrame(results)
                        # 统计信息
                        total = len(results)
                        acp_count = sum(1 for r in results if r["预测类别"] == "抗癌肽")
                        non_acp_count = total - acp_count
                        stats_text = f"📊 共预测 {total} 条序列 | 抗癌肽: {acp_count} 条 | 非抗癌肽: {non_acp_count} 条"
                        # 生成 CSV 文件
                        csv_str = df.to_csv(index=False)
                        # 保存临时文件供下载
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                        temp_file.write(csv_str.encode())
                        temp_file.close()
                        return df, f"✅ 预测完成! {stats_text}", temp_file.name, stats_text

                    batch_btn.click(
                        fn=predict_batch,
                        inputs=[file_input, batch_threshold_slider],
                        outputs=[batch_output, batch_status, download_file, batch_stats]
                    )

                    # 批处理结果统计（单独的组件用于显示Markdown统计，已在上述函数中返回）

    # ==================== 页脚 ====================
    gr.Markdown("""
    <div style="text-align: center; padding: 20px; color: #6b7280; font-size: 0.9rem; margin-top: 40px;">
        ⚡ 基于 ESM-2 + XGBoost | 仅供学术研究，不构成医疗诊断
    </div>
    """)

# ==================== 启动 ====================
if __name__ == "__main__":
    demo.launch()
