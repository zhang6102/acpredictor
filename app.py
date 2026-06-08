import streamlit as st
import torch
import esm
import joblib
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from functools import lru_cache

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="🧬 抗癌肽智能预测系统",
    page_icon="🧬",
    layout="wide"
)

# ==================== 自定义 CSS（优化点1：预测结果卡片 + 彩色样式） ====================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1, h2, h3 { color: #1a202c; font-weight: 600; }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.3);
    }
    .result-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .result-card.positive {
        background: #d1e7dd;
        border-left: 6px solid #198754;
    }
    .result-card.negative {
        background: #f8d7da;
        border-left: 6px solid #dc3545;
    }
    .result-card h3 { margin: 0; }
    .result-card p { margin: 5px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ==================== 模型配置 ====================
MODEL_DIR = "saved_model"
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
MODEL_PATH = os.path.join(MODEL_DIR, "best_xgb_model.pkl")

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
            if "metrics" in config:
                DEFAULT_METRICS.update(config["metrics"])
    except Exception as e:
        st.sidebar.warning(f"配置加载失败，使用默认阈值 0.5")

# ==================== 加载模型（缓存） ====================
@st.cache_resource
def load_esm_model():
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
    model.eval()
    return model, alphabet

@st.cache_resource
def load_xgb_model():
    try:
        model = joblib.load(MODEL_PATH)
        if hasattr(model, 'use_label_encoder'):
            delattr(model, 'use_label_encoder')
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        st.stop()

esm_model, alphabet = load_esm_model()
batch_converter = alphabet.get_batch_converter()
xgb_model = load_xgb_model()

# ==================== 特征提取（带缓存） ====================
@lru_cache(maxsize=500)
def extract_esm2_feature_cached(seq):
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

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 🧬 ACPredictor")
    st.markdown("---")
    st.markdown("**模型信息**")
    st.markdown(f"- **特征提取:** ESM-2 t6 8M")
    st.markdown(f"- **分类器:** XGBoost")
    st.markdown(f"- **最佳阈值:** `{BEST_THRESHOLD:.4f}`")
    st.markdown(f"- **AUC:** {DEFAULT_METRICS['auc']:.4f}")
    st.markdown(f"- **MCC:** {DEFAULT_METRICS['mcc']:.4f}")
    st.markdown(f"- **F1:** {DEFAULT_METRICS['f1']:.4f}")
    st.markdown(f"- **Recall:** {DEFAULT_METRICS['recall']:.4f}")
    st.markdown(f"- **Precision:** {DEFAULT_METRICS['precision']:.4f}")
    st.markdown("---")
    st.caption("✨ 基于 ESM-2 + XGBoost，仅供学术研究")

    # ==================== 优化点2：预测历史记录 ====================
    if "history" not in st.session_state:
        st.session_state.history = []
    
    with st.expander("📋 最近预测记录"):
        if not st.session_state.history:
            st.write("暂无记录")
        else:
            for item in st.session_state.history[-10:][::-1]:
                st.write(
                    f"{item['time']} | "
                    f"{item['sequence'][:20]}{'...' if len(item['sequence']) > 20 else ''} → "
                    f"**{item['pred']}** (prob: {item['prob']:.4f})"
                )

# ==================== 主界面 ====================
st.title("🧬 抗癌肽智能预测系统")
st.markdown("基于 ESM-2 预训练模型 + XGBoost 集成")

tab1, tab2 = st.tabs(["🔤 单序列预测", "📂 批量预测 (FASTA)"])

# ---------- Tab 1: 单序列 ----------
with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        sequence_input = st.text_area(
            "输入氨基酸序列",
            placeholder="请输入大写单字母氨基酸序列 (长度 5-100)",
            height=120
        )
        
        if st.button("📋 填入示例序列"):
            sequence_input = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
            st.rerun()
        
        threshold_slider = st.slider(
            "预测阈值（可调）",
            min_value=0.0,
            max_value=1.0,
            value=BEST_THRESHOLD,
            step=0.01,
            help=f"默认使用最佳阈值 {BEST_THRESHOLD:.4f}"
        )
        
        if st.button("🔬 开始预测", type="primary"):
            if not sequence_input:
                st.warning("请输入序列")
            else:
                seq_clean = sequence_input.strip().upper()
                pred, prob = predict_sequence(seq_clean, threshold_slider)
                if isinstance(pred, str) and ("长度" in pred or "字符" in pred or "出错" in pred):
                    st.error(pred)
                else:
                    with col2:
                        st.success("预测完成")
                        # 优化点1：预测结果卡片
                        if pred == "抗癌肽":
                            st.markdown(f"""
                            <div class="result-card positive">
                                <h3 style="color: #0f5132;">🧬 {pred}</h3>
                                <p style="color: #0f5132;">置信度: <strong>{prob:.4f}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card negative">
                                <h3 style="color: #842029;">🧬 {pred}</h3>
                                <p style="color: #842029;">置信度: <strong>{prob:.4f}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        st.caption(f"当前阈值: {threshold_slider:.4f}")
                        
                        # 优化点2：记录历史
                        st.session_state.history.append({
                            "sequence": seq_clean,
                            "pred": pred,
                            "prob": prob,
                            "time": datetime.now().strftime("%H:%M:%S")
                        })
                        if len(st.session_state.history) > 10:
                            st.session_state.history.pop(0)

# ---------- Tab 2: 批量预测 ----------
with tab2:
    uploaded_file = st.file_uploader("上传 FASTA 文件", type=["fasta", "fa", "txt"])
    batch_threshold = st.slider(
        "批量预测阈值 (可调)",
        min_value=0.0,
        max_value=1.0,
        value=BEST_THRESHOLD,
        step=0.01
    )
    
    if uploaded_file and st.button("🚀 开始批量预测", type="primary"):
        content = uploaded_file.read().decode("utf-8")
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
            st.warning("未解析到有效序列")
        else:
            results = []
            progress_bar = st.progress(0)
            # 优化点3：动态状态提示
            status_text = st.empty()
            for i, (seq_id, seq) in enumerate(sequences):
                pred, prob = predict_sequence(seq, batch_threshold)
                results.append({
                    "序列ID": seq_id,
                    "序列片段": seq[:50] + ("..." if len(seq) > 50 else ""),
                    "预测类别": pred,
                    "可信度": f"{prob:.4f}" if prob is not None else "N/A"
                })
                progress_bar.progress((i + 1) / len(sequences))
                status_text.text(f"正在预测第 {i+1}/{len(sequences)} 条序列...")
            status_text.empty()
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            acp_count = sum(1 for r in results if r["预测类别"] == "抗癌肽")
            total = len(results)
            st.success(f"✅ 共预测 {total} 条序列 | 抗癌肽: {acp_count} 条 | 非抗癌肽: {total - acp_count} 条")
            
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 下载预测结果 (CSV)",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv"
            )
