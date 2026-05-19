import streamlit as st
import torch
import esm
import joblib
import numpy as np
import pandas as pd
import json
import os

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="抗癌肽智能预测系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义 CSS（美化部分） ====================
st.markdown("""
<style>
    /* 全局背景 */
    .stApp {
        background: linear-gradient(to bottom, #f8f9fa, #e9ecef);
    }

    /* 卡片样式 */
    .main-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    /* 标题 */
    h1 {
        color: #1a202c;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    /* 输入框 */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 12px;
        font-size: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1);
    }

    /* 按钮 */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99,102,241,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.3);
    }

    /* 结果卡片 */
    .result-box {
        background: #f8fafc;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e2e8f0;
    }

    /* 进度条颜色 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981, #6366f1);
        border-radius: 10px;
    }

    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 模型与配置加载 ====================
MODEL_DIR = "saved_model"
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
MODEL_PATH = os.path.join(MODEL_DIR, "best_xgb_model.pkl")

# 加载阈值
BEST_THRESHOLD = 0.5
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
            BEST_THRESHOLD = config.get("best_threshold", 0.5)
    except Exception as e:
        st.sidebar.warning(f"⚠️ 配置加载失败，使用默认阈值 0.5")


# ==================== 加载 ESM-2 模型 ====================
@st.cache_resource
def load_esm_model():
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
    model.eval()
    return model, alphabet


esm_model, alphabet = load_esm_model()
batch_converter = alphabet.get_batch_converter()


# ==================== 加载 XGBoost 模型 ====================
@st.cache_resource
def load_xgb_model():
    try:
        model = joblib.load(MODEL_PATH)
        # 清理残留属性（防止报错，但不影响功能）
        if hasattr(model, 'use_label_encoder'):
            delattr(model, 'use_label_encoder')
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        st.stop()


xgb_model = load_xgb_model()


# ==================== 特征提取与预测函数 ====================
def extract_esm2_feature(seq):
    data = [("protein", seq)]
    _, _, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6])
        token_reps = results["representations"][6]
    seq_emb = token_reps[0, 1: len(seq) + 1].max(dim=0)[0].cpu().numpy()
    return seq_emb


def predict_sequence(seq):
    if len(seq) < 5 or len(seq) > 100:
        return "长度不符", None
    if not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq):
        return "含非法字符", None
    try:
        feature = extract_esm2_feature(seq).reshape(1, -1)
        prob = xgb_model.predict_proba(feature)[0][1]
        pred = "抗癌肽" if prob >= BEST_THRESHOLD else "非抗癌肽"
        return pred, prob
    except Exception as e:
        return f"预测出错: {str(e)}", None


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 🧬 ACPredictor")
    st.markdown("---")
    st.markdown(f"**模型信息**")
    st.markdown(f"- **特征提取:** ESM-2 t6 8M")
    st.markdown(f"- **分类器:** XGBoost")
    st.markdown(f"- **最佳阈值:** `{BEST_THRESHOLD:.4f}`")
    st.markdown(f"- **状态:** ✅ 已加载")
    st.markdown("---")
    st.caption("✨ 基于 ESM-2 + XGBoost，仅供学术研究")

# ==================== 主界面 ====================
st.markdown("""
<div class="main-card">
    <h1 style="margin-bottom: 0.2rem;">🧬 抗癌肽智能预测系统</h1>
    <p style="color: #6b7280; font-size: 1.1rem; margin-top: 0;">基于 ESM-2 预训练模型与 XGBoost 集成</p>
</div>
""", unsafe_allow_html=True)

# 标签页
tab1, tab2 = st.tabs(["🔤 单序列预测", "📂 批量预测 (FASTA)"])

# ---------- Tab 1: 单序列 ----------
with tab1:
    col_input, col_result = st.columns([3, 2])

    with col_input:
        st.markdown("##### 输入氨基酸序列")
        sequence_input = st.text_area(
            "",
            height=150,
            placeholder="请输入大写单字母氨基酸序列 (长度 5-100)",
            label_visibility="collapsed"
        )

        if st.button("📋 填入示例序列"):
            sequence_input = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
            st.rerun()

        if st.button("🔬 开始预测", type="primary"):
            if not sequence_input:
                st.warning("请输入序列")
            else:
                seq_clean = sequence_input.strip().upper()
                pred, prob = predict_sequence(seq_clean)

                if isinstance(pred, str) and ("长度" in pred or "字符" in pred or "出错" in pred):
                    st.error(pred)
                else:
                    with col_result:
                        st.markdown("##### 预测结果")
                        if pred == "抗癌肽":
                            st.markdown(f"""
                            <div class="result-box" style="border-left: 6px solid #6366f1;">
                                <div style="font-size: 1.5rem; font-weight: 600; color: #1a202c;">{pred}</div>
                                <div style="color: #6b7280; margin-top: 8px; font-size: 0.9rem;">
                                    置信度: <strong>{prob:.4f}</strong>
                                </div>
                                <div style="margin-top: 12px;">
                                    <div style="background: #f3f4f6; border-radius: 10px; height: 8px; overflow: hidden;">
                                        <div style="width: {prob * 100:.1f}%; height: 100%; background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 10px;"></div>
                                    </div>
                                </div>
                                <div style="margin-top: 12px; color: #6b7280; font-size: 0.85rem;">
                                    ✅ 概率 {prob:.4f} ≥ {BEST_THRESHOLD}，判定为抗癌肽
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box" style="border-left: 6px solid #10b981;">
                                <div style="font-size: 1.5rem; font-weight: 600; color: #1a202c;">{pred}</div>
                                <div style="color: #6b7280; margin-top: 8px; font-size: 0.9rem;">
                                    置信度: <strong>{prob:.4f}</strong>
                                </div>
                                <div style="margin-top: 12px;">
                                    <div style="background: #f3f4f6; border-radius: 10px; height: 8px; overflow: hidden;">
                                        <div style="width: {prob * 100:.1f}%; height: 100%; background: linear-gradient(90deg, #10b981, #34d399); border-radius: 10px;"></div>
                                    </div>
                                </div>
                                <div style="margin-top: 12px; color: #6b7280; font-size: 0.85rem;">
                                    ℹ️ 概率 {prob:.4f} < {BEST_THRESHOLD}，判定为非抗癌肽
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

# ---------- Tab 2: 批量预测 ----------
with tab2:
    st.markdown("##### 上传 FASTA 文件进行批量预测")
    uploaded_file = st.file_uploader("", type=["fasta", "fa", "txt"], label_visibility="collapsed")

    if uploaded_file:
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

        st.info(f"📄 共解析到 **{len(sequences)}** 条序列")

        if st.button("🚀 开始批量预测", type="primary"):
            results = []
            progress_bar = st.progress(0)
            for i, (seq_id, seq) in enumerate(sequences):
                pred, prob = predict_sequence(seq)
                results.append({
                    "序列ID": seq_id,
                    "序列片段": seq[:50] + ("..." if len(seq) > 50 else ""),
                    "预测类别": pred,
                    "可信度": f"{prob:.4f}" if prob is not None else "N/A"
                })
                progress_bar.progress((i + 1) / len(sequences))

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, height=400)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 下载预测结果 (CSV)",
                    data=csv_data,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            else:
                st.warning("没有有效的序列进行预测")

# ==================== 页脚 ====================
st.markdown("""
<div style="text-align: center; padding: 20px; color: #6b7280; font-size: 0.9rem; margin-top: 40px;">
    ⚡ 基于 ESM-2 + XGBoost | 仅供学术研究，不构成医疗诊断
</div>
""", unsafe_allow_html=True)