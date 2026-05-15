"""
ЕнергоАгент — веб-інтерфейс на Streamlit.

Запуск:
    cd energy_agent
    $env:OPENAI_API_KEY = "sk-..."
    streamlit run src/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import EnergyAgent, UserProfile  # noqa: E402

# ============================================================
#   Конфігурація сторінки
# ============================================================
st.set_page_config(
    page_title="ЕнергоАгент",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Маленький custom CSS
st.markdown("""
<style>
  .stChatMessage { padding: 0.5rem 0; }
  .agent-trace {
    font-size: 0.82rem;
    background: #f6f8fa;
    border-left: 3px solid #2E75B6;
    padding: 0.45rem 0.7rem;
    margin: 0.25rem 0;
    border-radius: 4px;
    color: #444;
  }
  .agent-trace-tool { border-left-color: #70AD47; }
  .agent-trace-final { border-left-color: #ED7D31; }
  .mode-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 6px;
  }
  .mode-llm { background: #d4f0d4; color: #2e7d32; }
  .mode-demo { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#   Ініціалізація стану сесії
# ============================================================
if "agent" not in st.session_state:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⛔ OPENAI_API_KEY не задано. Запустіть:\n```\n$env:OPENAI_API_KEY = 'sk-...'\nstreamlit run src/app.py\n```")
        st.stop()
    st.session_state.agent = EnergyAgent(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []   # [{role, content, charts, trace}]
if "show_trace" not in st.session_state:
    st.session_state.show_trace = True


# ============================================================
#   Бічна панель
# ============================================================
with st.sidebar:
    st.title("⚡ ЕнергоАгент")

    st.divider()
    st.subheader("📂 Датасет")
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "consumption.csv")
    st.metric("Записів", f"{len(df):,}")
    st.metric("Сумарне споживання, кВт·год", f"{df['consumption_kwh'].sum():.0f}")
    st.metric("Період", f"{df['timestamp'].min()[:10]} → {df['timestamp'].max()[:10]}")

    st.divider()
    st.subheader("🧠 Довгострокова пам'ять")
    profile = st.session_state.agent.profile
    new_name = st.text_input("Ім'я користувача", value=profile.user_name or "", key="input_name")
    new_tariff = st.selectbox(
        "Бажаний тариф",
        options=["", "single_zone", "two_zone", "three_zone"],
        index=["", "single_zone", "two_zone", "three_zone"].index(profile.preferred_tariff or ""),
        key="input_tariff",
    )
    if st.button("💾 Зберегти пам'ять", use_container_width=True):
        profile.user_name = new_name or None
        profile.preferred_tariff = new_tariff or None
        profile.save()
        # Перезавантажити профіль з диска щоб агент одразу бачив нові дані
        st.session_state.agent.profile = profile
        st.toast("Збережено ✓", icon="✅")
        st.rerun()

    st.divider()
    st.subheader("🔧 Налаштування")
    st.session_state.show_trace = st.checkbox(
        "Показувати ReAct-трасу", value=st.session_state.show_trace,
        help="Розгортає послідовність 'думка → виклик інструмента → результат'.",
    )
    if st.button("🔄 Скинути розмову", use_container_width=True):
        st.session_state.agent.reset()
        st.session_state.chat_log.clear()
        st.rerun()

    st.divider()
    with st.expander("💡 Приклади запитів"):
        examples = [
            "Скільки я спожив у січні?",
            "Який тариф мені вигідніший?",
            "Покажи прогноз на наступні 7 днів",
            "Які пікові години навантаження?",
            "Чи є аномалії в грудні?",
            "Покажи графік споживання за весь рік",
            "Скільки я заплачу за лютий за тарифом 'дві зони'?",
            "Порівняй літо і зиму",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state._queued_query = ex
                st.rerun()


# ============================================================
#   Основна область
# ============================================================
st.title("ЕнергоАгент — чат-інтерфейс до енергетичних даних")
st.caption("ШІ-агент, що поєднує LLM з 8 інструментами доступу до даних, "
           "пам'яттю та ReAct-плануванням. Варіант 19, курсова робота.")

# Рендеринг історії
for entry in st.session_state.chat_log:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        for chart_path in entry.get("charts", []):
            if Path(chart_path).exists():
                st.image(chart_path, use_container_width=True)
        if entry.get("trace") and st.session_state.show_trace and entry["role"] == "assistant":
            with st.expander("🔍 Як агент думав (ReAct-траса)"):
                for step in entry["trace"]:
                    if step.step_type == "thought":
                        st.markdown(
                            f"<div class='agent-trace'><b>💭 Думка:</b> {step.content}</div>",
                            unsafe_allow_html=True,
                        )
                    elif step.step_type == "tool_call":
                        args_str = json.dumps(step.tool_args, ensure_ascii=False)
                        st.markdown(
                            f"<div class='agent-trace agent-trace-tool'>"
                            f"<b>🔧 Виклик:</b> <code>{step.tool_name}({args_str})</code></div>",
                            unsafe_allow_html=True,
                        )
                    elif step.step_type == "tool_result":
                        result = step.content
                        if len(result) > 350:
                            result = result[:350] + "..."
                        st.markdown(
                            f"<div class='agent-trace agent-trace-tool'>"
                            f"<b>📋 Результат:</b> <code>{result}</code></div>",
                            unsafe_allow_html=True,
                        )
                    elif step.step_type == "final":
                        st.markdown(
                            f"<div class='agent-trace agent-trace-final'>"
                            f"<b>✅ Фінальна відповідь сформована</b></div>",
                            unsafe_allow_html=True,
                        )

# Поле вводу
queued = st.session_state.pop("_queued_query", None)
user_input = st.chat_input("Запитайте про ваше енергоспоживання…") or queued

if user_input:
    st.session_state.chat_log.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Агент думає…"):
            response = st.session_state.agent.chat(user_input)
        st.markdown(response["reply"])
        for chart_path in response["charts"]:
            if Path(chart_path).exists():
                st.image(chart_path, use_container_width=True)

    st.session_state.chat_log.append({
        "role": "assistant",
        "content": response["reply"],
        "charts": response["charts"],
        "trace": response["trace"],
    })
    st.rerun()