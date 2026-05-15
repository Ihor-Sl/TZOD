"""
ЕнергоАгент — ядро ШІ-агента з підтримкою ReAct-циклу.

Архітектурні компоненти (за схемою з лекції):
  1. LLM-основа       — gpt-4o-mini / gpt-3.5 / Groq / Ollama (через OpenAI-сумісний API)
  2. Інструменти       — модуль tools.py (8 функцій)
  3. Пам'ять           —
        короткострокова: повна історія повідомлень у поточній сесії
        довгострокова:   user_profile.json (зберігається між сесіями)
  4. Планувальник     — реалізується самим LLM через ReAct (Thought → Action → Observation → …)
  5. Саморефлексія    — агент може зробити кілька ітерацій tool-calls,
                         перш ніж сформулювати фінальну відповідь.

Підтримуються 2 режими:
  • LLM mode — справжній API (OpenAI / OpenAI-сумісний). Параметри в .env або
    у конструкторі. Якщо ключ є — агент користується LLM.
  • Demo mode — офлайн fallback з rule-based intent detection. Використовується,
    якщо ключ не знайдено. Дозволяє демонструвати UI/інструменти/пам'ять навіть
    без інтернету.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "user_profile.json"

SYSTEM_PROMPT = """Ти — ЕнергоАгент, вузькоспеціалізований асистент виключно для аналізу електроспоживання домогосподарства.

ТВОЯ РОЛЬ:
  • Аналізувати дані погодинного споживання електроенергії домогосподарства.
  • Допомагати оптимізувати витрати, обирати тариф, прогнозувати споживання.
  • Відповідати українською мовою, коротко й по суті, з конкретними числами.

ПРИНЦИП РОБОТИ (ReAct):
  1. Прочитай запит користувача.
  2. Визнач, які інструменти потрібні (можна викликати кілька).
  3. Виклич інструмент(и), отримай результати у вигляді JSON.
  4. Якщо для відповіді потрібно ще щось — виклич додаткові інструменти.
  5. Сформулюй фінальну відповідь — структуровану, з цифрами і одиницями.

ПАРАМЕТРИ ПЕРІОДУ:
  • Дані доступні за 2024 рік. Дати у форматі YYYY-MM-DD.
  • Якщо період не названий — використовуй увесь рік.
  • Місяці: січень=01, лютий=02, березень=03, …, грудень=12.

СТИЛЬ ВІДПОВІДЕЙ:
  • Завжди наводь конкретні числа: кВт·год, грн, дати.
  • Структуруй довгі відповіді списками або таблицями.
  • Якщо доречно — додавай коротку інтерпретацію або рекомендацію.
  • Не вигадуй чисел поза тим, що повернули інструменти.

СУВОРО ЗАБОРОНЕНО:
  • Відповідати на запити не пов'язані з енергоспоживанням, тарифами або аналізом даних.
  • Писати код, пояснювати алгоритми, відповідати на загальні питання.
  • Якщо запит не стосується енергетики — відповідай: «Я спеціалізуюся виключно на аналізі енергоспоживання. Спробуйте запитати про споживання, тарифи або прогнози.»
"""


# ============================================================
#   Довгострокова пам'ять (user profile)
# ============================================================
@dataclass
class UserProfile:
    user_name: Optional[str] = None
    preferred_tariff: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> "UserProfile":
        if PROFILE_PATH.exists():
            with open(PROFILE_PATH, encoding="utf-8") as f:
                return cls(**json.load(f))
        return cls()

    def save(self) -> None:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

    def context_block(self) -> str:
        parts = []
        if self.user_name:
            parts.append(f"Ім'я користувача: {self.user_name}.")
        if self.preferred_tariff:
            parts.append(f"Бажаний тариф за замовчуванням: {self.preferred_tariff}.")
        if self.notes:
            parts.append("Нотатки про користувача: " + "; ".join(self.notes[-5:]) + ".")
        return " ".join(parts) if parts else ""


# ============================================================
#   Лог ReAct-кроків (для UI та звіту)
# ============================================================
@dataclass
class TraceStep:
    step_type: str   # "thought" | "tool_call" | "tool_result" | "final"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


# ============================================================
#   Агент
# ============================================================
class EnergyAgent:
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 base_url: Optional[str] = None,
                 max_iterations: int = 5):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.max_iterations = max_iterations
        self.profile = UserProfile.load()
        self.history: list[dict] = []
        self.trace: list[TraceStep] = []

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY не задано. Встановіть змінну середовища.")

        from openai import OpenAI
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    # -----------------------------------------------------------
    def reset(self) -> None:
        self.history.clear()
        self.trace.clear()

    # -----------------------------------------------------------
    def chat(self, user_message: str) -> dict:
        """Основний цикл. Повертає {"reply": str, "trace": [...], "charts": [...]}."""
        self.trace.clear()
        result = self._chat_with_llm(user_message)
        # Автоматично витягти факти про користувача і зберегти в notes
        self._extract_memory(user_message, result["reply"])
        return result

    # ============================================================
    #   Автоматичне оновлення довгострокової пам'яті
    # ============================================================
    def _extract_memory(self, user_message: str, agent_reply: str) -> None:
        """Просить LLM знайти нові факти про користувача і дописати в notes."""
        try:
            existing = "; ".join(self.profile.notes[-10:]) if self.profile.notes else "немає"
            prompt = f"""Проаналізуй повідомлення користувача та відповідь агента. \
Знайди нові конкретні факти про користувача, які варто запам'ятати між сесіями \
(наприклад: цікавиться певним місяцем, має аномальне споживання, хоче знизити витрати тощо).

Вже збережені нотатки: {existing}

Повідомлення користувача: {user_message}
Відповідь агента: {agent_reply}

Відповідай ТІЛЬКИ у форматі JSON-масиву рядків з НОВИМИ фактами (не дублюй вже збережені).
Якщо нових фактів немає — поверни порожній масив: []
Приклад: ["Цікавиться споживанням у грудні", "Хоче перейти на двозонний тариф"]"""

            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or "[]"
            # Витягнути JSON навіть якщо модель обгорнула його в ```
            raw = raw.strip().strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            new_notes: list[str] = json.loads(raw)
            if new_notes:
                self.profile.notes.extend(new_notes)
                # Тримати не більше 20 нотаток
                self.profile.notes = self.profile.notes[-20:]
                self.profile.save()
        except Exception:
            pass  # пам'ять не критична — мовчки ігноруємо помилки

    # ============================================================
    #   LLM-режим (ReAct через function calling)
    # ============================================================
    def _chat_with_llm(self, user_message: str) -> dict:
        sys_prompt = SYSTEM_PROMPT
        ctx = self.profile.context_block()
        if ctx:
            sys_prompt += "\n\nДОВГОСТРОКОВА ПАМ'ЯТЬ ПРО КОРИСТУВАЧА:\n" + ctx

        messages = [{"role": "system", "content": sys_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        charts: list[str] = []

        for iteration in range(self.max_iterations):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
            )
            choice = response.choices[0]
            msg = choice.message

            # Зберегти thought (контент моделі) якщо є
            if msg.content:
                self.trace.append(TraceStep("thought", msg.content))

            if not msg.tool_calls:
                # Фінальна відповідь
                final = msg.content or ""
                self.trace.append(TraceStep("final", final))
                self.history.append({"role": "user", "content": user_message})
                self.history.append({"role": "assistant", "content": final})
                return {"reply": final, "trace": list(self.trace), "charts": charts}

            # Виклики інструментів
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                self.trace.append(TraceStep("tool_call", "", tool_name=name, tool_args=args))
                fn = TOOL_FUNCTIONS.get(name)
                if not fn:
                    result = json.dumps({"error": f"Невідомий інструмент {name}"}, ensure_ascii=False)
                else:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = json.dumps({"error": str(e)}, ensure_ascii=False)

                self.trace.append(TraceStep("tool_result", result, tool_name=name))

                # Якщо інструмент згенерував графік — підхопити шлях
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and "chart_path" in parsed:
                        charts.append(parsed["chart_path"])
                except Exception:
                    pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Перевищено max_iterations
        fallback = "Не вдалося отримати відповідь за відведену кількість ітерацій."
        self.trace.append(TraceStep("final", fallback))
        return {"reply": fallback, "trace": list(self.trace), "charts": charts}


# ============================================================
#   CLI для smoke-тесту
# ============================================================
if __name__ == "__main__":
    import sys
    agent = EnergyAgent()
    print(f"--- режим: {agent.mode()} ---")
    queries = sys.argv[1:] or [
        "Скільки я спожив у січні?",
        "Порівняй тарифи за весь рік",
        "Покажи прогноз на 5 днів",
        "Які пікові години?",
        "Чи є аномалії в грудні?",
    ]
    for q in queries:
        print(f"\n>>> {q}")
        out = agent.chat(q)
        print(out["reply"])