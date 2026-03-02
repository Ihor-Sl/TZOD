import random
from dataclasses import dataclass
from typing import Optional

import markovify


DEMO_CORPUS = """
The project aims to reduce peak demand by shifting non-critical loads to off-peak hours.
A baseline assessment was performed using smart meter data and weather-normalized consumption.
The facility operates with a two-zone tariff and includes electric heating and ventilation systems.
Energy efficiency measures include LED retrofits, VFD installation, and insulation improvements.
Demand response scenarios were evaluated for winter and summer operating conditions.
The monitoring system collects hourly readings and detects anomalies based on historical patterns.
A pilot was executed to validate savings under different occupancy profiles and temperature ranges.
The report describes CAPEX, OPEX, payback period, and sensitivity to electricity price changes.
Measured savings were compared with expected values using a regression model and confidence intervals.
Power factor correction was recommended to reduce reactive energy and transformer losses.
Renewable integration includes rooftop PV and battery storage for short-term balancing.
The dispatch strategy prioritizes self-consumption and limits grid export under local regulations.
Equipment downtime and maintenance windows were accounted for in the measurement plan.
The project scope includes HVAC optimization, boiler replacement, and building automation tuning.
Stakeholder interviews highlighted comfort constraints and operational risks.
The verification protocol follows IPMVP options and includes independent data validation.
Monthly energy performance indicators were produced for management review.
The control algorithm adapts setpoints based on outdoor temperature and day-night schedules.
The system flagged abnormal night consumption indicating potential leakage or standby losses.
A corrective action plan was issued with responsibilities and due dates.
The upgraded meters support Modbus and publish data into a centralized data lake.
Forecasting models were trained on two years of data including holidays and seasonal patterns.
The implementation reduced consumption while maintaining indoor air quality requirements.
The project roadmap includes scaling to additional sites after the pilot phase.
Procurement prioritized certified equipment and vendor warranties.
Grid constraints and transformer capacity were reviewed before adding new loads.
The expected annual savings are reported in kWh, kW peak reduction, and CO2 equivalent.
Risk analysis includes supply delays, inaccurate baselines, and behavioral rebound effects.
The final report summarizes outcomes, lessons learned, and next steps for continuous improvement.
"""


@dataclass
class MarkovConfig:
    state_size: int = 2                 # 2-3 зазвичай найбільш читабельні
    tries_per_sentence: int = 200
    seed: Optional[int] = None          # для відтворюваності


def build_model(corpus: str, cfg: MarkovConfig) -> markovify.Text:
    if cfg.seed is not None:
        random.seed(cfg.seed)

    model = markovify.NewlineText(
        corpus,
        state_size=cfg.state_size,
        retain_original=False,
        well_formed=True
    )
    return model


def generate_sentence(model: markovify.Text, min_words=7, max_words=30, tries=200) -> str:
    """
    Генерує одне речення, гарантуючи, що воно не None.
    """
    for _ in range(tries):
        s = model.make_sentence(tries=tries)
        if not s:
            continue
        w = s.split()
        if min_words <= len(w) <= max_words:
            return s
    s = model.make_sentence(tries=tries) or model.make_short_sentence(180) or ""
    return s.strip()


def generate_paragraph(model: markovify.Text, n_sentences=4) -> str:
    sentences = [generate_sentence(model) for _ in range(n_sentences)]
    sentences = [s for s in sentences if s]
    return " ".join(sentences)


def generate_energy_report(model: markovify.Text, rng: random.Random) -> str:
    titles = [
        "Energy Efficiency Upgrade Report",
        "Demand Management Pilot Summary",
        "Smart Metering & Forecasting Assessment",
        "HVAC Optimization Project Brief",
        "PV + Storage Integration Feasibility Report",
    ]
    sites = ["Office Building", "Logistics Warehouse", "Municipal School", "Manufacturing Site", "Data Center Edge Room"]
    periods = ["Q1 2026", "Winter Season 2025/2026", "Feb–Mar 2026", "2025 Baseline Year", "Pilot Month 3"]
    kpis = ["kWh reduction", "peak kW shaving", "CO2e savings", "payback period", "comfort compliance"]

    title = rng.choice(titles)
    site = rng.choice(sites)
    period = rng.choice(periods)
    kpi = rng.choice(kpis)

    parts = [
        f"# {title}\n",
        f"**Site:** {site}\n**Period:** {period}\n**Primary KPI:** {kpi}\n",
        "## Executive Summary\n" + generate_paragraph(model, n_sentences=4) + "\n",
        "## Baseline & Data\n" + generate_paragraph(model, n_sentences=4) + "\n",
        "## Measures Implemented\n" + generate_paragraph(model, n_sentences=4) + "\n",
        "## Results & Verification\n" + generate_paragraph(model, n_sentences=5) + "\n",
        "## Risks & Next Steps\n" + generate_paragraph(model, n_sentences=4) + "\n",
    ]
    return "\n".join(parts)


def main():
    cfg = MarkovConfig(state_size=2, seed=11)

    corpus = DEMO_CORPUS

    model = build_model(corpus, cfg)
    rng = random.Random(cfg.seed)

    print("=== Random project descriptions ===")
    for i in range(3):
        desc = generate_paragraph(model, n_sentences=3)
        print(f"\n[{i+1}] {desc}")

    print("\n\n=== Full synthetic report ===\n")
    report = generate_energy_report(model, rng)
    print(report)


if __name__ == "__main__":
    main()