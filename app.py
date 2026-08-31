from __future__ import annotations

import json
import os
from datetime import date
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.core_logic import (
    CoreLogicError,
    SUPPORTED_CITIES,
    WEEKDAYS,
    load_reference,
    query_congestion,
    recommend_adjacent_times,
    recommend_alternative_routes,
)
from src.ui_helpers import (
    build_hourly_profile,
    format_time_difference,
    level_style,
    station_options,
    weekday_label,
)


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "processed" / "dataset_manifest.json"
TAB_LABELS = ("🏠 메인", "⭐ MY", "📊 데이터 인사이트", "💬 챗봇")

CHATBOT_SYSTEM = """[신분]
너는 '천국철'의 직장인 맞춤형 출퇴근 상담 전문가다.
사용자의 출퇴근 상황과 천국철의 혼잡도 분석 결과를 바탕으로 가장 적합한 이동 시간과 이동 방법을 추천한다.

[지식 — 확인된 데이터만 사용]
아래에 제공되는 천국철 조회 결과와 현재 대화에서 사용자가 직접 알려준 정보만 사용한다.
사용자의 출근·퇴근 시간, 도착 희망 시간, 혼잡 회피 여부, 환승 및 도보 선호를 함께 고려한다.

[규칙]
천국철 데이터에 없는 내용은 추측하거나 임의로 만들지 않는다.
정보가 부족하면 출발역, 도착역, 도착 희망 시간 등 필요한 정보를 물어본다.
혼잡도와 이동 시간은 확정적인 정보처럼 말하지 않고 예상 또는 추천으로 안내한다.
실시간 지연이나 운행 상황처럼 확인할 수 없는 정보는 확인할 수 없다고 솔직하게 안내한다.
역 연결·소요 시간 데이터가 없으면 빠른 경로나 환승 경로를 만들어내지 않는다.

[형식]
회사 동료처럼 친절하고 간결한 존댓말을 사용한다.
핵심 정보를 먼저 전달하고, 가능하면 추천 시간과 추천 이유를 함께 설명한다.
필요하면 2~3개의 방법을 비교하고 이모지는 1~2개만 사용한다."""

st.set_page_config(
    page_title="천국철 | 덜 붐비는 지하철 시간 찾기",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def get_reference() -> pd.DataFrame:
    return load_reference()


@st.cache_data(show_spinner=False)
def get_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --green: #55AD66;
            --green-dark: #2F8044;
            --green-pale: #EDF8EF;
            --ink: #27343C;
            --muted: #6D7B83;
            --line: #DDE8DF;
            --surface: #F7F9FA;
        }
        .stApp { background: #F8FBF8; color: var(--ink); }
        [data-testid="stSidebar"] { background: #F4F8F4; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(248,251,248,.96); }
        [data-testid="stAppViewBlockContainer"], .block-container {
            max-width: 1380px;
            padding-top: 4.5rem !important;
            padding-bottom: 4rem;
        }
        .topbar {
            display: flex; align-items: center; justify-content: space-between; gap: 1.2rem;
            padding: .85rem 1.2rem; border: 1px solid var(--line); border-radius: 18px;
            background: rgba(255,255,255,.94); box-shadow: 0 8px 26px rgba(43, 90, 53, .06);
            margin-bottom: 1rem;
        }
        .brand { display: flex; align-items: center; gap: .75rem; }
        .brand-icon { font-size: 2rem; }
        .brand-name { color: var(--green-dark); font-weight: 900; font-size: 1.45rem; line-height: 1.1; }
        .brand-copy { color: var(--muted); font-size: .76rem; margin-top: .2rem; }
        .topnav { display: flex; align-items: center; gap: .3rem; color: #43554A; font-size: .9rem; }
        .topnav span { padding: .65rem .9rem; border-radius: 11px; }
        .topnav .active { background: var(--green-pale); color: var(--green-dark); font-weight: 800; }
        .welcome {
            padding: 1rem 1.2rem; border: 1px solid var(--line); border-radius: 16px;
            background: linear-gradient(120deg, #FFFFFF 0%, #F1FAF2 100%); margin-bottom: 1rem;
        }
        .welcome b { color: var(--green-dark); font-size: 1.08rem; }
        .welcome span { color: var(--muted); margin-left: .5rem; font-size: .9rem; }
        .step-head { display:flex; align-items:center; gap:.7rem; margin: 1.25rem 0 .7rem; }
        .step-number {
            width: 2rem; height: 2rem; border-radius: 50%; display:flex; align-items:center;
            justify-content:center; color:white; background:var(--green); font-weight:900;
        }
        .step-title { color:var(--green-dark); font-size:1.12rem; font-weight:900; }
        .step-subtitle { color:var(--muted); font-size:.86rem; }
        .soft-card {
            height: 100%; padding: 1.1rem 1.2rem; border-radius: 16px;
            border: 1px solid var(--line); background: white;
        }
        .soft-card h4 { margin: 0 0 .4rem; color: #30434C; }
        .soft-card p { margin: 0; color: var(--muted); line-height: 1.55; font-size: .93rem; }
        .condition-bar {
            padding: .9rem 1.1rem; border-radius: 14px; background: white;
            border: 1px solid var(--line); color: #3E5144; margin: .5rem 0 .9rem;
            display:flex; justify-content:space-between; flex-wrap:wrap; gap:.5rem;
        }
        .condition-main { font-weight:800; }
        .condition-date { color:var(--muted); font-size:.88rem; }
        .score-card {
            min-height: 250px; padding: 1.2rem 1.35rem; border-radius: 17px;
            background: white; border: 1px solid var(--line); display:flex;
            align-items:center; justify-content:center; gap:2rem;
        }
        .score-ring {
            width: 150px; height: 150px; border-radius:50%; display:flex; align-items:center;
            justify-content:center; position:relative; flex:none;
        }
        .score-ring::before { content:""; width:116px; height:116px; border-radius:50%; background:white; position:absolute; }
        .score-value { position:relative; text-align:center; font-size:2.2rem; font-weight:900; line-height:1; }
        .score-value small { display:block; font-size:.72rem; color:var(--muted); font-weight:500; margin-top:.4rem; }
        .score-copy h3 { margin:0 0 .45rem; font-size:1.65rem; }
        .score-copy p { color:var(--muted); margin:0; line-height:1.6; }
        .best-card {
            min-height:250px; padding:1.35rem; border-radius:17px; border:1px solid var(--line);
            background:linear-gradient(135deg,#FFFFFF,#EFF8F0); display:flex; flex-direction:column;
            justify-content:center;
        }
        .best-label { color:var(--green-dark); font-weight:800; font-size:.9rem; }
        .best-time { color:#263C2C; font-size:1.8rem; font-weight:900; margin:.45rem 0; }
        .best-note { color:var(--muted); line-height:1.6; }
        .best-badge { display:inline-block; padding:.3rem .65rem; background:#DFF2E2; color:var(--green-dark); border-radius:999px; font-size:.8rem; font-weight:800; }
        .favorite-row { padding:.25rem .3rem; }
        .favorite-row .rec-time { margin:.35rem 0; }
        .chat-shell {
            min-height: 500px; border: 1px solid var(--line); border-radius: 20px;
            background: white; display: flex; flex-direction: column; overflow: hidden;
            box-shadow: 0 8px 24px rgba(43, 90, 53, .05);
        }
        .chat-body {
            flex: 1; min-height: 400px; display: flex; align-items: center;
            justify-content: center; padding: 2rem; text-align: center;
            background: linear-gradient(180deg, #FFFFFF 0%, #F7FBF7 100%);
        }
        .chat-empty-icon { font-size: 3rem; margin-bottom: .6rem; }
        .chat-empty-title { font-size: 1.25rem; font-weight: 900; color: var(--green-dark); }
        .chat-empty-copy { color: var(--muted); margin-top: .45rem; line-height: 1.6; }
        .chat-input-placeholder {
            padding: 1rem 1.15rem; border-top: 1px solid var(--line); color: #98A39B;
            background: #FBFCFB; display: flex; justify-content: space-between;
            align-items: center;
        }
        .chart-card { border:1px solid var(--line); border-radius:17px; background:white; padding:1rem; }
        .chart-title { font-weight:900; color:#36483B; margin-bottom:.3rem; }
        .time-strip { display:flex; gap:.45rem; flex-wrap:wrap; padding:.2rem 0 .8rem; }
        .time-chip { border:1px solid var(--line); background:white; padding:.45rem .75rem; border-radius:10px; color:#526157; font-size:.83rem; }
        .time-chip.selected { background:#EC4F55; color:white; border-color:#EC4F55; font-weight:800; }
        .time-chip.recommended { background:#E8F5E9; color:var(--green-dark); border-color:#BFDCC4; font-weight:800; }
        .section-label { color: var(--green-dark); font-weight: 800; font-size: .82rem; letter-spacing: .07em; margin-top: 1.5rem; }
        }
        div[data-testid="stMetric"] {
            background: white; border: 1px solid var(--line); padding: 1rem 1.1rem;
            border-radius: 16px; box-shadow: 0 6px 20px rgba(54, 83, 94, .05);
        }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        .rec-card {
            min-height: 195px; padding: 1.15rem; border-radius: 17px; background: white;
            border: 1px solid var(--line); box-shadow: 0 7px 22px rgba(54, 83, 94, .06);
        }
        .rec-time { font-size: 1.18rem; font-weight: 800; color: #2D434D; margin-bottom: .25rem; }
        .rec-score { color: #438EA8; font-size: 1.45rem; font-weight: 800; margin: .45rem 0; }
        .rec-note { color: var(--muted); font-size: .9rem; line-height: 1.5; }
        .notice {
            border-left: 4px solid var(--green); background: #F1F8F2; padding: .85rem 1rem;
            border-radius: 4px 12px 12px 4px; color: #536A74; font-size: .92rem;
        }
        .stButton > button {
            width: 100%; border: 0; border-radius: 12px; background: var(--green); color: white;
            font-weight: 800; min-height: 3rem;
        }
        .stButton > button:hover { background: var(--green-dark); color: white; border: 0; }
        div[data-testid="stProgressBar"] > div > div { background-color: var(--green); }
        [data-baseweb="tab-list"] {
            gap: .35rem; padding: .35rem; border: 1px solid var(--line); border-radius: 14px;
            background: white; box-shadow: 0 6px 20px rgba(43, 90, 53, .05);
        }
        [data-baseweb="tab"] {
            height: 2.8rem; padding: 0 1.25rem; border-radius: 10px; color: #56655A;
            font-weight: 700;
        }
        [aria-selected="true"][data-baseweb="tab"] { background: var(--green-pale); color: var(--green-dark); }
        [data-baseweb="tab-highlight"] { background-color: var(--green) !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line); border-radius: 17px; background: rgba(255,255,255,.72);
        }
        @media (max-width: 800px) {
            [data-testid="stAppViewBlockContainer"], .block-container { padding-top: 4rem !important; }
            .topnav { display:none; }
            .score-card { flex-direction:column; text-align:center; gap:1rem; }
            .condition-bar { display:block; }
            .condition-date { margin-top:.35rem; }
        }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_intro() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand"><div class="brand-icon">🚇</div><div><div class="brand-name">천국철</div>
          <div class="brand-copy">지옥철 말고, 천국철 타자 🍀</div></div></div>
          <div class="brand-copy">서울 · 인천 · 대전 · 대구 데이터 기반</div>
        </div>
        <div class="welcome"><b>오늘의 지하철, 조금 더 여유롭게</b>
        <span>시간대별 혼잡도를 비교하고 나에게 맞는 출발 시간을 찾아보세요.</span></div>
        """,
        unsafe_allow_html=True,
    )


def section_title(number: int, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="step-head"><div class="step-number">{number}</div>'
        f'<div class="step-title">{title}</div><div class="step-subtitle">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def favorite_id(search: dict) -> str:
    return "|".join(
        [
            str(search["city"]),
            str(search["line"]),
            str(search["origin_id"]),
            str(search.get("destination_id", search["destination_name"])),
            weekday_label(search["weekday"]),
            str(search["hour"]),
        ]
    )


def save_favorite_and_open_my(item: dict) -> None:
    favorites = list(st.session_state.get("favorites", []))
    favorites = [favorite for favorite in favorites if favorite["id"] != item["id"]]
    favorites.insert(0, item)
    st.session_state["favorites"] = favorites
    st.session_state["report_tabs"] = "⭐ MY"


def remove_favorite(item_id: str) -> None:
    st.session_state["favorites"] = [
        item for item in st.session_state.get("favorites", []) if item["id"] != item_id
    ]
    st.session_state["report_tabs"] = "⭐ MY"


def restore_favorite(item: dict) -> None:
    search = dict(item["search"])
    st.session_state["search"] = search
    st.session_state["main_city"] = search["city"]
    st.session_state["main_line"] = search["line"]
    st.session_state["main_origin"] = search["origin_id"]
    if search.get("destination_id"):
        st.session_state["main_destination"] = search["destination_id"]
    if isinstance(search["weekday"], date):
        st.session_state["main_day_mode"] = "날짜"
        st.session_state["main_date"] = search["weekday"]
    else:
        st.session_state["main_day_mode"] = "요일"
        st.session_state["main_weekday"] = str(search["weekday"])
    st.session_state["main_hour"] = search["hour"]
    st.session_state["main_window"] = search["window_hours"]
    st.session_state["report_tabs"] = "🏠 메인"


def congestion_chart(profile: pd.DataFrame, selected_hour: int) -> alt.Chart:
    base = alt.Chart(profile).encode(
        x=alt.X("시간:N", sort=profile["시간"].tolist(), title="시간대"),
        y=alt.Y("혼잡도 점수:Q", scale=alt.Scale(domain=[0, 100]), title="예상 혼잡도 점수"),
        tooltip=["시간:N", "혼잡도 점수:Q", "혼잡 단계:N", "승하차 수요 중앙값:Q"],
    )
    area = base.mark_area(
        line={"color": "#55AD66", "strokeWidth": 3},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#67B974", offset=0),
                alt.GradientStop(color="#EDF8EF", offset=1),
            ],
            x1=1,
            x2=1,
            y1=0,
            y2=1,
        ),
        opacity=0.72,
    )
    points = base.mark_circle(size=58, color="#4A9B5C")
    selected_time = f"{selected_hour:02d}:00"
    nearest = alt.selection_point(
        name="혼잡시간",
        nearest=True,
        on="pointerover",
        fields=["시간"],
        value=[{"시간": selected_time}],
        empty=False,
        clear=False,
        toggle=False,
    )
    selectors = (
        base.mark_point(opacity=0, size=220, cursor="ew-resize")
        .encode(
            tooltip=[
                alt.Tooltip("시간:N", title="시간대"),
                alt.Tooltip("승하차 수요 중앙값:Q", title="예상 이용 수요", format=",.0f"),
                alt.Tooltip("혼잡도 점수:Q", title="혼잡도 점수", format=".1f"),
                alt.Tooltip("혼잡 단계:N", title="혼잡 단계"),
            ]
        )
        .add_params(nearest)
    )
    marker = (
        base.transform_filter(nearest)
        .mark_rule(
            color="#E29A55",
            strokeDash=[5, 4],
            strokeWidth=2.5,
            cursor="ew-resize",
        )
    )
    active_point = (
        base.transform_filter(nearest)
        .mark_circle(size=125, color="#E29A55", stroke="white", strokeWidth=2)
    )
    value_label = (
        base.transform_filter(nearest)
        .transform_calculate(
            안내="'예상 이용 수요 ' + format(datum['승하차 수요 중앙값'], ',.0f') + '명 · 혼잡도 ' + format(datum['혼잡도 점수'], '.1f') + '점'"
        )
        .mark_text(
            align="center",
            baseline="bottom",
            dy=-12,
            fontSize=11,
            fontWeight="bold",
            color="#B96F2D",
        )
        .encode(text=alt.Text("안내:N"))
    )
    return (
        (area + points + marker + active_point + value_label + selectors)
        .properties(height=330)
        .configure_view(strokeWidth=0)
    )


def demand_chart(profile: pd.DataFrame, selected_hour: int) -> alt.Chart:
    chart_data = profile.copy()
    chart_data["선택"] = chart_data["hour"].eq(selected_hour)
    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("시간:N", sort=chart_data["시간"].tolist(), title="시간대"),
            y=alt.Y("승하차 수요 중앙값:Q", title="예상 이용 수요"),
            color=alt.condition("datum['선택']", alt.value("#EC4F55"), alt.value("#91C99A")),
            tooltip=["시간:N", "승하차 수요 중앙값:Q", "혼잡도 점수:Q", "혼잡 단계:N"],
        )
        .properties(height=330)
        .configure_view(strokeWidth=0)
    )


def favorite_preview_chart(
    profile: pd.DataFrame,
    *,
    selected_hour: int,
    recommended_time: str | None,
) -> alt.Chart:
    chart_data = profile.copy()
    recommended_hour = (
        int(recommended_time.split(":", maxsplit=1)[0]) if recommended_time else None
    )
    chart_data["표시"] = "일반"
    if recommended_hour is not None:
        chart_data.loc[chart_data["hour"].eq(recommended_hour), "표시"] = "추천"
    chart_data.loc[chart_data["hour"].eq(selected_hour), "표시"] = "저장 시간"
    colors = alt.Scale(
        domain=["일반", "추천", "저장 시간"],
        range=["#C8E2CD", "#55AD66", "#EC5B61"],
    )
    bars = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "시간:N",
                sort=chart_data["시간"].tolist(),
                title=None,
                axis=alt.Axis(labelAngle=0, labelExpr="indexof(['06:00','09:00','12:00','15:00','18:00','21:00'], datum.label) >= 0 ? datum.label : ''"),
            ),
            y=alt.Y(
                "혼잡도 점수:Q",
                scale=alt.Scale(domain=[0, 100]),
                title=None,
                axis=alt.Axis(labels=False, ticks=False, domain=False, grid=True),
            ),
            color=alt.Color("표시:N", scale=colors, legend=None),
            tooltip=["시간:N", "혼잡도 점수:Q", "혼잡 단계:N", "표시:N"],
        )
    )
    labels = (
        alt.Chart(chart_data.loc[chart_data["표시"].ne("일반")])
        .mark_text(dy=-7, fontSize=10, fontWeight="bold")
        .encode(
            x=alt.X("시간:N", sort=chart_data["시간"].tolist()),
            y=alt.Y("혼잡도 점수:Q", scale=alt.Scale(domain=[0, 100])),
            text=alt.Text("혼잡도 점수:Q", format=".0f"),
            color=alt.Color("표시:N", scale=colors, legend=None),
        )
    )
    return (bars + labels).properties(height=145).configure_view(strokeWidth=0)


def render_time_recommendations(result: dict) -> None:
    section_title(3, "추천 시간", "현재보다 덜 붐비는 인접 시간대")
    if result["decision"] == "keep_current":
        st.info(result["message"], icon="💙")
        return
    recommendations = result["recommendations"]
    columns = st.columns(len(recommendations))
    for rank, (column, item) in enumerate(zip(columns, recommendations), start=1):
        style = level_style(item["congestion_level"])
        with column:
            st.markdown(
                f"""
                <div class="rec-card">
                  <div class="rec-time">{style['emoji']} {rank}순위 · {item['time_range']}</div>
                  <div class="rec-score">{item['congestion_score']:.1f}점</div>
                  <div class="rec-note"><b>{item['congestion_level']}</b> · {format_time_difference(item['time_difference_hours'])}<br>
                  현재보다 <b>{item['improvement_points']:.1f}점 낮아요.</b><br>{style['message']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_route_recommendations(result: dict) -> None:
    section_title(4, "대체 노선", "혼잡도·소요시간·환승을 함께 비교")
    if result["status"] == "unavailable":
        st.markdown(
            f'<div class="notice">ℹ️ {result["reason"]}<br>현재는 같은 역의 시간대 추천을 우선 제공해요.</div>',
            unsafe_allow_html=True,
        )
        return
    columns = st.columns(len(result["recommendations"]))
    for rank, (column, item) in enumerate(zip(columns, result["recommendations"]), start=1):
        with column:
            st.markdown(
                f"""
                <div class="rec-card">
                  <div class="rec-time">🚇 {rank}순위 · {item['route_name']}</div>
                  <div class="rec-score">비용 {item['recommendation_cost']:.1f}</div>
                  <div class="rec-note">{item['reason']}<br>혼잡도·소요시간·환승 부담을 함께 비교했어요.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption(result["policy"])


def render_data_notice(manifest: dict, *, expanded: bool = False) -> None:
    st.divider()
    with st.expander("📌 데이터 기준과 꼭 알아둘 점", expanded=expanded):
        if manifest.get("cities"):
            basis = " · ".join(
                f"{item['city']} {item['date_min']}~{item['date_max']}"
                for item in manifest["cities"]
            )
            st.write(f"**지역별 데이터 기간:** {basis}")
        st.markdown(
            """
            - 서울·인천·대전·대구에서 공통으로 비교 가능한 **06:00~24:00, 1시간 단위** 데이터를 사용합니다.
            - 혼잡도 점수는 객차 정원이나 실시간 탑승률이 아니라, 선택한 역·요일의 18개 시간대 승하차 수요 중앙값을 서로 비교한 **상대 지표**입니다.
            - 지역별 수집 연도가 다르므로 지역 사이의 절대 인원 순위를 직접 비교하는 용도로 사용하지 않습니다.
            - 날씨·행사·운행 장애·공휴일 등 당일 상황은 반영하지 않으며 실제 혼잡과 차이가 날 수 있습니다.
            - 인천 일부 역 코드 보정과 대구 호선 추정은 공식 코드표로 추가 확인이 필요합니다.
            """
        )


def render_search_results(reference: pd.DataFrame, manifest: dict) -> None:
    section_title(1, "조건 선택", "날짜·시간·노선·역을 고르면 바로 예측해 드려요")
    with st.container(border=True):
        first_row = st.columns([0.8, 0.8, 1.2, 1.2])
        with first_row[0]:
            city = st.selectbox("📍 지역", SUPPORTED_CITIES, key="main_city")
        lines = sorted(
            reference.loc[reference["city"].astype(str).eq(city), "line"].astype(str).unique()
        )
        with first_row[1]:
            line = st.selectbox("🚇 노선", lines, key="main_line")
        origins = station_options(reference, city, line)
        origin_labels = origins.set_index("station_id")["label"].to_dict()
        with first_row[2]:
            origin_id = st.selectbox(
                "🚉 출발역",
                origins["station_id"].tolist(),
                format_func=lambda value: origin_labels[value],
                key="main_origin",
            )
        origin = origins.loc[origins["station_id"].eq(origin_id)].iloc[0]
        destinations = station_options(reference, city)
        destinations = destinations.loc[
            ~destinations["station_id"].eq(origin_id)
        ].reset_index(drop=True)
        destination_labels = destinations.set_index("station_id")["label"].to_dict()
        with first_row[3]:
            destination_id = st.selectbox(
                "🏁 도착역",
                destinations["station_id"].tolist(),
                format_func=lambda value: destination_labels[value],
                key="main_destination",
            )
        destination = destinations.loc[destinations["station_id"].eq(destination_id)].iloc[0]

        second_row = st.columns([0.9, 1.15, 1.1, 1.1])
        with second_row[0]:
            day_mode = st.radio(
                "📅 날짜 기준", ("요일", "날짜"), horizontal=True, key="main_day_mode"
            )
        with second_row[1]:
            if day_mode == "요일":
                weekday_kwargs = {"index": 0} if "main_weekday" not in st.session_state else {}
                weekday_value: object = st.selectbox(
                    "요일",
                    [f"{weekday}요일" for weekday in WEEKDAYS],
                    key="main_weekday",
                    **weekday_kwargs,
                )
            else:
                date_kwargs = {"value": date.today()} if "main_date" not in st.session_state else {}
                weekday_value = st.date_input("이용 날짜", key="main_date", **date_kwargs)
                st.caption(f"{weekday_label(weekday_value)} 패턴으로 조회해요.")
        with second_row[2]:
            hour_kwargs = {"value": 8} if "main_hour" not in st.session_state else {}
            hour = st.select_slider(
                "🕒 출발 시간",
                options=list(range(6, 24)),
                format_func=lambda value: f"{value:02d}:00",
                key="main_hour",
                **hour_kwargs,
            )
        with second_row[3]:
            window_kwargs = {"value": 2} if "main_window" not in st.session_state else {}
            window_hours = st.slider(
                "↔ 비교 범위",
                min_value=1,
                max_value=4,
                help="선택 시간의 앞뒤를 비교해요.",
                key="main_window",
                **window_kwargs,
            )
        submitted = st.button(
            "🔍 혼잡도와 추천 확인하기", type="primary", key="main_submit"
        )
        st.caption("승하차 공개 데이터를 바탕으로 한 예상 결과이며, 실시간 객차 혼잡률은 아니에요.")

    if submitted:
        st.session_state["search"] = {
            "city": city,
            "line": line,
            "origin_id": origin_id,
            "origin_code": str(origin["station_code"]),
            "origin_name": str(origin["station_name"]),
            "destination_id": destination_id,
            "destination_name": str(destination["station_name"]),
            "weekday": weekday_value,
            "hour": hour,
            "window_hours": window_hours,
        }

    if "search" not in st.session_state:
        section_title(2, "혼잡도와 추천", "조건을 선택하면 이곳에 결과가 나타나요")
        placeholder_columns = st.columns([1, 1.25])
        with placeholder_columns[0]:
            st.markdown(
                """<div class="score-card"><div class="score-ring" style="background:conic-gradient(#DDE8DF 0deg,#DDE8DF 360deg)">
                <div class="score-value" style="color:#7A8A7E">?<small>/100</small></div></div>
                <div class="score-copy"><h3 style="color:#5D6B60">혼잡도 확인</h3><p>위에서 조건을 선택하면<br>예상 혼잡 요약이 나타나요.</p></div></div>""",
                unsafe_allow_html=True,
            )
        with placeholder_columns[1]:
            st.markdown(
                """<div class="best-card"><div class="best-label">🍀 천국철 추천</div>
                <div class="best-time">조금 더 편안한 이동</div><div class="best-note">요일과 시간을 고르면 전후 시간대를 비교해 가장 여유로운 후보를 알려드릴게요.</div></div>""",
                unsafe_allow_html=True,
            )
        st.info("위에서 이동 조건을 고른 뒤 **혼잡도와 추천 확인하기**를 눌러 주세요.", icon="👆")
        return
    search = st.session_state["search"]

    try:
        query = query_congestion(
            city=search["city"],
            weekday=search["weekday"],
            hour=search["hour"],
            line=search["line"],
            station_code=search["origin_code"],
            reference=reference,
        )
        time_result = recommend_adjacent_times(
            city=search["city"],
            weekday=search["weekday"],
            hour=search["hour"],
            line=search["line"],
            station_code=search["origin_code"],
            window_hours=search["window_hours"],
            reference=reference,
        )
        route_result = recommend_alternative_routes(
            city=search["city"],
            origin_station=search["origin_name"],
            destination_station=search["destination_name"],
            weekday=search["weekday"],
            hour=search["hour"],
            reference=reference,
        )
        profile = build_hourly_profile(
            reference, station_id=search["origin_id"], weekday=search["weekday"]
        )
    except CoreLogicError as error:
        st.error(f"조건에 맞는 결과를 만들지 못했어요. {error}")
        st.info("메인 탭 위쪽에서 다른 지역·노선·역 또는 시간을 선택해 주세요.")
        return

    summary = query["summary"]
    style = level_style(summary["congestion_level"])
    section_title(2, "혼잡도 결과", "선택한 조건의 예상 혼잡 요약")
    st.markdown(
        f"""
        <div class="condition-bar"><div class="condition-main">{search['origin_name']} &nbsp;→&nbsp; {search['destination_name']}</div>
        <div class="condition-date">{search['city']} · {search['line']} · {weekday_label(search['weekday'])} · {search['hour']:02d}:00 기준</div></div>
        """,
        unsafe_allow_html=True,
    )
    overview_columns = st.columns([1, 1.25])
    score_degrees = summary["congestion_score"] * 3.6
    with overview_columns[0]:
        st.markdown(
            f"""<div class="score-card"><div class="score-ring" style="background:conic-gradient({style['color']} 0deg,{style['color']} {score_degrees:.1f}deg,#F0F2F0 {score_degrees:.1f}deg,#F0F2F0 360deg)">
            <div class="score-value" style="color:{style['color']}">{summary['congestion_score']:.0f}<small>/100</small></div></div>
            <div class="score-copy"><h3 style="color:{style['color']}">{style['emoji']} {summary['congestion_level']}</h3>
            <p>{style['message']}<br>출발역 예상 수요 <b>{summary['median_passenger_volume']:,}명</b></p></div></div>""",
            unsafe_allow_html=True,
        )
    best = time_result["recommendations"][0] if time_result["recommendations"] else None
    with overview_columns[1]:
        if best:
            st.markdown(
                f"""<div class="best-card"><div class="best-label">🍀 천국철 추천 시간 &nbsp;<span class="best-badge">추천</span></div>
                <div class="best-time">{best['time_range']} 이용을 추천해요</div>
                <div class="best-note">현재보다 예상 혼잡도 점수가 <b>{best['improvement_points']:.1f}점 낮아요.</b><br>
                예상 점수 {best['congestion_score']:.1f}점 · {best['congestion_level']} · {format_time_difference(best['time_difference_hours'])}</div></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="best-card"><div class="best-label">🍀 천국철 추천</div>
                <div class="best-time">현재 시간을 유지해도 좋아요</div><div class="best-note">{time_result['message']}</div></div>""",
                unsafe_allow_html=True,
            )
    metric_columns = st.columns(3)
    metric_columns[0].metric("예상 혼잡 단계", f"{style['emoji']} {summary['congestion_level']}")
    metric_columns[1].metric("승하차 수요 중앙값", f"{summary['median_passenger_volume']:,}명")
    metric_columns[2].metric("분석 표본", f"최소 {summary['minimum_sample_days']}일")
    favorite = {
        "id": favorite_id(search),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "search": dict(search),
        "score": summary["congestion_score"],
        "level": summary["congestion_level"],
        "demand": summary["median_passenger_volume"],
        "recommended_time": best["time_range"] if best else None,
        "improvement_points": best["improvement_points"] if best else 0.0,
    }
    saved_ids = {item["id"] for item in st.session_state.get("favorites", [])}
    favorite_label = "✅ MY에서 즐겨찾기 보기" if favorite["id"] in saved_ids else "⭐ 이 조건을 MY에 즐겨찾기"
    st.button(
        favorite_label,
        key="save_current_favorite",
        type="secondary",
        on_click=save_favorite_and_open_my,
        args=(favorite,),
    )

    section_title(2, "예측하기", "시간대별 예상 이용 수요와 혼잡도 비교")
    recommended_hour = best["hour"] if best else None
    chip_hours = sorted(set([max(6, search["hour"] - 2), max(6, search["hour"] - 1), search["hour"], min(23, search["hour"] + 1), min(23, search["hour"] + 2)]))
    prediction_key = f"prediction_hour_{abs(hash(favorite_id(search)))}"
    preview_hour = st.segmented_control(
        "그래프에서 강조할 시간",
        options=chip_hours,
        default=search["hour"],
        format_func=lambda value: (
            f"{value:02d}:00 · 추천" if value == recommended_hour else f"{value:02d}:00"
        ),
        key=prediction_key,
        selection_mode="single",
        required=True,
        width="stretch",
    )
    preview_row = profile.loc[profile["hour"].eq(preview_hour)].iloc[0]
    preview_marker = "추천 시간" if preview_hour == recommended_hour else "선택 시간"
    st.caption(
        f"{preview_marker} {preview_hour:02d}:00~{preview_hour + 1:02d}:00 · "
        f"혼잡도 {preview_row['혼잡도 점수']:.1f}점 · {preview_row['혼잡 단계']}"
    )
    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown('<div class="chart-title">시간대별 예상 이용 수요</div>', unsafe_allow_html=True)
        st.altair_chart(demand_chart(profile, preview_hour), width="stretch")
    with chart_columns[1]:
        st.markdown('<div class="chart-title">시간대별 혼잡도 추이</div>', unsafe_allow_html=True)
        st.altair_chart(congestion_chart(profile, preview_hour), width="stretch")
    st.caption(
        "혼잡도 추이 그래프에서 마우스를 움직이거나 드래그하면 주황 점선이 따라가며 "
        "해당 시간의 예상 이용 수요와 혼잡도 점수를 보여줘요."
    )
    st.caption("혼잡도는 선택한 출발역의 해당 요일·시간대 승하차 수요를 기준으로 계산합니다.")

    render_time_recommendations(time_result)
    render_route_recommendations(route_result)
    render_data_notice(manifest)


def render_my_tab(reference: pd.DataFrame) -> None:
    section_title(1, "MY", "즐겨찾기한 이동 조건과 추천을 모아봐요")
    favorites = list(st.session_state.get("favorites", []))
    if not favorites:
        st.markdown(
            """<div class="best-card"><div class="best-label">⭐ 나만의 이동 목록</div>
            <div class="best-time">아직 즐겨찾기가 없어요</div>
            <div class="best-note">메인 탭에서 혼잡도를 확인한 뒤 <b>이 조건을 MY에 즐겨찾기</b>를 눌러 주세요.</div></div>""",
            unsafe_allow_html=True,
        )
        return

    summary_columns = st.columns(3)
    summary_columns[0].metric("저장한 이동", f"{len(favorites)}개")
    summary_columns[1].metric(
        "가장 여유로운 조건", f"{min(item['score'] for item in favorites):.1f}점"
    )
    recommended_count = sum(bool(item.get("recommended_time")) for item in favorites)
    summary_columns[2].metric("추천 시간 포함", f"{recommended_count}개")

    for index, item in enumerate(favorites):
        search = item["search"]
        style = level_style(item["level"])
        with st.container(border=True):
            content, preview, actions = st.columns([2.4, 1.8, 0.75])
            with content:
                recommendation = (
                    f"추천 시간 <b>{item['recommended_time']}</b> · 현재보다 {item['improvement_points']:.1f}점 낮음"
                    if item.get("recommended_time")
                    else "현재 시간 유지 추천"
                )
                st.markdown(
                    f"""<div class="favorite-row"><div class="best-label">{style['emoji']} {search['city']} · {search['line']} · {weekday_label(search['weekday'])} {search['hour']:02d}:00</div>
                    <div class="rec-time">{search['origin_name']} → {search['destination_name']}</div>
                    <div class="rec-note"><b style="color:{style['color']}">{item['score']:.1f}점 · {item['level']}</b> &nbsp;|&nbsp; 예상 수요 {item['demand']:,}명<br>
                    {recommendation}<br><small>{item['saved_at']} 저장</small></div></div>""",
                    unsafe_allow_html=True,
                )
            with preview:
                st.markdown('<div class="chart-title">혼잡도 미리보기</div>', unsafe_allow_html=True)
                try:
                    profile = build_hourly_profile(
                        reference,
                        station_id=search["origin_id"],
                        weekday=search["weekday"],
                    )
                    st.altair_chart(
                        favorite_preview_chart(
                            profile,
                            selected_hour=search["hour"],
                            recommended_time=item.get("recommended_time"),
                        ),
                        width="stretch",
                        key=f"favorite_preview_{index}",
                    )
                    st.caption("🔴 저장 시간  ·  🟢 추천 시간")
                except CoreLogicError as error:
                    st.caption(f"미리보기를 만들 수 없어요: {error}")
            with actions:
                st.button(
                    "🔎 다시 조회",
                    key=f"restore_favorite_{index}",
                    on_click=restore_favorite,
                    args=(item,),
                )
                st.button(
                    "🗑️ 삭제",
                    key=f"remove_favorite_{index}",
                    on_click=remove_favorite,
                    args=(item["id"],),
                )


def render_insights_tab(reference: pd.DataFrame, manifest: dict) -> None:
    section_title(1, "데이터 인사이트", "지역별 지하철 수요 패턴을 한눈에 확인해요")
    station_count = int(reference["station_id"].nunique())
    line_count = int(reference[["city", "line"]].drop_duplicates().shape[0])
    metric_columns = st.columns(4)
    metric_columns[0].metric("대상 지역", f"{reference['city'].nunique()}곳")
    metric_columns[1].metric("분석 노선", f"{line_count}개")
    metric_columns[2].metric("분석 역", f"{station_count:,}개")
    metric_columns[3].metric("혼잡도 참조값", f"{len(reference):,}개")

    selected_city = st.selectbox("살펴볼 지역", SUPPORTED_CITIES, key="insight_city")
    city_data = reference.loc[reference["city"].astype(str).eq(selected_city)].copy()
    hourly = (
        city_data.groupby("hour", observed=True)
        .agg(혼잡도_중앙값=("congestion_score", "median"), 승하차_수요=("median_passenger_volume", "sum"))
        .reset_index()
    )
    hourly["시간"] = hourly["hour"].map(lambda value: f"{int(value):02d}:00")
    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown('<div class="chart-title">지역 전체 시간대별 상대 혼잡도</div>', unsafe_allow_html=True)
        chart = (
            alt.Chart(hourly)
            .mark_line(point=True, color="#55AD66", strokeWidth=3)
            .encode(
                x=alt.X("시간:N", sort=hourly["시간"].tolist(), title="시간대"),
                y=alt.Y("혼잡도_중앙값:Q", scale=alt.Scale(domain=[0, 100]), title="혼잡도 중앙값"),
                tooltip=["시간:N", alt.Tooltip("혼잡도_중앙값:Q", format=".1f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, width="stretch")
    with chart_columns[1]:
        st.markdown('<div class="chart-title">평균 수요가 높은 역 TOP 10</div>', unsafe_allow_html=True)
        top_stations = (
            city_data.groupby(["line", "station_name", "station_id"], observed=True)[
                "median_passenger_volume"
            ]
            .mean()
            .nlargest(10)
            .reset_index()
        )
        top_stations["역"] = top_stations["station_name"].astype(str) + " · " + top_stations["line"].astype(str)
        station_chart = (
            alt.Chart(top_stations)
            .mark_bar(color="#79B985", cornerRadiusEnd=5)
            .encode(
                y=alt.Y("역:N", sort="-x", title=None),
                x=alt.X("median_passenger_volume:Q", title="시간대 평균 수요 중앙값"),
                tooltip=["역:N", alt.Tooltip("median_passenger_volume:Q", format=",.0f")],
            )
            .properties(height=320)
        )
        st.altair_chart(station_chart, width="stretch")
    st.caption("지역별 데이터 기간이 다르므로 서로 다른 지역의 절대 수요를 직접 비교하지 않습니다.")


def chatbot_api_key() -> str:
    """Read the Claude key without ever storing it in source code."""
    try:
        secret_key = str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError):
        secret_key = ""
    return secret_key or os.getenv("ANTHROPIC_API_KEY", "").strip()


def chatbot_model() -> str:
    try:
        secret_model = str(st.secrets.get("ANTHROPIC_MODEL", "")).strip()
    except (FileNotFoundError, KeyError):
        secret_model = ""
    return secret_model or os.getenv("ANTHROPIC_MODEL", "").strip() or "claude-opus-4-8"


def chatbot_context(reference: pd.DataFrame) -> str:
    search = st.session_state.get("search")
    if not search:
        return "[현재 천국철 조회 결과]\n아직 사용자가 메인 탭에서 이동 조건을 조회하지 않았다."

    lines = [
        "[현재 천국철 조회 결과]",
        f"지역: {search['city']}",
        f"노선: {search['line']}",
        f"출발역: {search['origin_name']}",
        f"도착역: {search['destination_name']}",
        f"요일: {weekday_label(search['weekday'])}",
        f"선택 시간: {search['hour']:02d}:00~{search['hour'] + 1:02d}:00",
    ]
    try:
        query = query_congestion(
            city=search["city"],
            weekday=search["weekday"],
            hour=search["hour"],
            line=search["line"],
            station_code=search["origin_code"],
            reference=reference,
        )
        summary = query["summary"]
        lines.extend(
            [
                f"출발역 예상 혼잡도: {summary['congestion_score']:.1f}점 ({summary['congestion_level']})",
                f"출발역 예상 승하차 수요 중앙값: {summary['median_passenger_volume']:,}명",
                f"최소 분석 표본: {summary['minimum_sample_days']}일",
            ]
        )
        time_result = recommend_adjacent_times(
            city=search["city"],
            weekday=search["weekday"],
            hour=search["hour"],
            line=search["line"],
            station_code=search["origin_code"],
            window_hours=search["window_hours"],
            reference=reference,
        )
        recommendations = time_result.get("recommendations", [])[:3]
        if recommendations:
            lines.append("더 한산한 추천 시간:")
            lines.extend(
                f"- {item['time_range']}: {item['congestion_score']:.1f}점 "
                f"({item['congestion_level']}), 현재보다 {item['improvement_points']:.1f}점 낮음"
                for item in recommendations
            )
        else:
            lines.append(f"시간 추천: {time_result['message']}")
    except CoreLogicError as error:
        lines.append(f"조회 결과 계산 오류: {error}")

    lines.append("역 연결·환승·구간별 소요 시간 자료는 현재 제공되지 않는다.")
    return "\n".join(lines)


def render_chatbot_tab(reference: pd.DataFrame) -> None:
    section_title(1, "챗봇", "천국철 조회 결과를 바탕으로 Claude와 상담해요")
    api_key = chatbot_api_key()
    if not api_key:
        st.warning(
            "Claude API 키가 설정되지 않았습니다. 로컬에서는 `ANTHROPIC_API_KEY` 환경변수, "
            "배포 환경에서는 Streamlit Secrets에 같은 이름으로 등록해 주세요."
        )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": "안녕하세요! 메인 탭에서 조회한 혼잡도 결과를 바탕으로 이동 시간을 함께 골라드릴게요. 무엇이 궁금하세요?",
            }
        ]

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "예: 지금보다 덜 붐비는 시간은 언제야?",
        disabled=not bool(api_key),
    )
    if not prompt:
        st.caption("Claude는 현재 대화 내용과 천국철의 조회 결과만 전달받습니다. 실시간 운행 정보는 확인하지 못합니다.")
        return

    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("천국철 상담 답변을 만들고 있어요..."):
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=chatbot_model(),
                    max_tokens=1024,
                    system=f"{CHATBOT_SYSTEM}\n\n{chatbot_context(reference)}",
                    messages=st.session_state["chat_messages"],
                )
                answer = "\n".join(
                    block.text for block in response.content if getattr(block, "type", "") == "text"
                ).strip()
                if not answer:
                    answer = "답변을 만들지 못했습니다. 질문을 조금 다르게 입력해 주세요."
                st.markdown(answer)
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as error:
                error_name = type(error).__name__
                st.error(f"Claude API 호출에 실패했습니다 ({error_name}). API 키와 모델 설정을 확인해 주세요.")

    st.caption("Claude 답변은 예상 혼잡도 상담을 돕기 위한 참고 정보이며, 실시간 운행 정보가 아닙니다.")


def main() -> None:
    inject_style()
    try:
        reference = get_reference()
    except CoreLogicError as error:
        st.error(f"분석 데이터를 불러오지 못했어요: {error}")
        st.stop()
    manifest = get_manifest()

    render_intro()
    main_tab, my_tab, insights_tab, chatbot_tab = st.tabs(
        TAB_LABELS,
        key="report_tabs",
        default="🏠 메인",
    )
    with main_tab:
        render_search_results(reference, manifest)
    with my_tab:
        render_my_tab(reference)
    with insights_tab:
        render_insights_tab(reference, manifest)
    with chatbot_tab:
        render_chatbot_tab(reference)


if __name__ == "__main__":
    main()
