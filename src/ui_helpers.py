from __future__ import annotations

from typing import Any

import pandas as pd

from src.core_logic import DataNotFoundError, WEEKDAYS, normalize_weekday


LEVEL_STYLES: dict[str, dict[str, str]] = {
    "매우 여유": {"emoji": "🍀", "color": "#55AD66", "message": "한결 여유로운 시간대예요."},
    "여유": {"emoji": "🌿", "color": "#79B985", "message": "비교적 편안하게 이동할 수 있어요."},
    "보통": {"emoji": "🚇", "color": "#7F8C96", "message": "이용객이 적당히 있는 시간대예요."},
    "혼잡": {"emoji": "🚶", "color": "#E5A45D", "message": "조금 붐빌 수 있어요."},
    "매우 혼잡": {"emoji": "🚨", "color": "#D96B6B", "message": "가능하면 다른 시간대를 살펴보세요."},
}


def station_options(reference: pd.DataFrame, city: str, line: str | None = None) -> pd.DataFrame:
    """사이드바에서 사용할 중복 없는 역 선택 목록을 만든다."""

    mask = reference["city"].astype(str).eq(str(city))
    if line is not None:
        mask &= reference["line"].astype(str).eq(str(line))
    result = (
        reference.loc[mask, ["city", "line", "station_code", "station_name", "station_id"]]
        .drop_duplicates("station_id")
        .sort_values(["station_name", "line", "station_code"])
        .reset_index(drop=True)
    )
    result["label"] = (
        result["station_name"].astype(str)
        + " · "
        + result["line"].astype(str)
        + " ("
        + result["station_code"].astype(str)
        + ")"
    )
    return result


def build_hourly_profile(
    reference: pd.DataFrame,
    *,
    station_id: str,
    weekday: Any,
) -> pd.DataFrame:
    """선택 역·요일의 06~24시 차트용 데이터를 만든다."""

    weekday_num = normalize_weekday(weekday)
    profile = reference.loc[
        reference["station_id"].astype(str).eq(str(station_id))
        & reference["weekday_num"].eq(weekday_num),
        ["hour", "median_passenger_volume", "congestion_score", "congestion_level"],
    ].copy()
    if profile.empty:
        raise DataNotFoundError("선택한 역과 요일의 시간대 그래프 데이터가 없습니다.")
    profile = profile.sort_values("hour").drop_duplicates("hour")
    profile["시간"] = profile["hour"].map(lambda hour: f"{int(hour):02d}:00")
    profile["혼잡도 점수"] = profile["congestion_score"].astype(float)
    profile["승하차 수요 중앙값"] = profile["median_passenger_volume"].round().astype(int)
    profile["혼잡 단계"] = profile["congestion_level"].astype(str)
    return profile


def format_time_difference(delta: int) -> str:
    if delta == 0:
        return "현재 시간"
    return f"{abs(delta)}시간 {'뒤' if delta > 0 else '앞'}"


def weekday_label(value: Any) -> str:
    return f"{WEEKDAYS[normalize_weekday(value)]}요일"


def level_style(level: str) -> dict[str, str]:
    return LEVEL_STYLES.get(
        str(level),
        {"emoji": "🚇", "color": "#7F8C96", "message": "예상 혼잡도를 확인해 주세요."},
    )
