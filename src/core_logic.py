from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_PATH = ROOT / "data" / "processed" / "congestion_reference.parquet"
DEFAULT_ROUTE_CANDIDATES = (
    ROOT / "data" / "processed" / "routes.parquet",
    ROOT / "data" / "processed" / "routes.csv",
    ROOT / "data" / "routes.parquet",
    ROOT / "data" / "routes.csv",
)
SUPPORTED_CITIES = ("서울", "인천", "대전", "대구")
WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")
WEEKDAY_ALIASES = {
    **{name: index for index, name in enumerate(WEEKDAYS)},
    **{f"{name}요일": index for index, name in enumerate(WEEKDAYS)},
    **{
        name: index
        for index, name in enumerate(
            ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        )
    },
}


class CoreLogicError(Exception):
    """천국철 핵심 로직에서 사용자에게 안내할 수 있는 기본 오류."""


class InputValidationError(CoreLogicError):
    """지원 범위를 벗어나거나 서로 모순되는 입력."""


class DataNotFoundError(CoreLogicError):
    """입력은 유효하지만 일치하는 분석 데이터가 없는 경우."""


@lru_cache(maxsize=2)
def _load_reference_cached(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        raise DataNotFoundError(f"혼잡도 참조 데이터가 없습니다: {path}")
    return pd.read_parquet(path)


def load_reference(path: str | Path = DEFAULT_REFERENCE_PATH) -> pd.DataFrame:
    """혼잡도 참조 데이터를 읽는다. 동일 경로는 프로세스 안에서 캐시한다."""

    return _load_reference_cached(str(Path(path).resolve())).copy()


def normalize_weekday(value: Any) -> int:
    """날짜, 요일 번호(월=0), 한글/영문 요일을 0~6으로 통일한다."""

    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value <= 6:
            return value
        raise InputValidationError("요일 번호는 월요일 0부터 일요일 6 사이여야 합니다.")
    if hasattr(value, "weekday") and not isinstance(value, str):
        return int(value.weekday())
    text = str(value).strip().lower()
    if text in WEEKDAY_ALIASES:
        return WEEKDAY_ALIASES[text]
    try:
        parsed = pd.Timestamp(text)
    except (TypeError, ValueError):
        parsed = pd.NaT
    if not pd.isna(parsed):
        return int(parsed.weekday())
    raise InputValidationError(
        "요일은 0~6, 월~일, 월요일~일요일 또는 YYYY-MM-DD 날짜로 입력해 주세요."
    )


def normalize_hour(value: Any) -> int:
    """8, '08:00', '8시' 입력을 시간 구간 시작값 8로 통일한다."""

    if isinstance(value, bool):
        raise InputValidationError("시간은 06:00부터 23:00 사이로 입력해 주세요.")
    if isinstance(value, int):
        hour = value
    else:
        text = str(value).strip().lower().replace("시", "")
        if ":" in text:
            parts = text.split(":")
            if len(parts) != 2 or parts[1] not in {"0", "00"}:
                raise InputValidationError("현재 데이터는 정각 기준 1시간 단위만 지원합니다.")
            text = parts[0]
        try:
            hour = int(text)
        except (TypeError, ValueError) as error:
            raise InputValidationError("시간은 06:00부터 23:00 사이로 입력해 주세요.") from error
    if not 6 <= hour <= 23:
        raise InputValidationError("공통 분석 시간은 06:00부터 23:00 시작 구간까지입니다.")
    return hour


def score_to_level(score: float) -> str:
    """0~100 혼잡도 점수를 확정된 다섯 단계로 변환한다."""

    if pd.isna(score) or not 0 <= float(score) <= 100:
        raise InputValidationError("혼잡도 점수는 0부터 100 사이여야 합니다.")
    if score <= 20:
        return "매우 여유"
    if score <= 40:
        return "여유"
    if score <= 60:
        return "보통"
    if score <= 80:
        return "혼잡"
    return "매우 혼잡"


def calculate_congestion(rows: pd.DataFrame) -> dict[str, Any]:
    """조회된 한 개 이상의 역·시간 행을 대표 혼잡도로 집계한다.

    여러 역의 상대 점수는 이상치 영향을 줄이기 위해 중앙값으로 집계한다.
    승하차 수요는 각 행의 중앙값 수요 합계를 함께 제공한다.
    """

    required = {"congestion_score", "median_passenger_volume", "sample_days"}
    missing = required.difference(rows.columns)
    if missing:
        raise InputValidationError(f"혼잡도 계산에 필요한 컬럼이 없습니다: {sorted(missing)}")
    if rows.empty:
        raise DataNotFoundError("혼잡도를 계산할 데이터가 없습니다.")
    score = round(float(rows["congestion_score"].median()), 1)
    return {
        "congestion_score": score,
        "congestion_level": score_to_level(score),
        "median_passenger_volume": int(round(rows["median_passenger_volume"].sum())),
        "matched_station_hours": int(len(rows)),
        "minimum_sample_days": int(rows["sample_days"].min()),
        "basis": "동일 역·요일의 18개 시간대 승하차 수요 중앙값 상대 백분위",
    }


def _validate_and_filter(
    reference: pd.DataFrame,
    *,
    city: str,
    weekday: Any,
    hour: Any,
    line: str | None = None,
    station_name: str | None = None,
    station_code: str | int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    city = str(city).strip()
    if city not in SUPPORTED_CITIES:
        raise InputValidationError(
            f"지원하지 않는 지역입니다: {city}. 가능한 지역: {', '.join(SUPPORTED_CITIES)}"
        )
    weekday_num = normalize_weekday(weekday)
    hour_num = normalize_hour(hour)
    mask = (
        reference["city"].astype(str).eq(city)
        & reference["weekday_num"].eq(weekday_num)
        & reference["hour"].eq(hour_num)
    )
    conditions: dict[str, Any] = {
        "city": city,
        "weekday_num": weekday_num,
        "weekday": WEEKDAYS[weekday_num],
        "hour": hour_num,
        "time_range": f"{hour_num:02d}:00~{hour_num + 1:02d}:00",
    }
    if line is not None:
        line = str(line).strip()
        city_lines = sorted(reference.loc[reference["city"].astype(str).eq(city), "line"].astype(str).unique())
        if line not in city_lines:
            raise InputValidationError(
                f"{city}에 없는 호선입니다: {line}. 가능한 호선: {', '.join(city_lines)}"
            )
        mask &= reference["line"].astype(str).eq(line)
        conditions["line"] = line
    if station_name is not None and station_code is not None:
        raise InputValidationError("역명과 역번호 중 하나만 입력해 주세요.")
    if station_name is not None:
        station_name = str(station_name).strip()
        mask &= reference["station_name"].astype(str).eq(station_name)
        conditions["station_name"] = station_name
    if station_code is not None:
        station_code = str(station_code).strip()
        mask &= reference["station_code"].astype(str).eq(station_code)
        conditions["station_code"] = station_code
    result = reference.loc[mask].copy()
    if result.empty:
        detail = ", ".join(f"{key}={value}" for key, value in conditions.items() if key != "time_range")
        raise DataNotFoundError(f"조건에 맞는 혼잡도 데이터가 없습니다: {detail}")
    return result.sort_values(["line", "station_code"]), conditions


def query_congestion(
    *,
    city: str,
    weekday: Any,
    hour: Any,
    line: str | None = None,
    station_name: str | None = None,
    station_code: str | int | None = None,
    reference: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """지역·요일·시간·호선·역 조건에 맞는 혼잡도를 조회한다."""

    source = load_reference() if reference is None else reference
    rows, conditions = _validate_and_filter(
        source,
        city=city,
        weekday=weekday,
        hour=hour,
        line=line,
        station_name=station_name,
        station_code=station_code,
    )
    record_columns = [
        "line",
        "station_code",
        "station_name",
        "station_id",
        "median_passenger_volume",
        "sample_days",
        "congestion_score",
        "congestion_level",
    ]
    return {
        "status": "ok",
        "conditions": conditions,
        "summary": calculate_congestion(rows),
        "records": rows[record_columns].to_dict(orient="records"),
    }


def _single_station_rows(
    reference: pd.DataFrame,
    *,
    city: str,
    weekday: Any,
    hour: Any,
    line: str | None,
    station_name: str | None,
    station_code: str | int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if station_name is None and station_code is None:
        raise InputValidationError("시간대 추천에는 역명 또는 역번호가 필요합니다.")
    rows, conditions = _validate_and_filter(
        reference,
        city=city,
        weekday=weekday,
        hour=hour,
        line=line,
        station_name=station_name,
        station_code=station_code,
    )
    station_ids = rows["station_id"].astype(str).unique()
    if len(station_ids) != 1:
        lines = ", ".join(sorted(rows["line"].astype(str).unique()))
        raise InputValidationError(
            f"동일한 역명이 여러 노선에 있습니다. 호선을 함께 입력해 주세요: {lines}"
        )
    return rows, conditions


def recommend_adjacent_times(
    *,
    city: str,
    weekday: Any,
    hour: Any,
    line: str | None = None,
    station_name: str | None = None,
    station_code: str | int | None = None,
    window_hours: int = 2,
    limit: int = 3,
    minimum_improvement_points: float = 1.0,
    reference: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """같은 역의 선택 시간 전후를 비교해 더 낮은 혼잡도 후보를 추천한다."""

    if not isinstance(window_hours, int) or not 1 <= window_hours <= 6:
        raise InputValidationError("비교 범위는 1시간부터 6시간 사이의 정수여야 합니다.")
    if not isinstance(limit, int) or not 1 <= limit <= 10:
        raise InputValidationError("추천 개수는 1개부터 10개 사이의 정수여야 합니다.")
    source = load_reference() if reference is None else reference
    selected_rows, conditions = _single_station_rows(
        source,
        city=city,
        weekday=weekday,
        hour=hour,
        line=line,
        station_name=station_name,
        station_code=station_code,
    )
    current = calculate_congestion(selected_rows)
    selected_hour = conditions["hour"]
    station_id = str(selected_rows.iloc[0]["station_id"])
    candidate_rows = source.loc[
        source["station_id"].astype(str).eq(station_id)
        & source["weekday_num"].eq(conditions["weekday_num"])
        & source["hour"].between(max(6, selected_hour - window_hours), min(23, selected_hour + window_hours))
        & source["hour"].ne(selected_hour)
    ].copy()
    candidates: list[dict[str, Any]] = []
    for candidate_hour, group in candidate_rows.groupby("hour", observed=True):
        summary = calculate_congestion(group)
        improvement = round(current["congestion_score"] - summary["congestion_score"], 1)
        if improvement < minimum_improvement_points:
            continue
        delta = int(candidate_hour) - selected_hour
        direction = "뒤" if delta > 0 else "앞"
        candidates.append(
            {
                "hour": int(candidate_hour),
                "time_range": f"{int(candidate_hour):02d}:00~{int(candidate_hour) + 1:02d}:00",
                "time_difference_hours": delta,
                **summary,
                "improvement_points": improvement,
                "reason": (
                    f"{abs(delta)}시간 {direction}에 출발하면 예상 혼잡도 점수가 "
                    f"{improvement:.1f}점 낮습니다."
                ),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["congestion_score"],
            abs(item["time_difference_hours"]),
            item["hour"],
        )
    )
    recommendations = candidates[:limit]
    if recommendations:
        decision = "change_time"
        message = recommendations[0]["reason"]
    else:
        decision = "keep_current"
        message = "비교 범위 안에 혼잡도 점수가 유의미하게 낮은 시간대가 없어 현재 시간을 유지해도 좋습니다."
    return {
        "status": "ok",
        "conditions": conditions,
        "current": current,
        "decision": decision,
        "message": message,
        "recommendations": recommendations,
    }


def _read_route_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp949")
    raise InputValidationError("경로 데이터는 CSV 또는 Parquet 파일이어야 합니다.")


def load_route_data(path: str | Path | None = None) -> tuple[pd.DataFrame | None, Path | None]:
    """명시한 경로 또는 기본 후보 위치에서 경로 데이터를 찾는다."""

    if path is not None:
        route_path = Path(path)
        if not route_path.exists():
            raise DataNotFoundError(f"경로 데이터 파일이 없습니다: {route_path}")
        return _read_route_data(route_path), route_path
    for candidate in DEFAULT_ROUTE_CANDIDATES:
        if candidate.exists():
            return _read_route_data(candidate), candidate
    return None, None


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise InputValidationError(f"경로 목록 JSON 형식이 잘못되었습니다: {text}") from error
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(";") if item.strip()]


def recommend_alternative_routes(
    *,
    city: str,
    origin_station: str,
    destination_station: str,
    weekday: Any,
    hour: Any,
    route_data: pd.DataFrame | None = None,
    route_path: str | Path | None = None,
    reference: pd.DataFrame | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """경로 후보의 혼잡도·시간·환승을 비교한다.

    경로 데이터 필수 컬럼은 route_id, city, origin_station, destination_station,
    duration_minutes, transfers이며, 혼잡도 계산을 위해 station_ids(JSON 배열 또는
    세미콜론 구분)도 필요하다. 경로 파일이 없으면 unavailable 상태를 반환한다.
    """

    city = str(city).strip()
    if city not in SUPPORTED_CITIES:
        raise InputValidationError(f"지원하지 않는 지역입니다: {city}")
    weekday_num = normalize_weekday(weekday)
    hour_num = normalize_hour(hour)
    origin_station = str(origin_station).strip()
    destination_station = str(destination_station).strip()
    if not origin_station or not destination_station:
        raise InputValidationError("출발역과 도착역을 모두 입력해 주세요.")
    if origin_station == destination_station:
        raise InputValidationError("출발역과 도착역은 서로 달라야 합니다.")
    if not isinstance(limit, int) or not 1 <= limit <= 10:
        raise InputValidationError("추천 개수는 1개부터 10개 사이의 정수여야 합니다.")
    source_routes, used_path = (route_data, None) if route_data is not None else load_route_data(route_path)
    if source_routes is None:
        return {
            "status": "unavailable",
            "reason": "공식 역 연결·소요시간 경로 데이터가 아직 없습니다.",
            "recommendations": [],
        }
    required = {
        "route_id",
        "city",
        "origin_station",
        "destination_station",
        "duration_minutes",
        "transfers",
        "station_ids",
    }
    missing = required.difference(source_routes.columns)
    if missing:
        raise InputValidationError(f"경로 데이터에 필요한 컬럼이 없습니다: {sorted(missing)}")
    routes = source_routes.loc[
        source_routes["city"].astype(str).eq(city)
        & source_routes["origin_station"].astype(str).eq(origin_station)
        & source_routes["destination_station"].astype(str).eq(destination_station)
    ].copy()
    if routes.empty:
        raise DataNotFoundError(f"{origin_station}에서 {destination_station}까지의 경로 후보가 없습니다.")
    congestion = load_reference() if reference is None else reference
    evaluated: list[dict[str, Any]] = []
    for _, route in routes.iterrows():
        try:
            duration = float(route["duration_minutes"])
            transfers_raw = float(route["transfers"])
        except (TypeError, ValueError) as error:
            raise InputValidationError("소요시간과 환승 횟수는 숫자여야 합니다.") from error
        if not math.isfinite(duration) or duration <= 0:
            raise InputValidationError("소요시간은 0보다 큰 유한한 숫자여야 합니다.")
        if not math.isfinite(transfers_raw) or transfers_raw < 0 or not transfers_raw.is_integer():
            raise InputValidationError("환승 횟수는 0 이상의 정수여야 합니다.")
        station_ids = _parse_list(route["station_ids"])
        if not station_ids:
            raise InputValidationError(f"경로 {route['route_id']}에 역 목록이 없습니다.")
        matched = congestion.loc[
            congestion["city"].astype(str).eq(city)
            & congestion["weekday_num"].eq(weekday_num)
            & congestion["hour"].eq(hour_num)
            & congestion["station_id"].astype(str).isin(station_ids)
        ]
        if matched.empty:
            continue
        summary = calculate_congestion(matched)
        evaluated.append(
            {
                "route_id": str(route["route_id"]),
                "route_name": str(route.get("route_name", route["route_id"])),
                "duration_minutes": duration,
                "transfers": int(transfers_raw),
                "matched_stations": int(matched["station_id"].nunique()),
                **summary,
            }
        )
    if not evaluated:
        raise DataNotFoundError("경로 후보와 일치하는 역별 혼잡도 데이터가 없습니다.")
    max_duration = max(item["duration_minutes"] for item in evaluated) or 1
    max_transfers = max(item["transfers"] for item in evaluated) or 1
    for item in evaluated:
        item["recommendation_cost"] = round(
            item["congestion_score"] * 0.6
            + item["duration_minutes"] / max_duration * 100 * 0.3
            + item["transfers"] / max_transfers * 100 * 0.1,
            1,
        )
        item["reason"] = (
            f"혼잡도 {item['congestion_score']:.1f}점, "
            f"약 {item['duration_minutes']:g}분, 환승 {item['transfers']}회"
        )
    evaluated.sort(key=lambda item: (item["recommendation_cost"], item["duration_minutes"]))
    return {
        "status": "ok",
        "route_source": str(used_path) if used_path else "provided_dataframe",
        "policy": "혼잡도 60% + 정규화 소요시간 30% + 환승 부담 10%",
        "recommendations": evaluated[:limit],
    }


def to_json(data: dict[str, Any]) -> str:
    """CLI와 테스트에서 동일한 결과를 재현하기 위한 JSON 직렬화."""

    return json.dumps(data, ensure_ascii=False, indent=2, default=str)
