# weather_module.py
# -----------------
# OpenWeather(키 필요) → 실패/없음 시 Open-Meteo 폴백
# 날짜가 있으면 해당 날짜 기준(OWM 5일 / OM 16일), 없거나 범위 초과면 조회시점 현재 날씨 표시
# 시간 표시는 "조회 시점: ..."로 노출, 안내 문구는 연한 빨강(#FF6666)
# 나라명만 입력해도 대표도시로 자동 보정

import os
import datetime as dt
import requests
import streamlit as st

# ---------- 입력 보정(나라 → 대표도시) ----------
_COUNTRY_TO_CITY = {
    # 미국/영문
    "미국": "Washington", "usa": "Washington", "united states": "Washington", "us": "Washington",
    # 한국
    "한국": "Seoul", "대한민국": "Seoul", "south korea": "Seoul", "kr": "Seoul",
    # 일본
    "일본": "Tokyo", "japan": "Tokyo", "jp": "Tokyo",
    # 중국
    "중국": "Beijing", "china": "Beijing", "cn": "Beijing",
    # 영국
    "영국": "London", "uk": "London", "united kingdom": "London", "gb": "London",
    # 프랑스
    "프랑스": "Paris", "france": "Paris", "fr": "Paris",
    # 독일
    "독일": "Berlin", "germany": "Berlin", "de": "Berlin",
    # 이탈리아
    "이탈리아": "Rome", "italy": "Rome", "it": "Rome",
    # 스페인
    "스페인": "Madrid", "spain": "Madrid", "es": "Madrid",
    # 캐나다
    "캐나다": "Ottawa", "canada": "Ottawa", "ca": "Ottawa",
    # 호주
    "호주": "Canberra", "australia": "Canberra", "au": "Canberra",
}

def _normalize_place(text: str) -> str:
    if not text:
        return text
    key = text.strip().lower()
    return _COUNTRY_TO_CITY.get(key, text.strip())

# ---------- 안전한 OpenWeather 키 로딩 ----------
def _get_openweather_key():
    # 1) 환경변수
    key = os.getenv("OPENWEATHER_API_KEY")
    if key:
        return key
    # 2) secrets.toml 이 있을 때만 접근
    try:
        return st.secrets["OPENWEATHER_API_KEY"]  # type: ignore[index]
    except Exception:
        return None

OWM_KEY = _get_openweather_key()

# ---------- OpenWeather 유틸 ----------
def _owm_geocode(city: str):
    city = _normalize_place(city)
    r = requests.get(
        "https://api.openweathermap.org/geo/1.0/direct",
        params={"q": city, "limit": 1, "appid": OWM_KEY},
        timeout=10,
    )
    r.raise_for_status()
    d = r.json()
    return None if not d else {
        "lat": d[0]["lat"],
        "lon": d[0]["lon"],
        "name": d[0]["name"],
        "country": d[0]["country"],
    }

def _owm_current(lat: float, lon: float):
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric", "lang": "kr"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

def _owm_forecast(lat: float, lon: float):
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params={"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric", "lang": "kr"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

def _owm_pick_slot(fc_json: dict, day: dt.date):
    """3시간 간격 예보 중 day의 정오(12:00)와 가장 가까운 슬롯 선택"""
    slots = fc_json.get("list", [])
    if not slots:
        return None
    target = dt.datetime.combine(day, dt.time(12, 0))
    return min(slots, key=lambda it: abs(dt.datetime.utcfromtimestamp(it["dt"]) - target))

# ---------- Open-Meteo 유틸 ----------
def _om_geocode(city: str):
    city = _normalize_place(city)
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "ko"},
        timeout=10,
    )
    r.raise_for_status()
    d = r.json()
    if not d.get("results"):
        return None
    g = d["results"][0]
    return {
        "lat": g["latitude"],
        "lon": g["longitude"],
        "name": g["name"],
        "country": g["country_code"],
    }

def _om_forecast(lat: float, lon: float, start: dt.date, end: dt.date):
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "current_weather": "true",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

# ---------- 공통 렌더러 ----------
def render_weather(city: str, day: dt.date | None):
    if not city:
        st.info("도착지를 입력하세요.")
        return

    today = dt.date.today()

    # ===== OpenWeather 우선 =====
    if OWM_KEY:
        try:
            geo = _owm_geocode(city)
            if not geo:
                st.warning(f"도시를 찾지 못했습니다: {city}")
                return

            lat, lon = geo["lat"], geo["lon"]
            st.write(f"🌍 {geo['name']}, {geo['country']} (OpenWeather)")

            # 날짜 있고 OWM 예보 범위(5일) 이내 → 예보
            if day and today <= day <= today + dt.timedelta(days=5):
                fc = _owm_forecast(lat, lon)
                slot = _owm_pick_slot(fc, day)
                if not slot:
                    st.warning("예보 데이터를 찾을 수 없습니다.")
                else:
                    main, w, wind = slot["main"], slot["weather"][0], slot["wind"]
                    cols = st.columns(4)
                    cols[0].metric("예보기온(°C)", f"{main.get('temp', 0):.1f}")
                    cols[1].metric("체감(°C)", f"{main.get('feels_like', 0):.1f}")
                    cols[2].metric("습도(%)", main.get("humidity", "—"))
                    cols[3].metric("풍속(m/s)", f"{wind.get('speed', 0):.1f}")
                    st.write(f"설명: {w['description'].capitalize()}")
                    st.image(f"https://openweathermap.org/img/wn/{w['icon']}@2x.png", width=64)
                    st.caption(f"선택 날짜({day}) 기준 예보입니다.")
            else:
                # 날짜 없음/범위 초과 → 현재 날씨
                cur = _owm_current(lat, lon)
                w = cur["weather"][0]
                cols = st.columns(3)
                cols[0].metric("현재기온(°C)", f"{cur['main'].get('temp', 0):.1f}")
                cols[1].metric("체감(°C)", f"{cur['main'].get('feels_like', 0):.1f}")
                cols[2].metric("습도(%)", cur["main"].get("humidity", "—"))
                st.write(f"풍속(m/s): {cur.get('wind', {}).get('speed', 0):.1f}")
                st.write("조회 시점: 현재(실시간)")
                st.write(f"설명: {w['description'].capitalize()}")
                st.image(f"https://openweathermap.org/img/wn/{w['icon']}@2x.png", width=64)
                st.markdown(
                    "<span style='color:#FF6666'>※ 선택한 날짜가 없거나 지원 범위를 벗어나 "
                    "조회시점 기준 현재 날씨를 표시합니다.</span>",
                    unsafe_allow_html=True,
                )
            return
        except Exception as e:
            st.warning(f"⚠️ OpenWeather 실패 → Open-Meteo로 전환: {e}")

    # ===== Fallback: Open-Meteo =====
    try:
        geo = _om_geocode(city)
        if not geo:
            st.warning(f"도시를 찾지 못했습니다: {city}")
            return

        st.write(f"🌍 {geo['name']}, {geo['country']} (Open-Meteo)")

        if day and today <= day <= today + dt.timedelta(days=16):
            data = _om_forecast(geo["lat"], geo["lon"], day, day)
            d = data.get("daily", {})
            if not d:
                st.warning("예보 데이터를 찾을 수 없습니다.")
                return
            cols = st.columns(4)
            cols[0].metric("최고(°C)", d["temperature_2m_max"][0])
            cols[1].metric("최저(°C)", d["temperature_2m_min"][0])
            cols[2].metric("강수량(mm)", d["precipitation_sum"][0])
            cols[3].metric("최대풍속(m/s)", d["windspeed_10m_max"][0])
            st.caption(f"선택 날짜({day}) 기준 예보입니다.")
        else:
            data = _om_forecast(geo["lat"], geo["lon"], today, today)
            cur = data.get("current_weather", {})
            if not cur:
                st.warning("날씨 데이터를 찾을 수 없습니다.")
                return
            cols = st.columns(2)
            cols[0].metric("현재기온(°C)", cur["temperature"])
            cols[1].metric("풍속(m/s)", cur["windspeed"])
            # 시간은 metric 대신 일반 텍스트 + '조회 시점' 라벨
            st.write(f"조회 시점: {cur['time']}")
            st.markdown(
                "<span style='color:#FF6666'>※ 선택한 날짜가 없거나 지원 범위를 벗어나 "
                "조회시점 기준 현재 날씨를 표시합니다.</span>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Open-Meteo 조회 실패: {e}")
