"""천만 영화가 다른 영화를 죽일까? — KOBIS 주간 박스오피스 대시보드

실행: streamlit run app.py
데이터: data/weekly_summary.csv, data/weekly_clean.csv (clean_weekly.py 결과)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st

st.set_page_config(page_title="천만 영화와 극장 시장", page_icon="🎬", layout="wide")

YEAR_COLORS = {2019: "#9AA0A6", 2023: "#8AB4F8", 2024: "#4C8BF5", 2025: "#1A56C4", 2026: "#E8710A"}
ORANGE, BLUE, GRAY = "#E8710A", "#1A56C4", "#9AA0A6"
CONTROLS = "C(month) + is_holiday + is_public_holiday + is_christmas"
HERE = Path(__file__).parent  # app.py 위치 기준 경로 (배포 서버에서도 똑같이 찾게)


# ---------------- 데이터 ----------------
@st.cache_data
def load():
    s = pd.read_csv(HERE / "data" / "weekly_summary.csv", parse_dates=["weekEnd"])
    d = pd.read_csv(HERE / "data" / "weekly_clean.csv", parse_dates=["weekEnd", "openDt"])
    for c in ["total_aud_est", "top1_aud", "others_aud"]:
        s[c + "_만"] = s[c] / 1e4
    for c in ["is_holiday", "is_public_holiday", "is_christmas"]:
        s[c] = s[c].astype(int)
    return s, d


@st.cache_data
def film_sizes(d):
    """개봉 연도별 영화 최종 관객 규모 (재개봉·특별상영 제외)"""
    f = d[~d["is_rerelease"] & ~d["is_event"]].groupby("movieCd").agg(
        movieNm=("movieNm", "first"), openDt=("openDt", "first"), final_aud=("audiAcc", "max"),
        is_korean=("is_korean", "first"))
    f = f[f["openDt"].dt.year.isin(YEAR_COLORS)]
    f["open_year"] = f["openDt"].dt.year
    f["규모"] = pd.cut(f["final_aud"], [0, 1e6, 5e6, 1e7, 1e9],
                     labels=["100만 미만", "100~500만 (허리)", "500~1000만", "1000만 이상"])
    return f


def regress(s, controls):
    """나머지 관객 = a + b × 1위 관객 (+ 통제). b와 95% 신뢰구간, 산점도용 값을 돌려준다."""
    years = s["year"].nunique()
    rhs = CONTROLS + (" + C(year)" if years > 1 else "") if controls else "1"
    m = smf.ols(f"others_aud_만 ~ top1_aud_만 + {rhs}", s).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    b, (lo, hi) = m.params["top1_aud_만"], m.conf_int().loc["top1_aud_만"]
    if controls:
        # 통제변수 효과를 뺀 값끼리 그리면 기울기가 정확히 b가 된다 (부분회귀 그림)
        x = smf.ols(f"top1_aud_만 ~ {rhs}", s).fit().resid + s["top1_aud_만"].mean()
        y = smf.ols(f"others_aud_만 ~ {rhs}", s).fit().resid + s["others_aud_만"].mean()
    else:
        x, y = s["top1_aud_만"], s["others_aud_만"]
    return b, lo, hi, m.pvalues["top1_aud_만"], x, y


def window(s, d, name, before=4, after=8):
    """대작 개봉 전 4주 ~ 후 8주. 나머지 = 전체 − 대작 − (대작이 아니면) 그 주 1위"""
    f = d[d["movieNm"] == name]
    od = f["openDt"].dropna().iloc[0]
    first = od + pd.Timedelta(days=6 - od.weekday())
    w = pd.DataFrame({"weekEnd": pd.date_range(first - pd.Timedelta(weeks=before),
                                               first + pd.Timedelta(weeks=after), freq="7D")})
    w["rel_week"] = range(-before, after + 1)
    w = w.merge(f[["weekEnd", "audiCnt"]], on="weekEnd", how="left").fillna({"audiCnt": 0})
    w = w.merge(s[["weekEnd", "total_aud_est", "top1_name", "top1_aud"]], on="weekEnd", how="left")
    w["others"] = w["total_aud_est"] - w["audiCnt"] - np.where(w["top1_name"] == name, 0, w["top1_aud"])
    return w, od


s_all, d_all = load()

# ---------------- 사이드바 (위젯) ----------------
st.sidebar.header("조건 선택")
years = st.sidebar.multiselect("비교할 연도", list(YEAR_COLORS), default=list(YEAR_COLORS))
controls = st.sidebar.toggle("계절·명절·연도 효과 통제", value=True,
                             help="끄면 명절·방학에 다 같이 관객이 느는 효과가 섞인다")
LAST_WEEK = s_all["weekEnd"].max()
st.sidebar.caption(f"2020~2022년은 코로나로 극장 상황이 비정상이라 제외. 2026년은 {LAST_WEEK:%m월 %d일} 주까지.")
if not years:
    st.warning("연도를 하나 이상 골라주세요.")
    st.stop()

s = s_all[s_all["year"].isin(years)].copy()
d = d_all[d_all["year"].isin(years)].copy()
sizes = film_sizes(d_all)
sizes = sizes[sizes["open_year"].isin(years)]

# ---------------- 헤더 + 핵심 숫자 ----------------
st.title("천만 영화가 다른 영화를 죽일까?")
st.markdown("2026년 극장 관객은 늘었는데 극장은 문을 닫는다. 흔히 대작 쏠림을 탓하는데, "
            f"KOBIS 주간 박스오피스 {len(s_all)}주로 그 말이 맞는지 확인했다.")

b, lo, hi, p, x, y = regress(s, controls)
cr3 = s.groupby("year")["top3_share"].mean()
mid = sizes[sizes["규모"] == "100~500만 (허리)"].groupby("open_year").size()
if len(mid) > 1:
    mid = mid.drop(2026, errors="ignore")  # 2026년은 10월까지라 연간 편수 비교에서 뺀다

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"상위 3편 관객 점유율 ({cr3.index.max()})", f"{cr3.iloc[-1]:.1%}",
          f"{(cr3.iloc[-1] - cr3.iloc[0]) * 100:+.1f}%p vs {cr3.index.min()}" if len(cr3) > 1 else None)
k2.metric("잠식 계수 b", f"{b:.2f}", f"95% 구간 {lo:.2f} ~ {hi:.2f}", delta_color="off")
k3.metric("1위가 10만 명 늘 때", f"나머지 {10 * b:+.1f}만", f"전체 {10 * (1 + b):+.1f}만 명", delta_color="off")
if len(mid):
    k4.metric(f"허리 영화 수 ({mid.index.max()} 개봉)", f"{mid.iloc[-1]}편",
              f"{mid.iloc[-1] - mid.iloc[0]:+d}편 vs {mid.index.min()}" if len(mid) > 1 else None,
              delta_color="inverse")

# ---------------- ① 집중도 ----------------
st.header("① 관객은 정말 한 영화에 더 몰렸나")
wk = s.groupby("year")["total_aud_est_만"].mean()
if 2019 in wk.index and len(wk) > 1:
    st.metric(f"주 평균 전체 관객 ({wk.index.max()})", f"{wk.iloc[-1]:.0f}만 명",
              f"2019년의 {wk.iloc[-1] / wk.loc[2019]:.0%} 수준", delta_color="off")
share = s.groupby("year")[["top1_share", "top3_share"]].mean().reset_index()
share = share.melt("year", var_name="지표", value_name="점유율")
share["지표"] = share["지표"].map({"top1_share": "1위 영화", "top3_share": "상위 3편"})
fig = px.bar(share, x="year", y="점유율", color="지표", barmode="group", text_auto=".0%",
             color_discrete_map={"1위 영화": ORANGE, "상위 3편": BLUE})
fig.update_layout(xaxis=dict(type="category", title=""), yaxis=dict(tickformat=".0%", title="주 평균 관객 점유율"),
                  legend_title="", height=380)
st.plotly_chart(fig, use_container_width=True)
st.caption("시장 집중도(CR1·CR3). 점유율의 분모는 매출 점유율로 역산한 그 주 전체 관객 추정치.")

# ---------------- ② 잠식 ----------------
st.header("② 대작이 잘되면 다른 영화 관객이 줄었나")
c1, c2 = st.columns([3, 2])
with c1:
    pts = pd.DataFrame({"x": x, "y": y, "year": s["year"].astype(str), "영화": s["top1_name"],
                        "주": s["weekEnd"].dt.strftime("%Y-%m-%d")})
    fig = px.scatter(pts, x="x", y="y", color="year", hover_data=["영화", "주"],
                     color_discrete_map={str(k): v for k, v in YEAR_COLORS.items()})
    xs = np.linspace(pts["x"].min(), pts["x"].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=pts["y"].mean() + b * (xs - pts["x"].mean()), mode="lines",
                             line=dict(color="black", width=2), name=f"기울기 b = {b:.2f}"))
    label = " (계절·연도 효과 제거)" if controls else ""
    fig.update_layout(xaxis_title=f"1위 영화 관객 (만 명){label}", yaxis_title=f"1위 제외 관객 (만 명){label}",
                      legend_title="", height=430)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    verdict = ("**시장 확대**: 1위가 잘돼도 나머지가 줄지 않음" if lo > 0 or (lo <= 0 <= hi and b >= 0)
               else "**완전 잠식**: 1위가 는 만큼 이상 나머지가 줆" if b <= -1
               else "**일부 잠식, 전체는 증가**: 다른 영화 관객을 조금 가져가지만 그보다 많은 새 관객을 데려옴")
    st.markdown(f"""
점 하나 = 한 주. 검은 선의 기울기가 잠식 계수 **b**다.

- b ≥ 0: 시장 확대
- -1 < b < 0: 일부 잠식, 전체는 증가
- b ≤ -1: 완전 잠식

**지금 결과: b = {b:.2f}** (p = {p:.3f})
{verdict}

왼쪽 사이드바에서 **통제를 꺼 보면** 기울기가 양수로 바뀐다. 명절·방학엔 1위도 나머지도 다 같이
많이 보기 때문에 생기는 착시다.
""")

# ---------------- ③ 개봉 전후 ----------------
st.header("③ 천만 영화가 개봉하면 무슨 일이 생기나")
big = (d_all[(d_all["rank"] == 1) & d_all["openDt"].notna()].groupby("movieNm")["audiAcc"].max()
       .sort_values(ascending=False))
big = big[big >= 7e6].index.tolist()
default = big.index("왕과 사는 남자") if "왕과 사는 남자" in big else 0
name = st.selectbox("영화 선택 (누적 700만 명 이상)", big, index=default)
w, od = window(s_all, d_all, name)
fig = go.Figure()
fig.add_bar(x=w["rel_week"], y=w["audiCnt"] / 1e4, name=name, marker_color=ORANGE, opacity=0.65)
fig.add_scatter(x=w["rel_week"], y=w["others"] / 1e4, name="대작·그 주 1위 제외", mode="lines+markers",
                line=dict(color=BLUE, width=3))
fig.add_scatter(x=w["rel_week"], y=w["total_aud_est"] / 1e4, name="전체", mode="lines",
                line=dict(color=GRAY, dash="dash"))
fig.add_vline(x=-0.5, line_dash="dot")
fig.update_layout(xaxis=dict(title=f"개봉 주 기준 몇 주째 (0 = 개봉 주, {od:%Y-%m-%d} 개봉)", dtick=1),
                  yaxis_title="관객 (만 명)", legend_title="", height=400)
st.plotly_chart(fig, use_container_width=True)
pre, post = w["rel_week"] < 0, w["rel_week"].between(0, 3)
m1, m2 = st.columns(2)
m1.metric("전체 관객 (개봉 후 4주 vs 전 4주)", f"{w.loc[post, 'total_aud_est'].mean() / 1e4:.0f}만 명/주",
          f"{w.loc[post, 'total_aud_est'].mean() / w.loc[pre, 'total_aud_est'].mean() - 1:+.0%}")
m2.metric("나머지 영화 관객 (개봉 후 4주 vs 전 4주)", f"{w.loc[post, 'others'].mean() / 1e4:.0f}만 명/주",
          f"{w.loc[post, 'others'].mean() / w.loc[pre, 'others'].mean() - 1:+.0%}")
st.caption("개봉 첫 주에 나머지 영화 관객이 잠깐 꺼졌다가 1~2주 안에 회복되는지 볼 것. "
           "계절 효과가 섞여 있어서 결론은 ②의 회귀로 낸다.")

# ---------------- ④ 스크린 ----------------
st.header("④ 극장이 대작에 상영을 몰아줬나")
g = d.groupby("weekEnd")
d["관객 점유율"] = d["audiCnt"] / g["audiCnt"].transform("sum")
d["상영 점유율"] = d["showCnt"] / g["showCnt"].transform("sum")
gap = d.groupby("rank")[["관객 점유율", "상영 점유율"]].mean()
gap["차이"] = gap["상영 점유율"] - gap["관객 점유율"]
fig = px.bar(gap.reset_index(), x="rank", y="차이", text_auto="+.1%",
             color=np.where(gap["차이"] > 0, "관객보다 상영을 더 받음", "관객보다 상영을 덜 받음"),
             color_discrete_map={"관객보다 상영을 더 받음": BLUE, "관객보다 상영을 덜 받음": ORANGE})
fig.update_layout(xaxis=dict(title="주간 순위", dtick=1), yaxis=dict(title="상영 점유율 − 관객 점유율", tickformat="+.0%"),
                  legend_title="", height=380)
st.plotly_chart(fig, use_container_width=True)
st.caption("상위 10편 안에서 계산 (11위 이하 상영 횟수는 데이터에 없음). 1위는 수요보다 상영을 덜 받았다. "
           "상영 횟수 자체가 많다는 비판(독과점)과 수요보다 많이 줬다는 비판(몰아주기)은 다른 문제다.")

# ---------------- ⑤ 허리 영화 ----------------
st.header("⑤ 그럼 무엇이 줄었나: 허리 영화")
tbl = sizes.groupby(["open_year", "규모"], observed=False).size().reset_index(name="영화 수")
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(tbl[tbl["규모"] != "100만 미만"], x="open_year", y="영화 수", color="규모", barmode="group",
                 text_auto=True, color_discrete_map={"100~500만 (허리)": BLUE, "500~1000만": GRAY, "1000만 이상": ORANGE})
    fig.update_layout(title="관객 규모별 영화 수", xaxis=dict(type="category", title="개봉 연도"),
                      legend_title="최종 관객", height=380)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    midf = sizes[sizes["규모"] == "100~500만 (허리)"].copy()
    midf["국적"] = np.where(midf["is_korean"] == True, "한국영화", "외화")
    kt = midf.groupby(["open_year", "국적"]).size().reset_index(name="영화 수")
    fig = px.bar(kt, x="open_year", y="영화 수", color="국적", barmode="stack", text_auto=True,
                 color_discrete_map={"한국영화": BLUE, "외화": GRAY}, category_orders={"국적": ["한국영화", "외화"]})
    fig.update_layout(title="허리 영화(100~500만) 중 한국영화와 외화", xaxis=dict(type="category", title="개봉 연도"),
                      legend_title="", height=380)
    st.plotly_chart(fig, use_container_width=True)

rr = s.groupby("year")["rereleases"].mean()
r1, r2 = st.columns(2)
r1.metric(f"상위 10편에 든 재개봉작 ({rr.index.max()}, 주 평균)", f"{rr.iloc[-1]:.2f}편",
          f"{rr.iloc[-1] - rr.iloc[0]:+.2f}편 vs {rr.index.min()}" if len(rr) > 1 else None, delta_color="off")
nr = s.groupby("year")["new_releases"].mean()
r2.metric(f"상위 10편에 든 신작 ({nr.index.max()}, 주 평균)", f"{nr.iloc[-1]:.2f}편",
          f"{nr.iloc[-1] - nr.iloc[0]:+.2f}편 vs {nr.index.min()}" if len(nr) > 1 else None, delta_color="off")
st.caption("허리 영화 감소는 주로 한국영화에서 일어났고, 신작이 줄어든 자리를 재개봉작이 채우고 있다. "
           "최종 관객 = 데이터 안에서 확인한 누적 관객 최댓값. 2026년 하반기 개봉작은 아직 상영 중이라 작게 잡힐 수 있다.")

# ---------------- 결론 ----------------
st.header("결론")
st.markdown("""
- 2026년 관객은 실제로 소수 영화에 더 몰렸다.
- 하지만 대작은 다른 영화 관객을 조금 가져가는 대신 훨씬 많은 새 관객을 데려왔고, 극장이 수요보다 상영을 몰아준 증거도 없다.
- 대신 100만~500만 관객의 허리 영화가 2019년의 절반 수준으로 줄었고, 감소는 주로 한국영화에서 일어났다. 신작이 줄어든 자리는 재개봉작이 채우고 있다. 대작이 없는 주에 관객을 불러올 영화가 부족하다.
- 한계: 상관관계이며 인과는 아니다. 전체 관객은 매출 점유율로 역산한 추정치(영화별 평균 티켓 가격이 비슷하다는 가정).
""")
st.caption(f"데이터: 영화진흥위원회 KOBIS 오픈API 주간 박스오피스(2019, 2023~{LAST_WEEK:%Y.%m.%d} 주), 영화 상세정보.")
