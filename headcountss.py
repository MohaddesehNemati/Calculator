import streamlit as st
import pandas as pd
import math
import jdatetime
import matplotlib.pyplot as plt

# =========================
# Erlang C
# =========================
def erlang_c_probability_wait(a, n):
    if n <= a:
        return 1.0
    s = sum((a**k) / math.factorial(k) for k in range(n))
    pn = (a**n) / math.factorial(n) * (n / (n - a))
    return pn / (s + pn)

def erlang_c_service_level(a, n, aht_sec, t_sec):
    pw = erlang_c_probability_wait(a, n)
    mu = 1 / aht_sec
    expo = math.exp(-(n - a) * mu * t_sec)
    return 1 - pw * expo

def required_agents_erlang(volume_per_hour, aht_sec, sl_target=0.8, t_sec=20):
    lam = volume_per_hour / 3600.0
    mu = 1 / aht_sec
    a = lam / mu

    n = max(1, math.ceil(a))
    while True:
        sl = erlang_c_service_level(a, n, aht_sec, t_sec)
        if sl >= sl_target:
            return n, a, sl
        n += 1

# =========================
# Jalali helpers
# =========================
def normalize_date_str(x):
    return str(x).strip().replace("/", "-")

def parse_jalali_date_safe(s):
    try:
        y, m, d = map(int, s.split("-"))
        return jdatetime.date(y, m, d)
    except:
        return None

def jalali_weekday_name(jd: jdatetime.date):
    wd_map = {
        0:"دوشنبه",1:"سه‌شنبه",2:"چهارشنبه",3:"پنجشنبه",
        4:"جمعه",5:"شنبه",6:"یکشنبه"
    }
    return wd_map[jd.weekday()]

def next_jalali_days(start_date: jdatetime.date, n_days: int):
    return [start_date + jdatetime.timedelta(days=i) for i in range(n_days)]

# =========================
# Shift templates (9..23 only)
# =========================
SHIFT_TEMPLATES = {
    "09-17": list(range(9, 17)),   # 9-16
    "10-18": list(range(10, 18)),
    "11-19": list(range(11, 19)),
    "12-20": list(range(12, 20)),
    "13-21": list(range(13, 21)),
    "14-23": list(range(14, 23)),  # 14-22
    "15-23": list(range(15, 23)),  # 15-22
}

# هر روز (اگر تقاضا داشت) این دو شیفت الزاماً هست
FIXED_MINIMUM = {
    "09-17": 1,
    "14-23": 1
}

def allocate_shifts_for_day(hourly_need):
    """
    1) add fixed shifts at least 1 each (if day has demand)
    2) greedy add more shifts to cover remaining need
    """
    remaining = hourly_need.copy()
    shift_counts = {k: 0 for k in SHIFT_TEMPLATES}

    if sum(remaining.values()) > 0:
        for sh, min_count in FIXED_MINIMUM.items():
            for _ in range(min_count):
                shift_counts[sh] += 1
                for h in SHIFT_TEMPLATES[sh]:
                    if remaining.get(h, 0) > 0:
                        remaining[h] = max(0, remaining[h] - 1)

    for _ in range(1000):
        best_shift, best_cover = None, 0
        for sh, hours in SHIFT_TEMPLATES.items():
            cover = sum(1 for h in hours if remaining.get(h, 0) > 0)
            if cover > best_cover:
                best_cover, best_shift = cover, sh

        if best_cover == 0:
            break

        shift_counts[best_shift] += 1
        for h in SHIFT_TEMPLATES[best_shift]:
            if remaining.get(h, 0) > 0:
                remaining[h] = max(0, remaining[h] - 1)

    return shift_counts

# =========================
# Schedule builder for arbitrary days
# =========================
def build_schedule(days, experts, off_per_expert, daily_shift_counts):
    offs_left = {e: off_per_expert for e in experts}

    schedule = pd.DataFrame(
        index=[d.strftime("%Y-%m-%d") for d in days],
        columns=experts
    )

    ptr = 0
    for d in days:
        dkey = d.strftime("%Y-%m-%d")
        need = daily_shift_counts.get(dkey, None)

        if not need or sum(need.values()) == 0:
            for e in experts:
                schedule.loc[dkey, e] = "OFF"
            continue

        slots = []
        for sh, c in need.items():
            slots += [sh] * c

        extra = max(0, len(experts) - len(slots))
        off_today = []

        for i in range(len(experts)):
            if len(off_today) >= extra:
                break
            e = experts[(ptr + i) % len(experts)]
            if offs_left[e] > 0:
                off_today.append(e)
                offs_left[e] -= 1

        working = [e for e in experts if e not in off_today]
        working = working[ptr:] + working[:ptr]

        for i, e in enumerate(working):
            if i < len(slots):
                schedule.loc[dkey, e] = slots[i]
            else:
                if offs_left[e] > 0:
                    offs_left[e] -= 1
                    schedule.loc[dkey, e] = "OFF"
                else:
                    schedule.loc[dkey, e] = "09-17"

        for e in off_today:
            schedule.loc[dkey, e] = "OFF"

        ptr = (ptr + 1) % len(experts)

    return schedule

# =========================
# Coloring
# =========================
COLOR_MAP = {
    "OFF":   "#ffb3b3",
    "09-17": "#cfe8ff",
    "10-18": "#d6f5d6",
    "11-19": "#e6d6ff",
    "12-20": "#ffe0b3",
    "13-21": "#ffd6e8",
    "14-23": "#fff5b3",
    "15-23": "#cfffff",
}

def color_shifts(val):
    if pd.isna(val):
        return ""
    v = str(val)
    color = COLOR_MAP.get(v, "white")
    return f"background-color: {color}; font-weight: 600; text-align: center;"

# =========================
# UI
# =========================
st.set_page_config(page_title="Erlang Shift Planner", layout="wide")
st.title("📊 برنامه‌ریز هدکانت و شیفت با Erlang C — برنامه ۳۰ روز آینده")

with st.sidebar:
    st.header("ورودی‌ها")

    experts_text = st.text_area("اسم کارشناس‌ها (با کاما جدا کن)", "Ali, Sara, Reza, Mina")
    experts = [x.strip() for x in experts_text.split(",") if x.strip()]

    off_per_expert = st.number_input("تعداد OFF هر کارشناس در ۳۰ روز آینده", min_value=0, value=6)

    aht_sec = st.number_input("AHT (ثانیه)", min_value=10, value=300)

    sl_target = st.slider("Service Level Target", 0.5, 0.99, 0.8, 0.01)
    t_sec = st.selectbox("SLA Threshold (ثانیه)", [20, 40], index=0)

    shrinkage = st.slider("Shrinkage (٪ برای استراحت/غیبت)", 0, 50, 20, 1)

    peak_day = st.selectbox(
        "روز پیک تایم هفته",
        ["شنبه","یکشنبه","دوشنبه","سه‌شنبه","چهارشنبه","پنجشنبه","جمعه"]
    )
    peak_multiplier = st.slider("ضریب پیک برای آن روز", 1.0, 3.0, 1.3, 0.05)

    # شروع ۳۰ روز آینده (پیش‌فرض: فردا)
    today_j = jdatetime.date.today()
    default_start = (today_j + jdatetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start_date_str = st.text_input("تاریخ شروع برنامه (شمسی)", value=default_start)

uploaded = st.file_uploader("فایل ورودی ثابت (xlsx یا csv) با ستون‌های date,hour,volume",
                            type=["xlsx", "csv"])

if uploaded and experts:

    # ---------- read fixed schema ----------
    if uploaded.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    df.columns = [c.strip().lower() for c in df.columns]
    required = {"date", "hour", "volume"}
    if not required.issubset(df.columns):
        st.error("فایل باید دقیقاً ستون‌های date, hour, volume داشته باشد.")
        st.stop()

    df["date"] = df["date"].apply(normalize_date_str)
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    df["hour"] = df["hour"].astype(int)

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    # فقط ساعات 9 تا 23
    df = df[(df["hour"] >= 9) & (df["hour"] <= 23)]

    # aggregate actual daily-hourly
    hourly_vol = (
        df.groupby(["date","hour"], as_index=False)["volume"]
          .sum()
          .rename(columns={"volume":"volume_raw"})
    )

    # ================
    # 1) average profile from history
    # ================
    avg_hourly = (
        hourly_vol.groupby("hour")["volume_raw"]
                  .mean()
                  .reindex(range(9,24), fill_value=0)
    )

    # ================
    # 2) build forecast for next 30 days
    # ================
    start_j = parse_jalali_date_safe(start_date_str)
    if start_j is None:
        st.error("تاریخ شروع نامعتبر است. نمونه درست: 1404-08-01")
        st.stop()

    future_days = next_jalali_days(start_j, 30)

    def is_peak_future(jd):
        return jalali_weekday_name(jd) == peak_day

    # ================
    # 3) Erlang + shifts for future days
    # ================
    hourly_results = []
    daily_shift_counts = {}

    for jd in future_days:
        date_str = jd.strftime("%Y-%m-%d")

        hourly_need = {}
        for h in range(9,24):
            vol_base = float(avg_hourly.loc[h])
            vol_used = vol_base * peak_multiplier if is_peak_future(jd) else vol_base

            n, a, sl = required_agents_erlang(
                volume_per_hour=vol_used,
                aht_sec=aht_sec,
                sl_target=sl_target,
                t_sec=t_sec
            )

            n_eff = math.ceil(n / (1 - shrinkage/100))
            hourly_need[h] = n_eff

            hourly_results.append({
                "date": date_str,
                "weekday": jalali_weekday_name(jd),
                "hour": h,
                "forecast_volume": round(vol_used, 1),
                "offered_load_erlangs": round(a, 2),
                "agents_needed": n_eff,
                "service_level": round(sl, 3),
                "sla_threshold_sec": t_sec
            })

        shifts = allocate_shifts_for_day(hourly_need)
        daily_shift_counts[date_str] = shifts

    hourly_df = pd.DataFrame(hourly_results)

    # =========================
    # Outputs
    # =========================
    st.subheader("۱) پروفایل میانگین ورودیِ تاریخی (ساعتی)")
    st.dataframe(avg_hourly.reset_index().rename(columns={"hour":"hour","volume_raw":"avg_volume"}))

    st.subheader("۲) هدکانت ساعتی پیش‌بینی‌شده برای ۳۰ روز آینده (Erlang C)")
    st.dataframe(hourly_df, use_container_width=True)

    st.download_button(
        "دانلود جدول هدکانت ساعتی پیش‌بینی‌شده",
        data=hourly_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="forecast_hourly_headcount_30days.csv",
        mime="text/csv"
    )

    st.subheader("۳) تعداد شیفت موردنیاز هر روز (۳۰ روز آینده)")
    daily_df = pd.DataFrame(
        [{"date": d, **c} for d, c in daily_shift_counts.items()]
    )
    st.dataframe(daily_df, use_container_width=True)

    st.subheader("۴) جدول نهایی شیفت ۳۰ روز آینده (OFF قرمز، شیفت‌های مشابه هم‌رنگ)")
    schedule_df = build_schedule(
        days=future_days,
        experts=experts,
        off_per_expert=off_per_expert,
        daily_shift_counts=daily_shift_counts
    )
    styled = schedule_df.style.applymap(color_shifts)
    st.dataframe(styled, use_container_width=True)

    st.download_button(
        "دانلود جدول شیفت ۳۰ روز آینده",
        data=schedule_df.to_csv().encode("utf-8-sig"),
        file_name="schedule_next_30_days.csv",
        mime="text/csv"
    )

    st.subheader("۵) نمودار میله‌ای میانگین ورودی روزانه به ازای هر ساعت (تاریخی)")
    fig, ax = plt.subplots()
    ax.bar(avg_hourly.index, avg_hourly.values)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Daily Volume")
    ax.set_xticks(range(9,24))
    st.pyplot(fig)

else:
    st.info("👈 فایل را آپلود کن و اسامی کارشناسان را وارد کن.")
