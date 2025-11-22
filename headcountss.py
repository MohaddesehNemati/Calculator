import streamlit as st
import pandas as pd
import math
import os
import jdatetime
import matplotlib.pyplot as plt

# =========================
# Header (Jabama logo)
# =========================
LOGO_PATH = "jabama_logo.png"
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=180)
st.title("📊 برنامه‌ریز هدکانت و شیفت با Erlang")

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
    if n <= 0:
        return 0.0
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
# Shifts (MAX 4 MODELS)
# =========================
SHIFT_TEMPLATES = {
    "09-17": list(range(9, 17)),
    "14-23": list(range(14, 23)),
    "10-18": list(range(10, 18)),
    "12-20": list(range(12, 20)),
}
SHIFT_ORDER = ["09-17", "14-23", "10-18", "12-20"]

# --- این دو شیفت فقط در چیدمان نفرات اجباری‌اند ---
MANDATORY_SLOTS = ["09-17", "14-23"]  # برای Schedule
MANDATORY_TOTAL = len(MANDATORY_SLOTS)  # =2

def allocate_shifts_for_day(hourly_need):
    """
    محاسبه‌ی هدکانت مورد نیاز شیفت‌ها
    """
    remaining = hourly_need.copy()
    shift_counts = {k: 0 for k in SHIFT_TEMPLATES}

    for _ in range(500):
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
# SLA achieved from actual schedule
# =========================
def achieved_sla_for_day(date_j, schedule_row, avg_hourly, aht_sec, t_sec, peak_day, peak_multiplier):
    per_hour_sla = {}
    is_peak = jalali_weekday_name(date_j) == peak_day

    for h in range(9, 24):
        agents = 0
        for sh in schedule_row.values:
            if sh == "OFF" or pd.isna(sh):
                continue
            if h in SHIFT_TEMPLATES.get(sh, []):
                agents += 1

        vol_base = float(avg_hourly.get(h, 0))
        vol_used = vol_base * peak_multiplier if is_peak else vol_base

        if vol_used <= 0:
            sl = 1.0
        else:
            lam = vol_used / 3600.0
            mu = 1 / aht_sec
            a = lam / mu
            sl = erlang_c_service_level(a, agents, aht_sec, t_sec)

        per_hour_sla[h] = sl

    daily_min = min(per_hour_sla.values()) if per_hour_sla else 1.0
    daily_avg = sum(per_hour_sla.values()) / len(per_hour_sla) if per_hour_sla else 1.0
    return per_hour_sla, daily_min, daily_avg

# =========================
# Schedule builder with constraints
# - Peak day: OFF تا حد امکان صفر
# - No 3 OFF in a row
# - OFF count must be met
# - BUT schedule must ALWAYS include:
#   at least 1 person in 09-17 and 1 in 14-23
# =========================
def build_schedule_with_constraints(days, experts, off_per_expert, daily_shift_counts, peak_day):
    offs_left = {e: off_per_expert for e in experts}
    off_streak = {e: 0 for e in experts}

    schedule = pd.DataFrame(
        index=[d.strftime("%Y-%m-%d") for d in days],
        columns=experts
    )

    ptr = 0
    total_days = len(days)

    for di, d in enumerate(days):
        dkey = d.strftime("%Y-%m-%d")
        need = daily_shift_counts.get(dkey, {})  # نیاز دیتامحور (ممکنه برای ثابت‌ها 0 باشه)

        is_peak = jalali_weekday_name(d) == peak_day

        # سقف OFF روزانه طوری که حداقل ۲ نفر برای دو شیفت اجباری بمانند
        max_off_today = 0 if is_peak else max(0, len(experts) - MANDATORY_TOTAL)

        remaining_days_including_today = total_days - di
        total_off_remaining = sum(offs_left.values())
        avg_needed_today = math.ceil(total_off_remaining / remaining_days_including_today) if remaining_days_including_today > 0 else 0

        target_off_today = 0 if is_peak else min(max_off_today, avg_needed_today)

        ordered = experts[ptr:] + experts[:ptr]
        eligible = [
            e for e in ordered
            if offs_left[e] > 0 and off_streak[e] < 2
        ]
        eligible.sort(key=lambda e: (-offs_left[e], off_streak[e]))

        off_today = []
        for e in eligible:
            if len(off_today) >= target_off_today:
                break
            off_today.append(e)

        working = [e for e in experts if e not in off_today]
        working = working[ptr:] + working[:ptr]

        # --- ساخت اسلات‌ها: اول دو اسلات اجباری ---
        slots = MANDATORY_SLOTS.copy()

        # بعد نیاز محاسبه‌شده دیتامحور (اگر همون شیفت‌ها دوباره اومدن مشکلی نیست)
        for sh in SHIFT_ORDER:
            slots += [sh] * need.get(sh, 0)

        # تخصیص شیفت به working
        for i, e in enumerate(working):
            if i < len(slots):
                schedule.loc[dkey, e] = slots[i]
            else:
                schedule.loc[dkey, e] = "09-17"

        for e in off_today:
            schedule.loc[dkey, e] = "OFF"
            offs_left[e] -= 1

        # update streak
        for e in experts:
            if schedule.loc[dkey, e] == "OFF":
                off_streak[e] += 1
            else:
                off_streak[e] = 0

        ptr = (ptr + 1) % len(experts)

    return schedule

# =========================
# Coloring
# =========================
COLOR_MAP = {
    "OFF":   "#ffb3b3",
    "09-17": "#cfe8ff",
    "10-18": "#d6f5d6",
    "12-20": "#e6d6ff",
    "14-23": "#fff5b3",
}
def color_shifts(val):
    if pd.isna(val):
        return ""
    v = str(val)
    color = COLOR_MAP.get(v, "white")
    return f"background-color: {color}; font-weight: 600; text-align: center;"

# =========================
# Sidebar
# =========================
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

    today_j = jdatetime.date.today()
    default_start = (today_j + jdatetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start_date_str = st.text_input("تاریخ شروع برنامه (شمسی)", value=default_start)

uploaded = st.file_uploader(
    "فایل ورودی ثابت (xlsx یا csv) با ستون‌های date,hour,volume",
    type=["xlsx", "csv"]
)

# =========================
# Main
# =========================
if uploaded and experts:
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

    df = df[(df["hour"] >= 9) & (df["hour"] <= 23)]

    hourly_vol = (
        df.groupby(["date", "hour"], as_index=False)["volume"]
          .sum()
          .rename(columns={"volume": "volume_raw"})
    )

    avg_hourly = (
        hourly_vol.groupby("hour")["volume_raw"]
                  .mean()
                  .reindex(range(9,24), fill_value=0)
    )

    st.subheader("۱) میانگین ورودیِ(ساعتی)")
    st.dataframe(
        avg_hourly.reset_index().rename(columns={"hour":"hour", "volume_raw":"avg_volume"}),
        use_container_width=True
    )

    start_j = parse_jalali_date_safe(start_date_str)
    if start_j is None:
        st.error("تاریخ شروع نامعتبر است. نمونه: 1404-08-01")
        st.stop()

    future_days = next_jalali_days(start_j, 30)

    daily_shift_counts = {}
    hourly_results = []

    for jd in future_days:
        date_str = jd.strftime("%Y-%m-%d")
        is_peak = jalali_weekday_name(jd) == peak_day

        hourly_need = {}
        for h in range(9,24):
            vol_base = float(avg_hourly.loc[h])
            vol_used = vol_base * peak_multiplier if is_peak else vol_base

            n_req, a, sl_req = required_agents_erlang(
                volume_per_hour=vol_used,
                aht_sec=aht_sec,
                sl_target=sl_target,
                t_sec=t_sec
            )

            n_eff = math.ceil(n_req / (1 - shrinkage/100))
            hourly_need[h] = n_eff

            hourly_results.append({
                "date": date_str,
                "weekday": jalali_weekday_name(jd),
                "hour": h,
                "forecast_volume": round(vol_used, 1),
                "required_agents": n_eff,
                "sla_threshold_sec": t_sec
            })

        # نیاز شیفت‌ها (بدون حداقل اجباری)
        shifts_needed = allocate_shifts_for_day(hourly_need)
        daily_shift_counts[date_str] = shifts_needed

    hourly_df = pd.DataFrame(hourly_results)

    st.subheader("۲) هدکانت ساعتی پیش‌بینی‌شده (۳۰ روز آینده)")
    st.dataframe(hourly_df, use_container_width=True)

    st.subheader("۳) هدکانت مورد نیاز شیفت هر روز ")
    daily_df = pd.DataFrame(
        [{"date": d, **c} for d, c in daily_shift_counts.items()]
    )
    st.dataframe(daily_df, use_container_width=True)

    # چیدمان نفرات با سیاست اجباری ثابت‌ها
    schedule_df = build_schedule_with_constraints(
        days=future_days,
        experts=experts,
        off_per_expert=off_per_expert,
        daily_shift_counts=daily_shift_counts,
        peak_day=peak_day
    )

    # SLA واقعی روزانه
    day_rows = []
    new_index = []
    for jd in future_days:
        dkey = jd.strftime("%Y-%m-%d")
        row = schedule_df.loc[dkey]

        per_hour_sla, daily_min, daily_avg = achieved_sla_for_day(
            date_j=jd,
            schedule_row=row,
            avg_hourly=avg_hourly,
            aht_sec=aht_sec,
            t_sec=t_sec,
            peak_day=peak_day,
            peak_multiplier=peak_multiplier
        )

        assigned_working = int((row != "OFF").sum())

        day_rows.append({
            "date": dkey,
            "weekday": jalali_weekday_name(jd),
            "assigned_working": assigned_working,
            "achieved_min_SL": round(daily_min * 100, 1),
            "achieved_avg_SL": round(daily_avg * 100, 1),
            "target_SL": round(sl_target * 100, 1),
            "peak_day": "YES" if jalali_weekday_name(jd) == peak_day else ""
        })

        new_index.append(f"{dkey} | SL≈{round(daily_min*100,1)}%")

    daily_sla_df = pd.DataFrame(day_rows)

    st.subheader("۴) SLA احتمالی هر روز با شیفت‌های واقعی")
    st.dataframe(daily_sla_df, use_container_width=True)

    st.subheader("۵) جدول نهایی شیفت ۳۰ روز آینده")
    schedule_df.index = new_index
    st.dataframe(schedule_df.style.applymap(color_shifts), use_container_width=True)

    st.subheader("۶) نمودار میانگین ورودی روزانه به ازای هر ساعت (تاریخی)")
    fig, ax = plt.subplots()
    ax.bar(avg_hourly.index, avg_hourly.values)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Daily Volume")
    ax.set_xticks(range(9,24))
    st.pyplot(fig)

else:
    st.info("👈 فایل را آپلود کن و اسامی کارشناسان را وارد کن.")


