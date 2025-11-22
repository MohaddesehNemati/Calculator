import streamlit as st
import pandas as pd
import math
import jdatetime
import matplotlib.pyplot as plt

# ----------------------------
# Erlang C functions
# ----------------------------
def erlang_c_probability_wait(a, n):
    if n <= a:
        return 1.0
    s = sum((a**k)/math.factorial(k) for k in range(n))
    pn = (a**n)/math.factorial(n) * (n/(n-a))
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

# ----------------------------
# Shift templates
# ----------------------------
SHIFT_TEMPLATES = {
    "M": list(range(8, 16)),   # 8-15
    "E": list(range(12, 20)),  # 12-19
    "N": list(range(16, 24))   # 16-23
}

def allocate_shifts_for_day(hourly_need):
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
                remaining[h] -= 1
    return shift_counts

# ----------------------------
# Jalali month generator
# ----------------------------
def jalali_month_days(year, month):
    first_day = jdatetime.date(year, month, 1)
    if month <= 6:
        days = 31
    elif month <= 11:
        days = 30
    else:
        days = 30 if first_day.isleap() else 29
    return [jdatetime.date(year, month, d) for d in range(1, days + 1)]

# ----------------------------
# Scheduler
# ----------------------------
def build_month_schedule(days, experts, off_per_expert, daily_shift_counts):
    offs_left = {e: off_per_expert for e in experts}
    schedule = pd.DataFrame(
        index=[d.strftime("%Y-%m-%d") for d in days],
        columns=experts
    )
    ptr = 0
    for d in days:
        dkey = d.strftime("%Y-%m-%d")
        need = daily_shift_counts.get(dkey, {"M":0,"E":0,"N":0})
        slots = (["M"] * need["M"]) + (["E"] * need["E"]) + (["N"] * need["N"])

        extra = max(0, len(experts) - len(slots))
        off_today = []
        idxs = list(range(len(experts)))

        for _ in range(extra):
            for i in idxs:
                e = experts[(ptr + i) % len(experts)]
                if offs_left[e] > 0 and e not in off_today:
                    off_today.append(e)
                    offs_left[e] -= 1
                    break

        working = [e for e in experts if e not in off_today]
        working = working[ptr:] + working[:ptr]

        for i, e in enumerate(working):
            if i < len(slots):
                schedule.loc[dkey, e] = slots[i]
            else:
                if offs_left[e] > 0:
                    schedule.loc[dkey, e] = "OFF"
                    offs_left[e] -= 1
                else:
                    schedule.loc[dkey, e] = "M"

        for e in off_today:
            schedule.loc[dkey, e] = "OFF"

        ptr = (ptr + 1) % len(experts)

    return schedule

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="Erlang Shift Planner", layout="wide")
st.title("📊 برنامه‌ریز هدکانت و شیفت با Erlang C (تاریخ شمسی)")

with st.sidebar:
    st.header("ورودی‌ها")

    year = st.number_input("سال شمسی", min_value=1390, max_value=1500, value=1404)
    month = st.number_input("ماه شمسی (1-12)", min_value=1, max_value=12, value=7)

    experts_text = st.text_area("اسم کارشناس‌ها (با کاما جدا کن)", "Ali, Sara, Reza, Mina")
    experts = [x.strip() for x in experts_text.split(",") if x.strip()]

    off_per_expert = st.number_input("تعداد OFF هر کارشناس در ماه", min_value=0, value=6)

    aht_sec = st.number_input("AHT (ثانیه)", min_value=10, value=300)

    sl_target = st.slider("Service Level Target", 0.5, 0.99, 0.8, 0.01)
    t_sec = st.number_input("آستانه پاسخگویی SLA (ثانیه)", min_value=1, value=20)

    shrinkage = st.slider("Shrinkage (٪ برای استراحت/غیبت)", 0, 50, 20, 1)

    peak_day = st.selectbox(
        "روز پیک تایم هفته",
        ["شنبه","یکشنبه","دوشنبه","سه‌شنبه","چهارشنبه","پنجشنبه","جمعه"]
    )
    peak_multiplier = st.slider("ضریب پیک برای آن روز", 1.0, 3.0, 1.3, 0.05)

uploaded = st.file_uploader("فایل ورودی (xlsx یا csv)", type=["xlsx", "csv"])

def normalize_date_str(x):
    # 1404/07/01 -> 1404-07-01
    s = str(x).strip()
    s = s.replace("/", "-")
    return s

if uploaded and experts:
    # ---- read file
    if uploaded.name.lower().endswith(".xlsx"):
        raw = pd.read_excel(uploaded)
    else:
        raw = pd.read_csv(uploaded)

    raw.columns = [str(c).strip() for c in raw.columns]

    # ---- detect format
    if set(["date", "hour", "volume"]).issubset({c.lower() for c in raw.columns}):
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = df["date"].apply(normalize_date_str)
        df["hour"] = df["hour"].astype(int)
        df["volume"] = df["volume"].astype(float)
    else:
        # Persian sample like yours
        date_col = "تاریخ تماس" if "تاریخ تماس" in raw.columns else None
        hour_col = "hour" if "hour" in raw.columns else None
        vol_col  = "تعداد" if "تعداد" in raw.columns else None

        if not date_col or not hour_col:
            st.error("ستون‌های فایل باید حداقل شامل 'تاریخ تماس' و 'hour' باشد.")
            st.stop()

        df = raw[[date_col, hour_col] + ([vol_col] if vol_col else [])].copy()
        df.rename(columns={date_col: "date", hour_col: "hour"}, inplace=True)
        df["date"] = df["date"].apply(normalize_date_str)
        df["hour"] = df["hour"].astype(int)

        if vol_col:
            df.rename(columns={vol_col: "volume"}, inplace=True)
        else:
            df["volume"] = 1

        df["volume"] = df["volume"].astype(float)

    # ---- build hourly volumes per day-hour
    hourly_vol = (
        df.groupby(["date", "hour"], as_index=False)["volume"]
          .sum()
          .rename(columns={"volume": "volume_raw"})
    )

    # ---- peak-day helper
    def is_peak(jdate_str):
        try:
            y, m, d = map(int, jdate_str.split("-"))
            jd = jdatetime.date(y, m, d)
            wd_map = {
                0:"دوشنبه",1:"سه‌شنبه",2:"چهارشنبه",3:"پنجشنبه",
                4:"جمعه",5:"شنبه",6:"یکشنبه"
            }
            return wd_map[jd.weekday()] == peak_day
        except:
            return False

    hourly_results = []
    daily_shift_counts = {}

    for date_str, g in hourly_vol.groupby("date"):
        hourly_need = {}
        for _, row in g.iterrows():
            vol_raw = row["volume_raw"]
            vol_used = vol_raw * peak_multiplier if is_peak(date_str) else vol_raw

            n, a, sl = required_agents_erlang(
                volume_per_hour=vol_used,
                aht_sec=aht_sec,
                sl_target=sl_target,
                t_sec=t_sec
            )

            n_eff = math.ceil(n / (1 - shrinkage/100))
            hourly_need[int(row["hour"])] = n_eff

            hourly_results.append({
                "date": date_str,
                "hour": int(row["hour"]),
                "volume_raw": int(vol_raw),
                "volume_used": int(vol_used),
                "offered_load_erlangs": round(a, 2),
                "agents_needed": n_eff,
                "service_level": round(sl, 3)
            })

        shifts = allocate_shifts_for_day(hourly_need)
        daily_shift_counts[date_str] = shifts

    hourly_df = pd.DataFrame(hourly_results).sort_values(["date","hour"])

    st.subheader("۱) هدکانت لازم به ازای هر ساعت (Erlang C)")
    st.dataframe(hourly_df, use_container_width=True)

    st.subheader("۲) تعداد شیفت موردنیاز هر روز")
    daily_df = pd.DataFrame(
        [{"date": d, **c} for d, c in daily_shift_counts.items()]
    ).sort_values("date")
    st.dataframe(daily_df, use_container_width=True)

    st.subheader("۳) جدول نهایی شیفت ماه شمسی")
    days = jalali_month_days(year, month)
    schedule_df = build_month_schedule(
        days=days,
        experts=experts,
        off_per_expert=off_per_expert,
        daily_shift_counts=daily_shift_counts
    )
    st.dataframe(schedule_df, use_container_width=True)

    st.download_button(
        "دانلود خروجی CSV",
        data=schedule_df.to_csv().encode("utf-8-sig"),
        file_name=f"schedule_{year}_{month}.csv",
        mime="text/csv"
    )

    # ----------------------------
    # 4) Bar chart: average daily inputs per hour
    # ----------------------------
    st.subheader("۴) نمودار میانگین ورودی روزانه به ازای هر ساعت")

    avg_hourly = (
        hourly_df.groupby("hour")["volume_raw"]
                 .mean()
                 .reindex(range(0,24), fill_value=0)
    )

    fig, ax = plt.subplots()
    ax.bar(avg_hourly.index, avg_hourly.values)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Daily Volume")
    ax.set_xticks(range(0,24))
    st.pyplot(fig)

else:
    st.info("👈 فایل را آپلود کن و اسامی کارشناسان را وارد کن.")
