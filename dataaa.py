import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px

# --- تنظیمات صفحه ---
st.set_page_config(page_title="داشبورد ریسک اختیار معامله", layout="wide")

# --- عنوان و توضیحات ---
st.title("📊 داشبورد تحلیل ریسک و مهندسی مالی")
st.markdown("""
این سیستم برای محاسبه سود و زیان (P&L) استراتژی اختیار معامله با استفاده از **شبیه‌سازی مونت‌کارلو** طراحی شده است.
همبستگی بین دارایی‌ها با روش **تجزيه چولسکی (Cholesky Decomposition)** اعمال می‌شود.
""")

# --- سایدبار: تنظیمات ورودی ---
st.sidebar.header("⚙️ تنظیمات شبیه‌سازی")

n_sims = st.sidebar.slider("تعداد شبیه‌سازی (سناریو)", 
                           min_value=1000, max_value=50000, value=1000, step=1000)

rho_input = st.sidebar.slider("ضریب همبستگی (Correlation)", 
                              min_value=-1.0, max_value=1.0, value=0.6043, step=0.01)

run_button = st.sidebar.button("🚀 اجرای محاسبات")

# --- توابع محاسباتی (Black-Scholes) ---
def black_scholes_call(S, K, T, r, sigma):
    # تبدیل زمان به سال
    T_year = T / 250.0 
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_year) / (sigma * np.sqrt(T_year))
    d2 = d1 - sigma * np.sqrt(T_year)
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T_year) * stats.norm.cdf(d2)
    return call_price, d1, d2, stats.norm.cdf(d1)

# --- داده‌های اولیه ---
# پارامترهای اهرم (Long Call)
S0_ahrom = 30230
K_ahrom = 28000
T_ahrom_days = 64
sigma_ahrom = 0.02877 * np.sqrt(250) # سالانه سازی تقریبی برای نمایش، اما در GBM از روزانه استفاده میکنیم
mu_daily_ahrom = 0.0012
sigma_daily_ahrom = np.sqrt(0.0008280517868559158)
premium_ahrom = 5113
qty_ahrom = 100 * 1000 # 100 قرارداد

# پارامترهای وبملت (Short Call)
S0_mellat = 1365
K_mellat = 1200
T_mellat_days = 22
sigma_mellat = 0.02341 * np.sqrt(250)
mu_daily_mellat = 0.0022203855624574834
sigma_daily_mellat = np.sqrt(0.000548162149229913)
premium_mellat = 206
qty_mellat = 100 * 1000 # 100 قرارداد

r_annual = 0.33
days_in_year = 250

# --- بخش اصلی محاسبات ---
if run_button:
    with st.spinner('در حال انجام شبیه‌سازی مونت‌کارلو...'):
        
        # 1. تولید اعداد تصادفی همبسته (Cholesky)
        # ماتریس همبستگی
        corr_matrix = np.array([[1.0, rho_input], 
                                [rho_input, 1.0]])
        
        # تجزیه چولسکی
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            st.error("ماتریس همبستگی مثبت معین نیست. لطفاً ضریب همبستگی را تغییر دهید.")
            st.stop()

        # تولید Z نرمال استاندارد مستقل
        Z_uncorrelated = np.random.normal(0, 1, size=(2, n_sims))
        
        # اعمال همبستگی
        Z_correlated = L @ Z_uncorrelated
        
        Z_ahrom = Z_correlated[0, :]
        Z_mellat = Z_correlated[1, :]

        # 2. شبیه‌سازی قیمت نهایی (GBM) برای اهرم (64 روز)
        # توجه: چون شبیه‌سازی مسیر کامل نیست و فقط قیمت نهایی مهم است، از فرمول صریح استفاده می‌کنیم
        # Drift = (mu - 0.5 * sigma^2) * T
        # Diffusion = sigma * sqrt(T) * Z
        
        # محاسبات اهرم (64 روز)
        drift_ahrom = (mu_daily_ahrom - 0.5 * sigma_daily_ahrom**2) * T_ahrom_days
        diff_ahrom = sigma_daily_ahrom * np.sqrt(T_ahrom_days) * Z_ahrom
        S_T_ahrom = S0_ahrom * np.exp(drift_ahrom + diff_ahrom)
        
        # محاسبات وبملت (22 روز)
        # نکته مهم: برای همبستگی دقیق، باید فرض کنیم Z برای بازه زمانی مشترک است.
        # اما چون زمان‌ها متفاوت است (64 vs 22)، مدل ساده‌سازی شده و از همان Z همبسته استفاده می‌کند
        # که برای نشان دادن اثر همبستگی کلی کافی است.
        drift_mellat = (mu_daily_mellat - 0.5 * sigma_daily_mellat**2) * T_mellat_days
        diff_mellat = sigma_daily_mellat * np.sqrt(T_mellat_days) * Z_mellat
        S_T_mellat = S0_mellat * np.exp(drift_mellat + diff_mellat)

        # 3. محاسبه Payoff و P&L
        # سود اهرم (Long Call): Max(S_T - K, 0) - Cost
        payoff_ahrom = np.maximum(S_T_ahrom - K_ahrom, 0)
        pnl_ahrom = (payoff_ahrom - premium_ahrom) * qty_ahrom
        
        # سود وبملت (Short Call): Premium - Max(S_T - K, 0)
        payoff_mellat = np.maximum(S_T_mellat - K_mellat, 0)
        pnl_mellat = (premium_mellat - payoff_mellat) * qty_mellat
        
        # سود کل پرتفوی
        total_pnl = pnl_ahrom + pnl_mellat

        # --- نمایش داده‌های آماری ---
        st.success(f"محاسبات برای {n_sims:,} سناریو با موفقیت انجام شد.")
        
        # --- محاسبات آماری ---
        mean_pnl = np.mean(total_pnl)
        var_95 = np.percentile(total_pnl, 5) # صدک 5م (95% اطمینان)
        win_rate = np.mean(total_pnl > 0) * 100
        
        # نمایش KPI
        col1, col2, col3 = st.columns(3)
        col1.metric("میانگین سود کل", f"{mean_pnl:,.0f} IRR", delta="مورد انتظار")
        col2.metric("ریسک (VaR 95%)", f"{var_95:,.0f} IRR", delta="ریسک", delta_color="inverse")
        col3.metric("احتمال سود (Win Rate)", f"{win_rate:.2f}%")

        # --- نمودارها (با اصلاحات کامل) ---
        st.markdown("### 📊 تحلیل بصری")
        tab1, tab2 = st.tabs(["نمودار پراکندگی (Scatter)", "توزیع سود (Histogram)"])
        
        # محدود کردن داده‌ها برای رسم نمودار اسکتر (جلوگیری از سنگینی)
        plot_limit = 2000
        indices = np.random.choice(len(total_pnl), size=min(len(total_pnl), plot_limit), replace=False)
        
        df_plot = pd.DataFrame({
            'Ahrom Price': S_T_ahrom[indices],
            'Mellat Price': S_T_mellat[indices],
            'P&L': total_pnl[indices]
        })

        with tab1:
            try:
                fig_scatter = px.scatter(
                    df_plot,
                    x='Ahrom Price',
                    y='Mellat Price',
                    color='P&L',
                    color_continuous_scale='RdYlGn',
                    title=f'همبستگی قیمت نهایی (نمایش {len(df_plot)} نقطه تصادفی)',
                    labels={'Ahrom Price': 'قیمت نهایی اهرم', 'Mellat Price': 'قیمت نهایی وبملت'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.error(f"خطا در رسم نمودار پراکندگی: {e}")

        with tab2:
            try:
                # استفاده از تمام داده‌ها برای هیستوگرام (چون سنگین نیست)
                fig_hist = px.histogram(
                    total_pnl,
                    nbins=50,
                    title='توزیع سود و زیان پرتفوی',
                    labels={'value': 'سود/زیان (ریال)'},
                    color_discrete_sequence=['#636EFA']
                )
                # افزودن خطوط عمودی
                fig_hist.add_vline(x=0, line_color="black", annotation_text="نقطه سربه‌سر")
                fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="VaR 95%")
                
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(f"خطا در رسم هیستوگرام: {e}")

        # --- بخش جزئیات بلک-شولز ---
        with st.expander("مشاهده جزئیات محاسبات تئوریک (Black-Scholes)"):
            c_ahrom, d1_a, d2_a, N_d1_a = black_scholes_call(S0_ahrom, K_ahrom, T_ahrom_days, r_annual, sigma_ahrom)
            c_mellat, d1_m, d2_m, N_d1_m = black_scholes_call(S0_mellat, K_mellat, T_mellat_days, r_annual, sigma_mellat)
            
            bs_col1, bs_col2 = st.columns(2)
            
            with bs_col1:
                st.info("**اهرم (Long Call)**")
                st.write(f"Price (Theoretical): {c_ahrom:,.2f}")
                st.write(f"d1: {d1_a:.4f}")
                st.write(f"d2: {d2_a:.4f}")
                st.write(f"Delta (N(d1)): {N_d1_a:.4f}")
            
            with bs_col2:
                st.info("**وبملت (Short Call)**")
                st.write(f"Price (Theoretical): {c_mellat:,.2f}")
                st.write(f"d1: {d1_m:.4f}")
                st.write(f"d2: {d2_m:.4f}")
                st.write(f"Delta (N(d1)): {N_d1_m:.4f}")

else:
    st.info("👈 لطفاً پارامترها را در منوی سمت راست تنظیم کرده و دکمه **اجرای محاسبات** را بزنید.")
