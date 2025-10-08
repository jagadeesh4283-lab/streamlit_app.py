# streamlit_app.py
# Interactive Streamlit app for GEOLUC descriptive EDA with automated textual insights
# Expects a CSV with columns like:
# Region, Land_Cover_Type, Year, Area_sq_km, NDVI, Population_Density, Temperature_Anomaly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="GEOLUC Descriptive EDA", layout="wide")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            df = pd.read_csv(uploaded_file, encoding='windows-1252', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
    return df

def ensure_columns(df):
    expected = ['Region','Land_Cover_Type','Year','Area_sq_km','NDVI','Population_Density','Temperature_Anomaly']
    missing = [c for c in expected if c not in df.columns]
    return missing

def safe_cast_year(df):
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    return df

st.title("Global Earth Observation — Interactive Descriptive EDA")
st.markdown("Upload a CSV and use filters to explore. Automated textual insights appear under each chart.")

# Sidebar controls
with st.sidebar:
    st.header("Data & Plot Options")
    uploaded = st.file_uploader("Upload CSV (Region, Land_Cover_Type, Year, Area_sq_km, NDVI, Population_Density, Temperature_Anomaly)", type=['csv'])
    sample_mode = st.checkbox("Use random sampling for large data", True)
    sample_size = st.number_input("Sample size (when sampling)", min_value=100, max_value=500000, value=3000, step=100)
    show_raw = st.checkbox("Show filtered table", False)
    top_n = st.slider("Top N (bar charts)", 3, 20, 6)
    st.markdown("---")

df = load_data(uploaded)
if df is None:
    st.info("Please upload a CSV to begin. Expected columns: Region, Land_Cover_Type, Year, Area_sq_km, NDVI, Population_Density, Temperature_Anomaly.")
    st.stop()

missing_cols = ensure_columns(df)
if missing_cols:
    st.warning(f"Expected columns missing: {missing_cols}. Some charts/insights may be unavailable.")

df = safe_cast_year(df)

# dynamic filters
regions = df['Region'].dropna().unique().tolist() if 'Region' in df.columns else []
landcovers = df['Land_Cover_Type'].dropna().unique().tolist() if 'Land_Cover_Type' in df.columns else []
years = sorted(df['Year'].dropna().unique().tolist()) if 'Year' in df.columns else []

with st.sidebar:
    sel_regions = st.multiselect("Select Regions", options=regions, default=regions[:min(6,len(regions))])
    sel_landcovers = st.multiselect("Select Land Cover Types", options=landcovers, default=landcovers)
    if years:
        yr_min, yr_max = int(years[0]), int(years[-1])
        sel_years = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
    else:
        sel_years = None

# apply filters
df_filtered = df.copy()
if sel_regions:
    df_filtered = df_filtered[df_filtered['Region'].isin(sel_regions)]
if sel_landcovers:
    df_filtered = df_filtered[df_filtered['Land_Cover_Type'].isin(sel_landcovers)]
if sel_years and 'Year' in df_filtered.columns:
    df_filtered = df_filtered[(df_filtered['Year'] >= sel_years[0]) & (df_filtered['Year'] <= sel_years[1])]

st.sidebar.markdown(f"**Filtered records:** {len(df_filtered):,}")

plot_df = df_filtered
if sample_mode and len(plot_df) > sample_size:
    plot_df = plot_df.sample(sample_size, random_state=42)

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Filtered records", f"{len(df_filtered):,}")
with col2:
    if 'Region' in df_filtered.columns:
        st.metric("Unique regions", df_filtered['Region'].nunique())
with col3:
    if 'Land_Cover_Type' in df_filtered.columns:
        st.metric("Land cover types", df_filtered['Land_Cover_Type'].nunique())
with col4:
    if 'Year' in df_filtered.columns and not df_filtered['Year'].isna().all():
        yrs = df_filtered['Year'].dropna().unique()
        st.metric("Year range", f"{int(min(yrs))} — {int(max(yrs))}")

st.markdown("---")

# 1) Land Cover Distribution
st.subheader("1) Land Cover Distribution")
if 'Land_Cover_Type' in df_filtered.columns:
    lc_counts = df_filtered['Land_Cover_Type'].value_counts().reset_index()
    lc_counts.columns = ['Land_Cover_Type','count']
    fig = px.bar(lc_counts, x='Land_Cover_Type', y='count', text='count')
    fig.update_layout(xaxis_title="Land Cover Type", yaxis_title="Count", height=420)
    st.plotly_chart(fig, use_container_width=True)
    if len(lc_counts)>0:
        top = lc_counts.iloc[0]
        st.markdown(f"**Insight:** The most common land cover type is **{top['Land_Cover_Type']}** with {top['count']:,} records ({top['count']/lc_counts['count'].sum():.1%} of filtered data).")
else:
    st.info("Missing Land_Cover_Type column.")

# 2) Total Area by Region
st.subheader("2) Total Area by Region (sq.km) — Top N")
if 'Area_sq_km' in df_filtered.columns and 'Region' in df_filtered.columns:
    region_area = df_filtered.groupby('Region', as_index=False)['Area_sq_km'].sum().sort_values('Area_sq_km', ascending=False).head(top_n)
    fig = px.bar(region_area, x='Region', y='Area_sq_km', text='Area_sq_km')
    fig.update_layout(yaxis_title="Total Area (sq.km)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    if len(region_area)>0:
        topreg = region_area.iloc[0]
        st.markdown(f"**Insight:** Top region by area is **{topreg['Region']}** with {topreg['Area_sq_km']:,} sq.km (top {top_n}).")
else:
    st.info("Missing Area_sq_km or Region column.")

# 3) Yearly Total Area Trend
st.subheader("3) Yearly Total Area Trend")
if 'Year' in df_filtered.columns and 'Area_sq_km' in df_filtered.columns:
    yearly_area = df_filtered.groupby('Year', as_index=False)['Area_sq_km'].sum().sort_values('Year')
    fig = px.line(yearly_area, x='Year', y='Area_sq_km', markers=True)
    fig.update_layout(yaxis_title="Total Area (sq.km)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    if len(yearly_area)>=2:
        start_val = yearly_area.iloc[0]['Area_sq_km']
        end_val = yearly_area.iloc[-1]['Area_sq_km']
        pct = ((end_val - start_val)/start_val*100) if start_val != 0 else np.nan
        trend = "increased" if end_val > start_val else "decreased"
        st.markdown(f"**Insight:** From {int(yearly_area['Year'].min())} to {int(yearly_area['Year'].max())}, total area {trend} from {start_val:,.0f} to {end_val:,.0f} sq.km ({pct:.2f}% change).")
else:
    st.info("Missing Year or Area_sq_km column.")

# 4) Yearly Area by Land Cover (stacked)
st.subheader("4) Yearly Area by Land Cover Type (stacked)")
if set(['Year','Land_Cover_Type','Area_sq_km']).issubset(df_filtered.columns):
    yearly_lc = df_filtered.groupby(['Year','Land_Cover_Type'], as_index=False)['Area_sq_km'].sum()
    pivot = yearly_lc.pivot(index='Year', columns='Land_Cover_Type', values='Area_sq_km').fillna(0)
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(go.Scatter(x=pivot.index, y=pivot[col], stackgroup='one', name=col, mode='none'))
    fig.update_layout(yaxis_title="Area (sq.km)", height=450)
    st.plotly_chart(fig, use_container_width=True)
    prop = pivot.div(pivot.sum(axis=1), axis=0)
    if prop.shape[0] >= 2:
        change = (prop.iloc[-1] - prop.iloc[0]).sort_values(ascending=False)
        top_change = change.index[0]
        st.markdown(f"**Insight:** Between {prop.index.min()} and {prop.index.max()}, **{top_change}** changed its share most (∆ {change.iloc[0]:.2%}).")
else:
    st.info("Required columns for stacked area missing.")

# 5) NDVI stats by land cover
st.subheader("5) NDVI Statistics by Land Cover")
if 'NDVI' in df_filtered.columns and 'Land_Cover_Type' in df_filtered.columns:
    ndvi_stats = df_filtered.groupby('Land_Cover_Type')['NDVI'].agg(['mean','median','std','min','max']).reset_index()
    ndvi_stats['mean'] = ndvi_stats['mean'].round(3)
    fig = px.bar(ndvi_stats, x='Land_Cover_Type', y='mean', error_y='std', hover_data=['median','min','max'])
    fig.update_layout(yaxis_title="Mean NDVI (±std)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    best = ndvi_stats.loc[ndvi_stats['mean'].idxmax()]
    worst = ndvi_stats.loc[ndvi_stats['mean'].idxmin()]
    st.markdown(f"**Insight:** Highest mean NDVI: **{best['Land_Cover_Type']}** ({best['mean']}). Lowest: **{worst['Land_Cover_Type']}** ({worst['mean']}).")
else:
    st.info("NDVI or Land_Cover_Type missing.")

# 6) Population Density boxplot
st.subheader("6) Population Density by Region (boxplot)")
if 'Population_Density' in df_filtered.columns and 'Region' in df_filtered.columns:
    top_regions = df_filtered['Region'].value_counts().head(12).index.tolist()
    pd_plot = df_filtered[df_filtered['Region'].isin(top_regions)]
    fig = px.box(pd_plot, x='Region', y='Population_Density', points='outliers')
    fig.update_layout(yaxis_title="Population Density", height=480)
    st.plotly_chart(fig, use_container_width=True)
    median_density = pd_plot.groupby('Region')['Population_Density'].median().sort_values(ascending=False).head(3)
    st.markdown("**Insight:** Top 3 regions by median population density:\n" + "\n".join([f"- {r}: {v:.1f}" for r,v in median_density.items()]))
else:
    st.info("Population_Density or Region missing.")

# 7) Temperature Anomaly trend by Region
st.subheader("7) Temperature Anomaly Trend by Region")
if 'Temperature_Anomaly' in df_filtered.columns and 'Year' in df_filtered.columns and 'Region' in df_filtered.columns:
    temp_trend = df_filtered.groupby(['Year','Region'], as_index=False)['Temperature_Anomaly'].mean()
    chosen_regions = sel_regions if sel_regions else temp_trend['Region'].unique()[:6]
    temp_plot = temp_trend[temp_trend['Region'].isin(chosen_regions)]
    fig = px.line(temp_plot, x='Year', y='Temperature_Anomaly', color='Region', markers=True)
    fig.update_layout(yaxis_title="Mean Temperature Anomaly (°C)", height=450)
    st.plotly_chart(fig, use_container_width=True)
    if temp_plot['Year'].nunique() >= 2:
        start = temp_plot[temp_plot['Year']==temp_plot['Year'].min()].set_index('Region')['Temperature_Anomaly']
        end = temp_plot[temp_plot['Year']==temp_plot['Year'].max()].set_index('Region')['Temperature_Anomaly']
        diff = (end - start).dropna().sort_values(ascending=False)
        if not diff.empty:
            top_r = diff.index[0]; top_v = diff.iloc[0]
            st.markdown(f"**Insight:** From {temp_plot['Year'].min()} to {temp_plot['Year'].max()}, **{top_r}** shows the largest increase in mean temperature anomaly ({top_v:.3f} °C).")
else:
    st.info("Temperature_Anomaly, Year or Region missing.")

# 8) Correlation heatmap
st.subheader("8) Correlation Heatmap (numeric variables)")
num_cols = [c for c in ['Area_sq_km','NDVI','Population_Density','Temperature_Anomaly'] if c in df_filtered.columns]
if len(num_cols) >= 2:
    corr = df_filtered[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    flat = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
    val = None
    for (a,b),v in flat.items():
        if a!=b:
            val = (a,b,v); break
    if val:
        st.markdown(f"**Insight:** Strongest correlation is between **{val[0]}** and **{val[1]}** (r = {corr.loc[val[0], val[1]]:.2f}).")
else:
    st.info("Not enough numeric columns for correlation.")

# 9) Urban Growth YoY %
st.subheader("9) Urban Growth YoY % (Top regions)")
if set(['Land_Cover_Type','Region','Year','Area_sq_km']).issubset(df_filtered.columns):
    urban = df_filtered[df_filtered['Land_Cover_Type'].str.lower()=='urban']
    if len(urban) > 0:
        urban_grp = urban.groupby(['Region','Year'], as_index=False)['Area_sq_km'].sum().sort_values(['Region','Year'])
        urban_grp['YoY_pct'] = urban_grp.groupby('Region')['Area_sq_km'].pct_change()*100
        region_growth = urban_grp.groupby('Region', as_index=False)['YoY_pct'].mean().replace([np.inf,-np.inf], np.nan).dropna().sort_values('YoY_pct', ascending=False)
        top_growth = region_growth.head(top_n)
        fig = px.bar(top_growth, x='Region', y='YoY_pct', text='YoY_pct')
        fig.update_layout(yaxis_title="Avg YoY Growth (%)", height=420)
        st.plotly_chart(fig, use_container_width=True)
        if len(top_growth)>0:
            st.markdown(f"**Insight:** Top region by average urban YoY growth: **{top_growth.iloc[0]['Region']}** ({top_growth.iloc[0]['YoY_pct']:.2f}% average).")
    else:
        st.info("No Urban records found in filtered data.")
else:
    st.info("Required columns for Urban growth missing.")

# 10) Area share composition over time
st.subheader("10) Area Share Composition over Time")
if set(['Year','Land_Cover_Type','Area_sq_km']).issubset(df_filtered.columns):
    yearly_lc2 = df_filtered.groupby(['Year','Land_Cover_Type'], as_index=False)['Area_sq_km'].sum()
    pivot2 = yearly_lc2.pivot(index='Year', columns='Land_Cover_Type', values='Area_sq_km').fillna(0)
    prop = pivot2.div(pivot2.sum(axis=1), axis=0)
    fig = go.Figure()
    for col in prop.columns:
        fig.add_trace(go.Scatter(x=prop.index, y=prop[col], stackgroup='one', name=col, mode='none'))
    fig.update_layout(yaxis_title="Proportion of total area", height=450)
    st.plotly_chart(fig, use_container_width=True)
    if prop.shape[0]>=2:
        change = (prop.iloc[-1] - prop.iloc[0]).abs().sort_values(ascending=False)
        st.markdown(f"**Insight:** Largest composition change between {prop.index.min()} and {prop.index.max()}: **{change.index[0]}** (∆ {change.iloc[0]:.2%}).")
else:
    st.info("Required columns for area share missing.")

# 11) NDVI distribution
st.subheader("11) NDVI Distribution")
if 'NDVI' in df_filtered.columns:
    fig = px.histogram(plot_df, x='NDVI', nbins=40)
    fig.update_layout(xaxis_title="NDVI", yaxis_title="Frequency", height=420)
    st.plotly_chart(fig, use_container_width=True)
    q1 = df_filtered['NDVI'].quantile(0.25)
    q3 = df_filtered['NDVI'].quantile(0.75)
    med = df_filtered['NDVI'].median()
    st.markdown(f"**Insight:** Median NDVI = {med:.3f}. IQR = [{q1:.3f}, {q3:.3f}].")
else:
    st.info("NDVI missing.")

# 12) NDVI vs Temperature Anomaly scatter
st.subheader("12) NDVI vs Temperature Anomaly (scatter)")
if set(['NDVI','Temperature_Anomaly']).issubset(plot_df.columns):
    fig = px.scatter(plot_df, x='NDVI', y='Temperature_Anomaly', hover_data=[c for c in ['Region','Year','Land_Cover_Type'] if c in plot_df.columns], height=480)
    fig.update_layout(xaxis_title="NDVI", yaxis_title="Temperature Anomaly (°C)")
    st.plotly_chart(fig, use_container_width=True)
    corr = plot_df[['NDVI','Temperature_Anomaly']].dropna().corr().iloc[0,1] if len(plot_df[['NDVI','Temperature_Anomaly']].dropna())>1 else np.nan
    st.markdown(f"**Insight:** Pearson r between NDVI and Temperature Anomaly ≈ {corr:.3f}.")
else:
    st.info("NDVI or Temperature_Anomaly missing.")

st.markdown("---")
if show_raw:
    st.subheader("Filtered data (first 1000 rows)")
    st.dataframe(df_filtered.head(1000))

csv = df_filtered.to_csv(index=False)
st.download_button("Download filtered CSV", data=csv, file_name="filtered_geoluc.csv", mime="text/csv")

