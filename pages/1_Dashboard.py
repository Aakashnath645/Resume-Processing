import streamlit as st           
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Database Helpers
def adapt_datetime(ts):
    return ts.isoformat()

def convert_datetime(val):
    return datetime.fromisoformat(val.decode())

sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("timestamp", convert_datetime)

# Database Operations
def clear_database():
    try:
        conn = sqlite3.connect('resume_analysis.db')
        c = conn.cursor()
        c.execute('DELETE FROM analyses')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error clearing database: {e}")
        return False

@st.cache_data
def load_data():
    conn = sqlite3.connect('resume_analysis.db', detect_types=sqlite3.PARSE_DECLTYPES)
    query = '''
        SELECT candidate_name, department, role, match_percentage, suitable
        FROM analyses
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data
def get_departments():
    conn = sqlite3.connect('resume_analysis.db', detect_types=sqlite3.PARSE_DECLTYPES)
    query = 'SELECT DISTINCT department FROM analyses WHERE department IS NOT NULL'
    departments = pd.read_sql_query(query, conn)['department'].tolist()
    conn.close()
    return ["All"] + sorted(departments)

# Page Configuration
st.set_page_config(
    page_title="Resume Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load custom CSS from shared style file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header with consistent styling
st.markdown('''
    <div class="header">
        <h1>📊 Resume Analysis Dashboard</h1>
        <p class="subtitle">Interactive Analytics & Insights</p>
    </div>
''', unsafe_allow_html=True)

# Control buttons with consistent styling
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    if st.button("🔄 Refresh", type="primary", key="refresh_data_btn"):
        st.cache_data.clear()
        st.rerun()

with col2:
    if st.button("🗑️ Clear Data", key="clear_data_btn"):
        if clear_database():
            st.success("Database cleared successfully!")
            st.cache_data.clear()
            st.rerun()

# Department filter with enhanced styling
departments = get_departments()
selected_dept = st.selectbox("📂 Select Department", departments, key='dept_filter')

# Load and filter data
df = load_data()
if selected_dept != "All":
    df = df[df['department'] == selected_dept]

# Dashboard Tabs
tab1, tab2 = st.tabs(["📈 Overview", "📋 Candidate List"])

with tab1:
    # Key Metrics with consistent card styling
    st.markdown('<div class="section-heading">Key Recruitment Metrics</div>', unsafe_allow_html=True)
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        total_apps = len(df)
        st.metric("Total Applications", total_apps)
    
    with metric_cols[1]:
        suitable_candidates = len(df[df['suitable'] == 'Yes'])
        st.metric("Suitable Candidates", suitable_candidates)
    
    with metric_cols[2]:
        success_rate = (suitable_candidates / total_apps * 100) if total_apps > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with metric_cols[3]:
        avg_match = df['match_percentage'].mean() if not df.empty else 0
        st.metric("Average Match", f"{avg_match:.1f}%")

    # Charts with consistent container styling
    st.markdown('<div class="section-heading">Recruitment Insights</div>', unsafe_allow_html=True)
    
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        dept_suitable = df.groupby('department').agg(
            {'suitable': lambda x: (x == 'Yes').mean() * 100}
        ).reset_index()
        
        fig_dept = px.bar(
            dept_suitable,
            x='suitable', y='department',
            orientation='h',
            title="Department Performance",
            color='suitable',
            color_continuous_scale=['#ff6b6b', '#51cf66']
        )
        fig_dept.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
            showlegend=False
        )
        st.plotly_chart(fig_dept, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_cols[1]:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df['match_percentage'],
            nbinsx=20,
            marker_color='#51cf66'
        ))
        fig_dist.update_layout(
            title="Match Score Distribution",
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
        st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Candidate List with enhanced styling
    st.markdown('<div class="section-heading">Candidate Overview</div>', unsafe_allow_html=True)
    
    if not df.empty:
        st.dataframe(
            df,
            column_config={
                "match_percentage": st.column_config.ProgressColumn(
                    "Match %",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                )
            },
            use_container_width=True
        )

        # Export buttons with consistent styling
        export_cols = st.columns(2)
        with export_cols[0]:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export to CSV",
                data=csv,
                file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="export_csv",
                use_container_width=True
            )
    else:
        st.info("No data available to display")
