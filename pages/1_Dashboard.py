import streamlit as st           
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

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
        SELECT 
            candidate_name, 
            department, 
            role, 
            match_percentage, 
            suitable,
            timestamp,
            detailed_analysis
        FROM analyses
        ORDER BY timestamp DESC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Parse detailed_analysis JSON if present
    try:
        df['analysis_dict'] = df['detailed_analysis'].apply(json.loads)
        # Extract skill scores if available
        df['technical_score'] = df['analysis_dict'].apply(lambda x: x.get('scores', {}).get('technical', 50))
        df['experience_score'] = df['analysis_dict'].apply(lambda x: x.get('scores', {}).get('experience', 50))
        df['leadership_score'] = df['analysis_dict'].apply(lambda x: x.get('scores', {}).get('leadership', 50))
        df['cultural_score'] = df['analysis_dict'].apply(lambda x: x.get('scores', {}).get('cultural', 50))
    except:
        # Set default scores if parsing fails
        df['technical_score'] = 50
        df['experience_score'] = 50
        df['leadership_score'] = 50
        df['cultural_score'] = 50
    
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

    st.markdown('### 📈 Trend Analysis')
    trend_cols = st.columns(2)
    
    with trend_cols[0]:
        # Hiring patterns over time
        df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
        monthly_apps = df.groupby('month').size().reset_index(name='applications')
        monthly_apps['month'] = monthly_apps['month'].astype(str)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_apps['month'],
            y=monthly_apps['applications'],
            mode='lines+markers',
            name='Applications',
            line=dict(color='#51cf66')
        ))
        fig_trend.update_layout(
            title='Monthly Application Trends',
            xaxis_title='Month',
            yaxis_title='Number of Applications',
            height=300,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with trend_cols[1]:
        # Success rate over time
        monthly_success = df.groupby('month').agg({
            'suitable': lambda x: (x == 'Yes').mean() * 100
        }).reset_index()
        monthly_success['month'] = monthly_success['month'].astype(str)
        
        fig_success = go.Figure()
        fig_success.add_trace(go.Scatter(
            x=monthly_success['month'],
            y=monthly_success['suitable'],
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='#339af0')
        ))
        fig_success.update_layout(
            title='Success Rate Trends',
            xaxis_title='Month',
            yaxis_title='Success Rate (%)',
            height=300,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_success, use_container_width=True)

    # Skills Analysis Section
    st.markdown('### 🎯 Skills Analysis')
    skill_cols = st.columns(2)
    
    with skill_cols[0]:
        # Radar chart for average skill scores
        avg_scores = {
            'Technical': df['technical_score'].mean(),
            'Experience': df['experience_score'].mean(),
            'Leadership': df['leadership_score'].mean(),
            'Cultural': df['cultural_score'].mean()
        }
        
        categories = list(avg_scores.keys())
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=list(avg_scores.values()),
            theta=categories,
            fill='toself',
            name='Average Skills'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='Average Skill Distribution',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with skill_cols[1]:
        # Bubble chart for experience vs technical skills
        fig_bubble = go.Figure()
        
        fig_bubble.add_trace(go.Scatter(
            x=df['technical_score'],
            y=df['experience_score'],
            mode='markers',
            marker=dict(
                size=df['match_percentage']/2,
                color=df['match_percentage'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Match %')
            ),
            text=df['role'],
            hovertemplate="Technical: %{x:.1f}<br>Experience: %{y:.1f}<br>Role: %{text}<br>Match: %{marker.color:.1f}%"
        ))
        
        fig_bubble.update_layout(
            title='Skills vs Experience Distribution',
            xaxis_title='Technical Score',
            yaxis_title='Experience Score',
            height=400
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    # Department Performance
    st.markdown('### 🏢 Department Analytics')
    dept_cols = st.columns(2)
    
    with dept_cols[0]:
        # Department hiring funnel
        dept_metrics = df.groupby('department').agg({
            'candidate_name': 'count',
            'suitable': lambda x: (x == 'Yes').sum()
        }).reset_index()
        
        fig_funnel = go.Figure(go.Funnel(
            name='Hiring Funnel',
            y=dept_metrics['department'],
            x=dept_metrics['candidate_name'],
            textinfo="value+percent initial",
            textposition="inside"
        ))
        
        fig_funnel.update_layout(
            title='Department Hiring Funnel',
            height=400
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with dept_cols[1]:
        # Heatmap of skills by department
        dept_skills = df.groupby('department').agg({
            'technical_score': 'mean',
            'experience_score': 'mean',
            'leadership_score': 'mean',
            'cultural_score': 'mean'
        }).reset_index()
        
        fig_heatmap = px.imshow(
            dept_skills.set_index('department'),
            color_continuous_scale='Viridis',
            aspect='auto',
            title='Department Skills Heatmap'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

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

    # Results card display with updated recommendation styling
    st.markdown("### Analysis Results")
    for i, res in enumerate(st.session_state.current_batch_results):
        # Clean and format candidate name
        candidate_name = res.get('Candidate Name', 'Unknown Candidate')
        if candidate_name.startswith('Candidate_'):
            # Remove 'Candidate_' prefix and replace underscores with spaces
            candidate_name = ' '.join(candidate_name.split('_')[1:]).title()
        
        # Determine recommendation style
        recommendation = res['Suitable']
        if recommendation == 'Yes':
            badge_color = 'success'
        elif recommendation == 'Further Evaluation Needed':
            badge_color = 'warning'
        else:
            badge_color = 'danger'
        
        # Get detailed scores
        detailed_scores = {}
        try:
            if 'detailed_analysis' in res:
                analysis_dict = json.loads(res['detailed_analysis'])
                if 'scores' in analysis_dict:
                    detailed_scores = analysis_dict['scores'].get('detailed', {})
        except:
            detailed_scores = {}
        
        with st.container():
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-header">
                        <h3>{candidate_name}</h3>
                        <div class="match-badge">{res.get('Match %', '0')}%</div>
                    </div>
                    <div class="result-content">
                        <p><strong>Department:</strong> {res.get('Department', 'N/A')}</p>
                        <p><strong>Role:</strong> {res.get('Role', 'N/A')}</p>
                        <div class="score-grid">
                            <div class="score-item">
                                <span class="score-label">Technical</span>
                                <div class="score-bar">
                                    <div class="score-fill" style="width: {detailed_scores.get('technical', 0)}%;"></div>
                                </div>
                                <span class="score-value">{detailed_scores.get('technical', 0)}%</span>
                            </div>
                            <div class="score-item">
                                <span class="score-label">Experience</span>
                                <div class="score-bar">
                                    <div class="score-fill" style="width: {detailed_scores.get('experience', 0)}%;"></div>
                                </div>
                                <span class="score-value">{detailed_scores.get('experience', 0)}%</span>
                            </div>
                            <div class="score-item">
                                <span class="score-label">Leadership</span>
                                <div class="score-bar">
                                    <div class="score-fill" style="width: {detailed_scores.get('leadership', 0)}%;"></div>
                                </div>
                                <span class="score-value">{detailed_scores.get('leadership', 0)}%</span>
                            </div>
                            <div class="score-item">
                                <span class="score-label">Cultural</span>
                                <div class="score-bar">
                                    <div class="score-fill" style="width: {detailed_scores.get('cultural', 0)}%;"></div>
                                </div>
                                <span class="score-value">{detailed_scores.get('cultural', 0)}%</span>
                            </div>
                        </div>
                        <p><strong>Recommendation:</strong> 
                            <span class="{badge_color}-badge">
                                {recommendation}
                            </span>
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
