import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import datetime
import io
import base64

# ==== RETRO GAMING STYLING ====
# Custom CSS for retro gaming appearance
def apply_retro_styling():
    # Import Google Fonts
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True
    )
    
    # Apply custom CSS
    st.markdown(
        """
        <style>
        /* Base styling */
        body {
            background-color: #0c0c0c;
            color: #33ff00;
            font-family: 'Space Mono', monospace;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'VT323', monospace !important;
            letter-spacing: 2px;
            text-shadow: 3px 3px 0px #ff00ff;
            color: #33ff00;
        }
        
        h1 {
            font-size: 3.5rem !important;
            border-bottom: 4px solid #ff00ff;
            padding-bottom: 10px;
        }
        
        h2 {
            font-size: 2.5rem !important;
            border-left: 4px solid #00ffff;
            padding-left: 10px;
        }
        
        h3 {
            font-size: 2rem !important;
            color: #ffff00;
        }
        
        /* Containers styling */
        .stApp {
            background-color: #0c0c0c;
        }
        
        .element-container, div.stButton, div.stDownloadButton {
            border: 2px solid #33ff00;
            border-radius: 0px !important;
            padding: 5px;
            box-shadow: 5px 5px 0px #ff00ff;
            margin-bottom: 25px !important;
            background-color: #1a1a1a;
        }
        
        /* Block style for metrics */
        div.css-1xarl3l {
            background-color: #1a1a1a;
            border: 2px solid #ffcc00;
            padding: 10px;
            box-shadow: 5px 5px 0px #00ffff;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1a1a1a;
            border-right: 4px solid #ff00ff;
        }
        
        /* Interactive controls */
        .stSlider, .stSelectbox, .stMultiselect {
            border: 2px solid #00ffff !important;
            padding: 5px !important;
            box-shadow: 4px 4px 0px #ff00ff !important;
            background-color: #1a1a1a !important;
        }
        
        .stSlider > div {
            background-color: #1a1a1a !important;
        }
        
        /* Pixel-style buttons */
        .stButton button {
            font-family: 'VT323', monospace !important;
            font-size: 1.5rem !important;
            background-color: #33ff00 !important;
            color: #000000 !important;
            border: 3px solid #ffffff !important;
            border-radius: 0px !important;
            box-shadow: 4px 4px 0px #ff00ff !important;
            padding: 5px 20px !important;
            transition: transform 0.1s !important;
        }
        
        .stButton button:active {
            transform: translate(4px, 4px) !important;
            box-shadow: none !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            font-family: 'VT323', monospace !important;
            font-size: 1.3rem !important;
            color: #ffcc00 !important;
            background-color: #1a1a1a !important;
            border: 2px solid #ffcc00 !important;
            border-radius: 0px !important;
        }
        
        .streamlit-expanderContent {
            border: 2px solid #ffcc00 !important;
            border-top: none !important;
            background-color: #1a1a1a !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-family: 'VT323', monospace !important;
            font-size: 1.2rem !important;
            background-color: #1a1a1a !important;
            border: 2px solid #00ffff !important;
            border-radius: 0px !important;
            padding: 5px 20px !important;
            color: #00ffff !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #00ffff !important;
            color: #000000 !important;
        }
        
        /* Tables */
        .stDataFrame {
            border: 3px solid #ffcc00 !important;
        }
        
        .stDataFrame table {
            border-collapse: separate !important;
            border-spacing: 2px !important;
            background-color: #1a1a1a !important;
        }
        
        .stDataFrame th {
            background-color: #ff00ff !important;
            color: #ffffff !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.2rem !important;
            padding: 8px !important;
            border: 2px solid #ffffff !important;
        }
        
        .stDataFrame td {
            background-color: #2a2a2a !important;
            color: #33ff00 !important;
            font-family: 'Space Mono', monospace !important;
            border: 1px solid #33ff00 !important;
            padding: 5px !important;
        }
        
        /* Text elements */
        p, div, span {
            font-family: 'Space Mono', monospace !important;
            color: #ffffff;
        }
        
        code {
            font-family: 'Space Mono', monospace !important;
            border: 1px solid #33ff00 !important;
            border-radius: 0px !important;
            background-color: #1a1a1a !important;
            padding: 5px !important;
            color: #00ffff !important;
        }
        
        .stAlert {
            border: 2px solid #ffcc00 !important;
            border-radius: 0px !important;
            box-shadow: 4px 4px 0px #ff00ff !important;
        }
        
        .stAlert p {
            color: #ffcc00 !important;
        }
        
        a {
            color: #00ffff !important;
            text-decoration: none !important;
            border-bottom: 2px solid #00ffff !important;
        }
        
        a:hover {
            color: #ff00ff !important;
            border-bottom: 2px solid #ff00ff !important;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1a1a1a;
            border: 2px solid #33ff00;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #33ff00;
            border: 2px solid #1a1a1a;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #00ffff;
        }
        
        /* Custom pixel divider */
        .pixel-divider {
            height: 5px;
            background: repeating-linear-gradient(
                to right,
                #ff00ff,
                #ff00ff 10px,
                #00ffff 10px,
                #00ffff 20px,
                #33ff00 20px,
                #33ff00 30px,
                #ffcc00 30px,
                #ffcc00 40px
            );
            margin: 30px 0;
        }
        
        /* Game-like info box */
        .game-info-box {
            border: 4px solid #33ff00;
            background-color: #1a1a1a;
            padding: 15px;
            margin: 20px 0;
            box-shadow: 8px 8px 0px #ff00ff;
            position: relative;
        }
        
        .game-info-box::before {
            content: "";
            position: absolute;
            top: -4px;
            left: -4px;
            right: -4px;
            bottom: -4px;
            border: 2px dashed #00ffff;
            pointer-events: none;
        }
        
        /* Game-like button styling */
        .game-button {
            display: inline-block;
            font-family: 'VT323', monospace !important;
            font-size: 1.5rem !important;
            background-color: #33ff00;
            color: #000000;
            border: 3px solid #ffffff;
            border-radius: 0px;
            box-shadow: 5px 5px 0px #ff00ff;
            padding: 10px 30px;
            margin: 10px 0;
            text-align: center;
            text-decoration: none;
            cursor: pointer;
            transition: transform 0.1s;
        }
        
        .game-button:hover {
            background-color: #00ffff;
        }
        
        .game-button:active {
            transform: translate(5px, 5px);
            box-shadow: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def pixel_divider():
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

def game_info_box(content):
    st.markdown(f'<div class="game-info-box">{content}</div>', unsafe_allow_html=True)

def game_header(text, level=1):
    st.markdown(f'<h{level} style="margin-bottom: 20px;">{text}</h{level}>', unsafe_allow_html=True)

# Retro color schemes for plots
RETRO_COLORS = [
    '#33ff00',  # Neon Green
    '#ff00ff',  # Magenta
    '#00ffff',  # Cyan
    '#ffcc00',  # Gold
    '#ff3377',  # Pink
    '#3377ff',  # Blue
    '#ff5500'   # Orange
]

# Set page configuration
st.set_page_config(
    page_title="Retro Customer Segmentation",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply the retro gaming styling
apply_retro_styling()

# =====================================
# === MAIN APPLICATION BEGINS HERE ===
# =====================================

# Function to generate sample data
def generate_sample_data(num_customers=200, num_transactions=1000):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create customer IDs
    customer_ids = [f"C{i:04d}" for i in range(1, num_customers + 1)]
    
    # Generate random transaction data
    today = datetime.datetime.now().date()
    
    # Timeframe: Last 365 days
    start_date = today - datetime.timedelta(days=365)
    
    transactions = []
    
    for i in range(num_transactions):
        # Random customer
        customer_id = np.random.choice(customer_ids)
        
        # Random date
        days_ago = np.random.randint(0, 365)
        transaction_date = today - datetime.timedelta(days=days_ago)
        
        # Random amount between $10 and $500, with distribution skewed towards lower values
        amount_spent = 10 + np.random.exponential(50)
        if amount_spent > 500:
            amount_spent = 500
            
        # Transaction ID
        transaction_id = f"T{i+1:06d}"
        
        transactions.append({
            'CustomerID': customer_id,
            'TransactionDate': transaction_date,
            'TransactionID': transaction_id,
            'AmountSpent': round(amount_spent, 2)
        })
    
    return pd.DataFrame(transactions)

# Function to calculate RFM metrics
def calculate_rfm(df, customer_id_col='CustomerID', transaction_date_col='TransactionDate', 
                 amount_col='AmountSpent', transaction_id_col='TransactionID', reference_date=None):
    
    # Convert TransactionDate to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[transaction_date_col]):
        df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])
    
    # Set reference date if not provided
    if reference_date is None:
        reference_date = df[transaction_date_col].max() + datetime.timedelta(days=1)
    elif not isinstance(reference_date, datetime.datetime) and not isinstance(reference_date, datetime.date):
        reference_date = pd.to_datetime(reference_date)
    
    # Calculate RFM metrics
    rfm = df.groupby(customer_id_col).agg({
        transaction_date_col: lambda x: (reference_date - x.max()).days,  # Recency
        transaction_id_col: 'count',  # Frequency
        amount_col: 'sum'   # Monetary
    }).reset_index()
    
    # Rename columns
    rfm.columns = [customer_id_col, 'Recency', 'Frequency', 'Monetary']
    
    return rfm

# Function to apply clustering
def apply_kmeans_clustering(rfm_data, n_clusters=4, features=None):
    if features is None:
        features = ['Recency', 'Frequency', 'Monetary']
    
    # Extract the features to use
    X = rfm_data[features].copy()
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_data['Segment'] = kmeans.fit_predict(X_scaled)
    
    # Get cluster centers
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                                 columns=features)
    
    # Add segment label
    cluster_centers['Segment'] = range(n_clusters)
    
    return rfm_data, cluster_centers, X_scaled

# Function to create segment profiles based on RFM values
def create_segment_profiles(rfm_segmented, cluster_centers):
    # Create segment profiles
    segment_profiles = {}
    
    # Get the segment with best (lowest) recency
    best_recency_segment = cluster_centers['Segment'][cluster_centers['Recency'].idxmin()]
    
    # Get the segment with worst (highest) recency
    worst_recency_segment = cluster_centers['Segment'][cluster_centers['Recency'].idxmax()]
    
    # Get the segment with best (highest) frequency & monetary
    high_value_segment = cluster_centers['Segment'][
        (cluster_centers['Frequency'] * cluster_centers['Monetary']).idxmax()
    ]
    
    # Get the segment with lowest frequency & monetary
    low_value_segment = cluster_centers['Segment'][
        (cluster_centers['Frequency'] * cluster_centers['Monetary']).idxmin()
    ]
    
    # Retro gaming character names for segments
    character_names = [
        "LEGENDARY PLAYERS",
        "ACTIVE GAMERS",
        "CASUAL PLAYERS",
        "LOST PLAYERS",
        "NEW CHALLENGERS",
        "BOSS CHARACTERS",
        "SIDE QUEST USERS"
    ]
    
    # Assign names based on segment characteristics
    for segment in range(len(cluster_centers)):
        recency = cluster_centers.loc[cluster_centers['Segment'] == segment, 'Recency'].values[0]
        frequency = cluster_centers.loc[cluster_centers['Segment'] == segment, 'Frequency'].values[0]
        monetary = cluster_centers.loc[cluster_centers['Segment'] == segment, 'Monetary'].values[0]
        
        if segment == high_value_segment:
            name = character_names[0]  # "LEGENDARY PLAYERS"
            description = "Top-level gamers who play frequently and spend lots of coins. Your VIPs!"
            character = "🏆"
        elif segment == best_recency_segment and segment != high_value_segment:
            name = character_names[1]  # "ACTIVE GAMERS"
            description = "Recently active but not yet legendary status. Ready for level-up offers."
            character = "🎮"
        elif segment == worst_recency_segment and segment != low_value_segment:
            name = character_names[3]  # "LOST PLAYERS"
            description = "Haven't been seen in the game for a while. Send a power-up to bring them back!"
            character = "👻"
        elif segment == low_value_segment:
            name = character_names[2]  # "CASUAL PLAYERS"
            description = "Play occasionally with minimal coin spending. Need special quests to engage more."
            character = "🔍"
        else:
            idx = min(4 + segment, len(character_names) - 1)
            name = character_names[idx]
            description = "Players with mixed gaming patterns. Need more data to classify."
            character = "❓"
        
        segment_profiles[segment] = {
            'name': name,
            'description': description,
            'avg_recency': recency,
            'avg_frequency': frequency,
            'avg_monetary': monetary,
            'character': character
        }
    
    return segment_profiles

# Title with game-like header
game_header("CUSTOMER SEGMENT EXPLORER 8-BIT EDITION", level=1)

# Game-like intro text
game_info_box("""
<p style="font-family: 'VT323', monospace; font-size: 1.5rem; color: #33ff00;">
WELCOME, RETAIL ANALYTICS WARRIOR! 🎮<br>
YOUR MISSION: DISCOVER THE HIDDEN CUSTOMER SEGMENTS AND MAXIMIZE YOUR SCORE!<br>
USE THE CONTROLS ON THE LEFT TO CONFIGURE YOUR EXPLORATION.<br>
PRESS START TO BEGIN YOUR JOURNEY!
</p>
""")

pixel_divider()

# Sidebar for data loading and configuration
st.sidebar.markdown("""
<h2 style="text-align: center; border-bottom: 3px solid #ff00ff; padding-bottom: 10px;">
GAME CONTROLS 🕹️
</h2>
""", unsafe_allow_html=True)

# Data loading options
data_option = st.sidebar.radio(
    "SELECT YOUR DATA SOURCE:",
    ["USE SAMPLE DATA 💾", "UPLOAD YOUR OWN DATA 📂"]
)

if data_option == "USE SAMPLE DATA 💾":
    # Generate sample data
    df = generate_sample_data()
    st.sidebar.success("✅ SAMPLE DATA LOADED! PRESS START!")
else:
    # Upload data
    uploaded_file = st.sidebar.file_uploader("UPLOAD YOUR DATA FILE (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ DATA LOADED SUCCESSFULLY! PRESS START!")
        except Exception as e:
            st.sidebar.error(f"ERROR: {e}")
            df = None
    else:
        st.info("PLEASE INSERT COIN (UPLOAD A CSV FILE) OR SELECT 'USE SAMPLE DATA'")
        df = None

# Continue only if data is loaded
if df is not None:
    # Display raw data sample with retro styling
    with st.expander("VIEW RAW TRANSACTION DATA"):
        st.dataframe(df.head(10))
        st.info(f"LOADED {df.shape[0]} TRANSACTIONS FROM {df['CustomerID'].nunique()} UNIQUE CUSTOMERS")
    
    # Sidebar configurations for RFM analysis with gaming theme
    st.sidebar.markdown("""
    <h3 style="text-align: center; border-bottom: 2px solid #33ff00; padding-bottom: 5px;">
    RFM POWER-UPS ⚡
    </h3>
    """, unsafe_allow_html=True)
    
    # Column selection for RFM calculation
    cols = df.columns.tolist()
    
    # Function to safely create selectbox options with fallback
    def get_column_options(cols, filters, default_col):
        # Filter columns based on criteria
        options = [col for col in cols if any(f in col.lower() for f in filters)]
        # If no columns match filter criteria, use all columns
        if not options:
            options = cols
        # Determine the index
        if default_col in options:
            idx = options.index(default_col)
        else:
            idx = 0
        return options, idx
    
    # Customer ID column selection
    customer_options, customer_idx = get_column_options(
        cols, ['customer', 'id'], 'CustomerID'
    )
    customer_id_col = st.sidebar.selectbox(
        "PLAYER ID COLUMN", 
        options=customer_options,
        index=customer_idx
    )
    
    # Transaction ID column selection
    transaction_options, transaction_idx = get_column_options(
        cols, ['transaction', 'id'], 'TransactionID'
    )
    transaction_id_col = st.sidebar.selectbox(
        "TRANSACTION ID COLUMN", 
        options=transaction_options,
        index=transaction_idx
    )
    
    # Transaction Date column selection
    date_options, date_idx = get_column_options(
        cols, ['date'], 'TransactionDate'
    )
    transaction_date_col = st.sidebar.selectbox(
        "PLAY DATE COLUMN", 
        options=date_options,
        index=date_idx
    )
    
    # Amount column selection
    amount_options, amount_idx = get_column_options(
        cols, ['amount', 'spent', 'value'], 'AmountSpent'
    )
    amount_col = st.sidebar.selectbox(
        "COINS SPENT COLUMN", 
        options=amount_options,
        index=amount_idx
    )
    
    # Calculate RFM
    rfm = calculate_rfm(df, customer_id_col, transaction_date_col, amount_col, transaction_id_col=transaction_id_col)
    
    # Clustering settings with gaming theme
    st.sidebar.markdown("""
    <h3 style="text-align: center; border-bottom: 2px solid #00ffff; padding-bottom: 5px;">
    SEGMENTATION LEVEL SELECT 🎯
    </h3>
    """, unsafe_allow_html=True)
    
    n_clusters = st.sidebar.slider("NUMBER OF PLAYER CLASSES", min_value=2, max_value=7, value=4)
    
    # Feature selection for clustering
    features_for_clustering = st.sidebar.multiselect(
        "POWER METRICS TO USE",
        options=['Recency', 'Frequency', 'Monetary'],
        default=['Recency', 'Frequency', 'Monetary']
    )
    
    if not features_for_clustering:
        st.sidebar.warning("SELECT AT LEAST ONE METRIC TO CONTINUE YOUR QUEST!")
        features_for_clustering = ['Recency', 'Frequency', 'Monetary']
    
    # Apply clustering
    rfm_segmented, cluster_centers, X_scaled = apply_kmeans_clustering(rfm, n_clusters, features_for_clustering)
    
    # Create segment profiles
    segment_profiles = create_segment_profiles(rfm_segmented, cluster_centers)
    
    # ---- Main content with retro gaming theme----
    game_header("RFM PLAYER STATS", level=2)
    
    # Retro-styled metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("DAYS SINCE LAST PLAY", f"{rfm['Recency'].mean():.1f}")
        st.info("LOWER IS BETTER - RECENT PLAYERS ARE MORE ENGAGED")
    
    with col2:
        st.metric("AVG PLAY COUNT", f"{rfm['Frequency'].mean():.1f}")
        st.info("HIGHER IS BETTER - FREQUENT PLAYERS EARN MORE XP")
    
    with col3:
        st.metric("AVG COINS SPENT", f"${rfm['Monetary'].mean():.2f}")
        st.info("HIGHER IS BETTER - BIG SPENDERS UNLOCK SPECIAL ITEMS")
    
    pixel_divider()
    
    # Display RFM distributions with retro styling
    game_header("PLAYER METRICS DISTRIBUTION", level=2)
    
    tab1, tab2, tab3 = st.tabs(["RECENCY RADAR", "FREQUENCY METER", "MONETARY COUNTER"])
    
    # Define a retro gaming colorscale for histograms
    retro_colorscale = [
        [0, "#1a1a1a"],  # Dark background
        [0.5, "#00ffff"],  # Cyan midpoint
        [1, "#33ff00"]  # Neon green max
    ]
    
    with tab1:
        fig_recency = px.histogram(rfm, x='Recency', title="DAYS SINCE LAST PLAY", 
                                  labels={'Recency': 'DAYS SINCE LAST PLAY'})
        fig_recency.update_traces(marker_color="#33ff00", marker_line_color="#ff00ff", 
                                marker_line_width=1.5, opacity=0.7)
        fig_recency.update_layout(
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font=dict(family="VT323", size=18, color="#ffffff"),
            title=dict(font=dict(family="VT323", size=24, color="#33ff00")),
            xaxis=dict(
                title=dict(font=dict(family="VT323", size=18, color="#00ffff")),
                gridcolor="#333333",
                zerolinecolor="#00ffff"
            ),
            yaxis=dict(
                title=dict(text="PLAYER COUNT", font=dict(family="VT323", size=18, color="#00ffff")),
                gridcolor="#333333",
                zerolinecolor="#00ffff"
            )
        )
        st.plotly_chart(fig_recency, use_container_width=True)
    
    with tab2:
        fig_frequency = px.histogram(rfm, x='Frequency', title="NUMBER OF PLAYS", 
                                    labels={'Frequency': 'NUMBER OF PLAYS'})
        fig_frequency.update_traces(marker_color="#ff00ff", marker_line_color="#33ff00", 
                                  marker_line_width=1.5, opacity=0.7)
        fig_frequency.update_layout(
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font=dict(family="VT323", size=18, color="#ffffff"),
            title=dict(font=dict(family="VT323", size=24, color="#ff00ff")),
            xaxis=dict(
                title=dict(font=dict(family="VT323", size=18, color="#00ffff")),
                gridcolor="#333333",
                zerolinecolor="#00ffff"
            ),
            yaxis=dict(
                title=dict(text="PLAYER COUNT", font=dict(family="VT323", size=18, color="#00ffff")),
                gridcolor="#333333",
                zerolinecolor="#00ffff"
            )
        )
        st.plotly_chart(fig_frequency, use_container_width=True)
    
    with tab3:
        fig_monetary = px.histogram(rfm, x='Monetary', title="TOTAL COINS SPENT", 
                                   labels={'Monetary': 'TOTAL COINS SPENT ($)'})
        fig_monetary.update_traces(marker_color="#00ffff", marker_line_color="#ffcc00", 
                                 marker_line_width=1.5, opacity=0.7)
        fig_monetary.update_layout(
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font=dict(family="VT323", size=18, color="#ffffff"),
            title=dict(font=dict(family="VT323", size=24, color="#00ffff")),
            xaxis=dict(
                title=dict(font=dict(family="VT323", size=18, color="#ffcc00")),
                gridcolor="#333333",
                zerolinecolor="#00ffff"
            ),
            yaxis=dict(
                title=dict(text="PLAYER COUNT", font=dict(family="VT323", size=18, color="#ffcc00")),
                gridcolor="#333333",
                zerolinecolor="#00ffff"
            )
        )
        st.plotly_chart(fig_monetary, use_container_width=True)
    
    pixel_divider()
    
    # Customer Segmentation with gaming theme
    game_header("PLAYER CLASSES DISCOVERED!", level=2)
    
    # Display segmentation summary
    segment_summary = rfm_segmented.groupby('Segment').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    segment_summary.columns = ['Segment', 'Number of Players', 'Avg. Days Inactive', 
                              'Avg. Play Count', 'Avg. Coins Spent']
    
    # Add segment names
    segment_summary['Player Class'] = segment_summary['Segment'].apply(lambda x: segment_profiles[x]['name'])
    segment_summary['Character'] = segment_summary['Segment'].apply(lambda x: segment_profiles[x]['character'])
    
    # Reorder columns
    segment_summary = segment_summary[['Segment', 'Character', 'Player Class', 'Number of Players', 
                                      'Avg. Days Inactive', 'Avg. Play Count', 
                                      'Avg. Coins Spent']]
    
    st.subheader("PLAYER CLASS STATS")
    st.dataframe(segment_summary.style.format({
        'Avg. Days Inactive': '{:.1f}',
        'Avg. Play Count': '{:.1f}',
        'Avg. Coins Spent': '${:.2f}'
    }))
    
    # Segment Size Visualization with retro styling
    st.subheader("PLAYER CLASS DISTRIBUTION")
    fig_segment_size = px.pie(segment_summary, values='Number of Players', names='Player Class',
                            title='PLAYER DISTRIBUTION ACROSS CLASSES')
    
    fig_segment_size.update_traces(marker=dict(colors=RETRO_COLORS[:n_clusters]), 
                                textfont=dict(family="VT323", size=16, color="#000000"),
                                textinfo="percent+label")
    
    fig_segment_size.update_layout(
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(family="VT323", size=18, color="#ffffff"),
        title=dict(font=dict(family="VT323", size=24, color="#ffcc00")),
        legend=dict(
            font=dict(family="VT323", size=16, color="#ffffff"),
            bgcolor="#2a2a2a",
            bordercolor="#33ff00",
            borderwidth=2
        )
    )
    
    st.plotly_chart(fig_segment_size, use_container_width=True)
    
    pixel_divider()
    
    # Segment visualization with retro gaming style
    game_header("PLAYER CLASS MAP", level=2)
    
    viz_type = st.radio(
        "SELECT YOUR VIEWING MODE:",
        ["3D WORLD MAP", "2D LEVEL MAPS", "PCA RADAR"]
    )
    
    # Retro styling for 3D plots
    def style_3d_plot(fig):
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title=dict(text="RECENCY", font=dict(family="VT323", size=16, color="#33ff00")),
                    gridcolor="#333333",
                    zerolinecolor="#00ffff",
                    backgroundcolor="#1a1a1a"
                ),
                yaxis=dict(
                    title=dict(text="FREQUENCY", font=dict(family="VT323", size=16, color="#ff00ff")),
                    gridcolor="#333333",
                    zerolinecolor="#00ffff",
                    backgroundcolor="#1a1a1a"
                ),
                zaxis=dict(
                    title=dict(text="MONETARY", font=dict(family="VT323", size=16, color="#ffcc00")),
                    gridcolor="#333333",
                    zerolinecolor="#00ffff",
                    backgroundcolor="#1a1a1a"
                ),
                bgcolor="#1a1a1a"
            ),
            font=dict(family="VT323", size=16, color="#ffffff"),
            title=dict(font=dict(family="VT323", size=24, color="#33ff00")),
            paper_bgcolor="#1a1a1a",
            legend=dict(
                font=dict(family="VT323", size=14, color="#ffffff"),
                bgcolor="#2a2a2a",
                bordercolor="#33ff00",
                borderwidth=2
            )
        )
        return fig
    
    # Retro styling for 2D plots
    def style_2d_plot(fig, x_color="#33ff00", y_color="#ff00ff"):
        fig.update_layout(
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font=dict(family="VT323", size=16, color="#ffffff"),
            title=dict(font=dict(family="VT323", size=20, color="#ffcc00")),
            xaxis=dict(
                title=dict(font=dict(family="VT323", size=16, color=x_color)),
                gridcolor="#333333",
                zerolinecolor="#00ffff"
            ),
            yaxis=dict(
                title=dict(font=dict(family="VT323", size=16, color=y_color)),
                gridcolor="#333333",
                zerolinecolor="#00ffff"
            ),
            legend=dict(
                font=dict(family="VT323", size=14, color="#ffffff"),
                bgcolor="#2a2a2a",
                bordercolor="#33ff00",
                borderwidth=2
            )
        )
        return fig
    
    if viz_type == "3D WORLD MAP" and 'Recency' in features_for_clustering and 'Frequency' in features_for_clustering and 'Monetary' in features_for_clustering:
        fig_3d = px.scatter_3d(
            rfm_segmented, x='Recency', y='Frequency', z='Monetary',
            color='Segment', 
            color_discrete_sequence=RETRO_COLORS,
            opacity=0.8,
            labels={'Recency': 'DAYS INACTIVE', 
                   'Frequency': 'PLAY COUNT', 
                   'Monetary': 'COINS SPENT'},
            title="3D MAP OF PLAYER CLASSES"
        )
        
        # Add cluster centers with diamond symbols
        for i, center in cluster_centers.iterrows():
            fig_3d.add_scatter3d(
                x=[center['Recency']], 
                y=[center['Frequency']], 
                z=[center['Monetary']],
                mode='markers',
                marker=dict(color=RETRO_COLORS[i % len(RETRO_COLORS)], size=12, symbol='diamond'),
                name=f"BASE: {segment_profiles[i]['name']}"
            )
        
        # Apply retro styling
        fig_3d = style_3d_plot(fig_3d)
        fig_3d.update_layout(height=700)
        st.plotly_chart(fig_3d, use_container_width=True)
        
    elif viz_type == "2D LEVEL MAPS":
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Recency' in features_for_clustering and 'Frequency' in features_for_clustering:
                fig_rf = px.scatter(
                    rfm_segmented, x='Recency', y='Frequency',
                    color='Segment', 
                    color_discrete_sequence=RETRO_COLORS,
                    labels={'Recency': 'DAYS INACTIVE', 
                           'Frequency': 'PLAY COUNT'},
                    title="ACTIVITY vs FREQUENCY MAP"
                )
                
                # Add cluster centers with star symbols
                for i, center in cluster_centers.iterrows():
                    fig_rf.add_scatter(
                        x=[center['Recency']], 
                        y=[center['Frequency']],
                        mode='markers',
                        marker=dict(color=RETRO_COLORS[i % len(RETRO_COLORS)], size=18, symbol='star'),
                        name=f"BASE {i}"
                    )
                
                # Apply retro styling
                fig_rf = style_2d_plot(fig_rf, x_color="#33ff00", y_color="#ff00ff")
                st.plotly_chart(fig_rf, use_container_width=True)
        
        with col2:
            if 'Frequency' in features_for_clustering and 'Monetary' in features_for_clustering:
                fig_fm = px.scatter(
                    rfm_segmented, x='Frequency', y='Monetary',
                    color='Segment', 
                    color_discrete_sequence=RETRO_COLORS,
                    labels={'Frequency': 'PLAY COUNT', 
                           'Monetary': 'COINS SPENT'},
                    title="FREQUENCY vs SPENDING MAP"
                )
                
                # Add cluster centers with star symbols
                for i, center in cluster_centers.iterrows():
                    fig_fm.add_scatter(
                        x=[center['Frequency']], 
                        y=[center['Monetary']],
                        mode='markers',
                        marker=dict(color=RETRO_COLORS[i % len(RETRO_COLORS)], size=18, symbol='star'),
                        name=f"BASE {i}"
                    )
                
                # Apply retro styling
                fig_fm = style_2d_plot(fig_fm, x_color="#ff00ff", y_color="#ffcc00")
                st.plotly_chart(fig_fm, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            if 'Recency' in features_for_clustering and 'Monetary' in features_for_clustering:
                fig_rm = px.scatter(
                    rfm_segmented, x='Recency', y='Monetary',
                    color='Segment', 
                    color_discrete_sequence=RETRO_COLORS,
                    labels={'Recency': 'DAYS INACTIVE', 
                           'Monetary': 'COINS SPENT'},
                    title="ACTIVITY vs SPENDING MAP"
                )
                
                # Add cluster centers with star symbols
                for i, center in cluster_centers.iterrows():
                    fig_rm.add_scatter(
                        x=[center['Recency']], 
                        y=[center['Monetary']],
                        mode='markers',
                        marker=dict(color=RETRO_COLORS[i % len(RETRO_COLORS)], size=18, symbol='star'),
                        name=f"BASE {i}"
                    )
                
                # Apply retro styling
                fig_rm = style_2d_plot(fig_rm, x_color="#33ff00", y_color="#ffcc00")
                st.plotly_chart(fig_rm, use_container_width=True)
    
    else:  # PCA Radar
        # Apply PCA if we have more than 1 feature
        if len(features_for_clustering) > 1:
            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            
            # Create a DataFrame with PCA results
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            pca_df['Segment'] = rfm_segmented['Segment']
            
            # Apply PCA to cluster centers
            cluster_centers_scaled = StandardScaler().fit_transform(cluster_centers[features_for_clustering])
            centers_pca = pca.transform(cluster_centers_scaled)
            
            # Create PCA plot with retro styling
            fig_pca = px.scatter(
                pca_df, x='PC1', y='PC2',
                color='Segment', 
                color_discrete_sequence=RETRO_COLORS,
                title="PLAYER CLASS RADAR"
            )
            
            # Add cluster centers with star symbols
            for i, center in enumerate(centers_pca):
                fig_pca.add_scatter(
                    x=[center[0]], 
                    y=[center[1]],
                    mode='markers',
                    marker=dict(color=RETRO_COLORS[i % len(RETRO_COLORS)], size=18, symbol='star'),
                    name=f"BASE: {segment_profiles[i]['name']}"
                )
            
            # Apply retro styling
            fig_pca = style_2d_plot(fig_pca, x_color="#00ffff", y_color="#ff00ff")
            
            # Explained variance
            explained_var = pca.explained_variance_ratio_
            st.info(f"RADAR ACCURACY: PC1 = {explained_var[0]:.2%}, PC2 = {explained_var[1]:.2%}")
            
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.warning("PCA RADAR REQUIRES AT LEAST 2 METRICS. PLEASE SELECT MORE POWER-UPS!")
    
    pixel_divider()
    
    # Segment Profiling with game character theme
    game_header("PLAYER CLASS CARDS", level=2)
    
    # Use columns to create a grid of player cards
    cols = st.columns(min(4, n_clusters))
    
    # Define background colors for cards
    card_colors = ["#1a1a1a", "#222222", "#1a1a1a", "#222222"]
    border_colors = RETRO_COLORS[:n_clusters]
    
    for i, segment in enumerate(range(n_clusters)):
        col_idx = i % len(cols)
        with cols[col_idx]:
            # Create a pixel-art style card for each segment
            card_html = f"""
            <div style="border: 3px solid {border_colors[i]}; background-color: {card_colors[i % len(card_colors)]}; 
                        padding: 15px; margin-bottom: 20px; box-shadow: 5px 5px 0px #ff00ff;">
                <h3 style="font-family: 'VT323', monospace; font-size: 28px; color: {border_colors[i]}; text-align: center; 
                           margin-bottom: 10px; text-shadow: 2px 2px #1a1a1a;">
                    {segment_profiles[segment]['character']} {segment_profiles[segment]['name']}
                </h3>
                <div style="border-top: 2px dashed {border_colors[i]}; margin: 10px 0;"></div>
                <p style="font-family: 'VT323', monospace; font-size: 18px; color: #ffffff; margin: 5px 0;">
                    <span style="color: #33ff00;">CLASS ID:</span> {segment}
                </p>
                <p style="font-family: 'VT323', monospace; font-size: 18px; color: #ffffff; margin: 5px 0;">
                    <span style="color: #33ff00;">PLAYERS:</span> {(rfm_segmented['Segment'] == segment).sum()} 
                    ({(rfm_segmented['Segment'] == segment).sum() / len(rfm_segmented) * 100:.1f}%)
                </p>
                <p style="font-family: 'VT323', monospace; font-size: 18px; color: #ffffff; margin: 5px 0;">
                    <span style="color: #33ff00;">DAYS INACTIVE:</span> {cluster_centers.loc[cluster_centers['Segment'] == segment, 'Recency'].values[0]:.1f}
                </p>
                <p style="font-family: 'VT323', monospace; font-size: 18px; color: #ffffff; margin: 5px 0;">
                    <span style="color: #33ff00;">PLAY COUNT:</span> {cluster_centers.loc[cluster_centers['Segment'] == segment, 'Frequency'].values[0]:.1f}
                </p>
                <p style="font-family: 'VT323', monospace; font-size: 18px; color: #ffffff; margin: 5px 0;">
                    <span style="color: #33ff00;">COINS SPENT:</span> ${cluster_centers.loc[cluster_centers['Segment'] == segment, 'Monetary'].values[0]:.2f}
                </p>
                <div style="border-top: 2px dashed {border_colors[i]}; margin: 10px 0;"></div>
                <p style="font-family: 'Space Mono', monospace; font-size: 14px; color: #ffffff; margin-top: 10px;">
                    {segment_profiles[segment]['description']}
                </p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    
    pixel_divider()
    
    # Strategy recommendations based on segments with gaming theme
    game_header("VICTORY STRATEGIES", level=2)
    
    game_info_box("""
    <p style="font-family: 'VT323', monospace; font-size: 1.2rem; color: #ffcc00;">
    UNLOCK SPECIAL MARKETING POWER-UPS FOR EACH PLAYER CLASS!<br>
    SELECT A CLASS BELOW TO VIEW RECOMMENDED STRATEGIES.
    </p>
    """)
    
    # Gaming themed strategy cards
    for segment in range(n_clusters):
        with st.expander(f"{segment_profiles[segment]['character']} STRATEGY FOR {segment_profiles[segment]['name']} (CLASS {segment})"):
            recency = cluster_centers.loc[cluster_centers['Segment'] == segment, 'Recency'].values[0]
            frequency = cluster_centers.loc[cluster_centers['Segment'] == segment, 'Frequency'].values[0]
            monetary = cluster_centers.loc[cluster_centers['Segment'] == segment, 'Monetary'].values[0]
            
            # Generate custom recommendations based on segment characteristics
            if "LEGENDARY PLAYERS" in segment_profiles[segment]['name']:
                strategy_html = """
                <div style="background-color: #1a1a1a; border: 3px solid #33ff00; padding: 20px; margin: 10px 0;">
                    <h3 style="font-family: 'VT323', monospace; font-size: 1.8rem; color: #33ff00; border-bottom: 2px solid #ff00ff; padding-bottom: 5px;">
                        🏆 LEGENDARY PLAYER STRATEGY 🏆
                    </h3>
                    
                    <h4 style="font-family: 'VT323', monospace; font-size: 1.4rem; color: #ffcc00; margin-top: 15px;">
                        RECOMMENDED POWER-UPS:
                    </h4>
                    <ul style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff; list-style-type: none; padding-left: 0;">
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #33ff00;">➤</span>
                            <strong style="color: #00ffff;">VIP ACCESS:</strong> Exclusive content and early access to new releases
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #33ff00;">➤</span>
                            <strong style="color: #00ffff;">PREMIUM REWARDS:</strong> Special loyalty bonuses with premium benefits
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #33ff00;">➤</span>
                            <strong style="color: #00ffff;">PERSONAL QUESTS:</strong> Tailored recommendations based on purchase history
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #33ff00;">➤</span>
                            <strong style="color: #00ffff;">REFER-A-PLAYER:</strong> Special bonuses for bringing new players into the game
                        </li>
                    </ul>
                    
                    <div style="border-top: 2px dashed #ff00ff; margin: 15px 0;"></div>
                    
                    <h4 style="font-family: 'VT323', monospace; font-size: 1.4rem; color: #ffcc00;">
                        MISSION OBJECTIVE:
                    </h4>
                    <p style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff;">
                        Maintain loyalty and maximize lifetime value. These are your most valuable players!
                    </p>
                </div>
                """
                st.markdown(strategy_html, unsafe_allow_html=True)
                
            elif "ACTIVE GAMERS" in segment_profiles[segment]['name']:
                strategy_html = """
                <div style="background-color: #1a1a1a; border: 3px solid #ff00ff; padding: 20px; margin: 10px 0;">
                    <h3 style="font-family: 'VT323', monospace; font-size: 1.8rem; color: #ff00ff; border-bottom: 2px solid #33ff00; padding-bottom: 5px;">
                        🎮 ACTIVE PLAYER STRATEGY 🎮
                    </h3>
                    
                    <h4 style="font-family: 'VT323', monospace; font-size: 1.4rem; color: #ffcc00; margin-top: 15px;">
                        RECOMMENDED POWER-UPS:
                    </h4>
                    <ul style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff; list-style-type: none; padding-left: 0;">
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ff00ff;">➤</span>
                            <strong style="color: #00ffff;">ENGAGEMENT BOOST:</strong> Follow-up emails with personalized recommendations
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ff00ff;">➤</span>
                            <strong style="color: #00ffff;">LEVEL-UP BONUS:</strong> Special offer to encourage second purchase
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ff00ff;">➤</span>
                            <strong style="color: #00ffff;">FEEDBACK QUEST:</strong> Ask for product reviews to build relationship
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ff00ff;">➤</span>
                            <strong style="color: #00ffff;">SOCIAL CONNECT:</strong> Encourage following on social platforms
                        </li>
                    </ul>
                    
                    <div style="border-top: 2px dashed #33ff00; margin: 15px 0;"></div>
                    
                    <h4 style="font-family: 'VT323', monospace; font-size: 1.4rem; color: #ffcc00;">
                        MISSION OBJECTIVE:
                    </h4>
                    <p style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff;">
                        Convert to regular players and increase engagement frequency. These players have potential to level up!
                    </p>
                </div>
                """
                st.markdown(strategy_html, unsafe_allow_html=True)
                
            elif "LOST PLAYERS" in segment_profiles[segment]['name']:
                strategy_html = """
                <div style="background-color: #1a1a1a; border: 3px solid #ffcc00; padding: 20px; margin: 10px 0;">
                    <h3 style="font-family: 'VT323', monospace; font-size: 1.8rem; color: #ffcc00; border-bottom: 2px solid #00ffff; padding-bottom: 5px;">
                        👻 LOST PLAYER RECOVERY STRATEGY 👻
                    </h3>
                    
                    <h4 style="font-family: 'VT323', monospace; font-size: 1.4rem; color: #00ffff; margin-top: 15px;">
                        RECOMMENDED POWER-UPS:
                    </h4>
                    <ul style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff; list-style-type: none; padding-left: 0;">
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ffcc00;">➤</span>
                            <strong style="color: #33ff00;">REACTIVATION SPELL:</strong> "We miss you" email with special discount
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ffcc00;">➤</span>
                            <strong style="color: #33ff00;">NEW CONTENT ALERT:</strong> Highlight new offerings since their last purchase
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ffcc00;">➤</span>
                            <strong style="color: #33ff00;">SURVEY QUEST:</strong> Understand why they stopped playing
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ffcc00;">➤</span>
                            <strong style="color: #33ff00;">REVIVAL POTION:</strong> Limited-time significant discount to encourage return
                        </li>
                    </ul>
                    
                    <div style="border-top: 2px dashed #00ffff; margin: 15px 0;"></div>
                    
                    <h4 style="font-family: 'VT323', monospace; font-size: 1.4rem; color: #00ffff;">
                        MISSION OBJECTIVE:
                    </h4>
                    <p style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff;">
                        Reactivate and bring back to active player status. Time to use your revival items!
                    </p>
                </div>
                """
                st.markdown(strategy_html, unsafe_allow_html=True)
                
            elif "CASUAL PLAYERS" in segment_profiles[segment]['name']:
                strategy_html = """
                <div style="background-color: #1a1a1a; border: 3px solid #00ffff; padding: 20px; margin: 10px 0;">
                    <h3 style="font-family: 'VT323', monospace; font-size: 1.8rem; color: #00ffff; border-bottom: 2px solid #ffcc00; padding-bottom: 5px;">
                        🔍 CASUAL PLAYER STRATEGY 🔍
                    </h3>
                    
                    <h4 style="font-family: 'VT323', monospace; font-size: 1.4rem; color: #ff00ff; margin-top: 15px;">
                        RECOMMENDED POWER-UPS:
                    </h4>
                    <ul style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff; list-style-type: none; padding-left: 0;">
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #00ffff;">➤</span>
                            <strong style="color: #ff00ff;">VALUE PROPOSITION:</strong> Communicate specific benefits relevant to them
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #00ffff;">➤</span>
                            <strong style="color: #ff00ff;">TUTORIAL MODE:</strong> Help them understand product value better
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #00ffff;">➤</span>
                            <strong style="color: #ff00ff;">SPECIAL ITEMS:</strong> Occasional high-value offers to stimulate engagement
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #00ffff;">➤</span>
                            <strong style="color: #ff00ff;">TARGET EVALUATION:</strong> Consider if they align with your core audience
                        </li>
                    </ul>
                    
                    <div style="border-top: 2px dashed #ffcc00; margin: 15px 0;"></div>
                    
                    <h4 style="font-family: 'VT323', monospace; font-size: 1.4rem; color: #ff00ff;">
                        MISSION OBJECTIVE:
                    </h4>
                    <p style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff;">
                        Increase engagement or determine if resources should focus elsewhere. These players need special attention!
                    </p>
                </div>
                """
                st.markdown(strategy_html, unsafe_allow_html=True)
            
            else:
                # Generic recommendations based on RFM values
                strategy_html = f"""
                <div style="background-color: #1a1a1a; border: 3px solid {RETRO_COLORS[segment % len(RETRO_COLORS)]}; padding: 20px; margin: 10px 0;">
                    <h3 style="font-family: 'VT323', monospace; font-size: 1.8rem; color: {RETRO_COLORS[segment % len(RETRO_COLORS)]}; border-bottom: 2px solid #33ff00; padding-bottom: 5px;">
                        {segment_profiles[segment]['character']} {segment_profiles[segment]['name']} STRATEGY {segment_profiles[segment]['character']}
                    </h3>
                    
                    <h4 style="font-family: 'VT323', monospace; font-size: 1.4rem; color: #ffcc00; margin-top: 15px;">
                        RECOMMENDED POWER-UPS:
                    </h4>
                    <ul style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff; list-style-type: none; padding-left: 0;">
                """
                
                # Add recency strategy
                if recency < 30:  # Recent customers
                    strategy_html += """
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #33ff00;">➤</span>
                            <strong style="color: #00ffff;">ENGAGEMENT BOOST:</strong> Build on recent player activity
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #33ff00;">➤</span>
                            <strong style="color: #00ffff;">FOLLOW-UP QUEST:</strong> Offers related to recent interactions
                        </li>
                    """
                elif recency > 180:  # Not recent
                    strategy_html += """
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ffcc00;">➤</span>
                            <strong style="color: #ff00ff;">REACTIVATION SPELL:</strong> Powerful campaign to bring players back
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ffcc00;">➤</span>
                            <strong style="color: #ff00ff;">WIN-BACK POTION:</strong> Special discount for returning players
                        </li>
                    """
                
                # Add frequency strategy    
                if frequency > 10:  # High frequency
                    strategy_html += """
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #00ffff;">➤</span>
                            <strong style="color: #33ff00;">LOYALTY SHIELD:</strong> Reward player dedication
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #00ffff;">➤</span>
                            <strong style="color: #33ff00;">EXCLUSIVE ACCESS:</strong> Early access to new game content
                        </li>
                    """
                elif frequency < 3:  # Low frequency
                    strategy_html += """
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ff00ff;">➤</span>
                            <strong style="color: #ffcc00;">FREQUENCY BOOSTER:</strong> Increase play frequency power-up
                        </li>
                        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
                            <span style="position: absolute; left: 0; top: 0; color: #ff00ff;">➤</span>
                            <strong style="color: #ffcc00;">LIMITED-TIME EVENT:</strong> Create urgency to drive engagement
                        </li>
                    """
                
                # Close the HTML
                strategy_html += """
                    </ul>
                    
                    <div style="border-top: 2px dashed #33ff00; margin: 15px 0;"></div>
                    
                    <h4 style="font-family: 'VT323', monospace; font-size: 1.4rem; color: #ffcc00;">
                        MISSION OBJECTIVE:
                    </h4>
                    <p style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff;">
                        Custom strategy required for this unique player class. Analyze their behavior patterns further to optimize approach.
                    </p>
                </div>
                """
                st.markdown(strategy_html, unsafe_allow_html=True)
    
    pixel_divider()
    
    # Download options with retro gaming styling
    game_header("SAVE YOUR GAME DATA", level=2)
    
    # Function to create download link with retro styling
    def get_retro_download_link(df, filename="data.csv"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="game-button" style="text-decoration: none; display: inline-block;">[DOWNLOAD {filename}]</a>'
        return href
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(get_retro_download_link(rfm_segmented, "player_segments.csv"), unsafe_allow_html=True)
        st.caption("DOWNLOAD PLAYER DATA WITH SEGMENT ASSIGNMENTS")
    
    with col2:
        st.markdown(get_retro_download_link(segment_summary, "segment_summary.csv"), unsafe_allow_html=True)
        st.caption("DOWNLOAD SEGMENT SUMMARY STATISTICS")
    
    # Footer with retro gaming style
    pixel_divider()
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; font-family: 'VT323', monospace;">
        <p style="font-size: 20px; color: #33ff00;">
            CUSTOMER SEGMENT EXPLORER 8-BIT EDITION © 2025
        </p>
        <p style="font-size: 16px; color: #ff00ff;">
            THANK YOU FOR PLAYING!
        </p>
        <pre style="font-family: 'VT323', monospace; font-size: 14px; color: #00ffff; line-height: 1.2;">
         ____    _    __  __  _____     _____   _   _  ____    ____ 
        / ___|  / \  |  \/  || ____|   | ____| | \ | ||  _ \  / ___|
       | |  _  / _ \ | |\/| ||  _|     |  _|   |  \| || | | || |    
       | |_| |/ ___ \| |  | || |___    | |___  | |\  || |_| || |___ 
        \____/_/   \_\_|  |_||_____|   |_____| |_| \_||____/  \____|
        </pre>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display a retro gaming style message if no data is loaded yet
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px; border: 4px solid #33ff00; background-color: #1a1a1a; box-shadow: 10px 10px 0px #ff00ff;">
        <h2 style="font-family: 'VT323', monospace; font-size: 3rem; color: #33ff00; text-shadow: 3px 3px #ff00ff;">
            INSERT COIN TO START
        </h2>
        <div style="font-family: 'VT323', monospace; font-size: 1.5rem; color: #ffffff; margin: 30px 0;">
            👈 SELECT "USE SAMPLE DATA" OR UPLOAD CSV FILE IN THE SIDE MENU
        </div>
        <pre style="font-family: 'VT323', monospace; font-size: 14px; color: #00ffff; line-height: 1.2;">
          _____   ____   _____ ____  ____     ____    _    __  __  _____ 
         | ____| / ___| |  ___|  _ \|  _ \   / ___|  / \  |  \/  || ____|
         |  _|   \___ \ | |_  | |_) | |_) | | |  _  / _ \ | |\/| ||  _|  
         | |___   ___) ||  _| |  _ <|  _ <  | |_| |/ ___ \| |  | || |___ 
         |_____| |____/ |_|   |_| \_\_| \_\  \____/_/   \_\_|  |_||_____|
        </pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Game-like info box for expected data format
    game_info_box("""
    <h3 style="font-family: 'VT323', monospace; font-size: 1.8rem; color: #ffcc00; text-align: center;">
        DATA REQUIREMENTS
    </h3>
    
    <p style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff; margin: 15px 0;">
        Your CSV file should contain the following columns:
    </p>
    
    <ul style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff; list-style-type: none; padding-left: 10px;">
        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
            <span style="position: absolute; left: 0; top: 0; color: #33ff00;">➤</span>
            <strong style="color: #00ffff;">CustomerID:</strong> Unique identifier for each customer
        </li>
        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
            <span style="position: absolute; left: 0; top: 0; color: #33ff00;">➤</span>
            <strong style="color: #00ffff;">TransactionID:</strong> Unique identifier for each transaction
        </li>
        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
            <span style="position: absolute; left: 0; top: 0; color: #33ff00;">➤</span>
            <strong style="color: #00ffff;">TransactionDate:</strong> Date when the transaction occurred (YYYY-MM-DD)
        </li>
        <li style="margin: 10px 0; padding-left: 25px; position: relative;">
            <span style="position: absolute; left: 0; top: 0; color: #33ff00;">➤</span>
            <strong style="color: #00ffff;">AmountSpent:</strong> The monetary value of the transaction
        </li>
    </ul>
    
    <p style="font-family: 'Space Mono', monospace; font-size: 1rem; color: #ffffff; margin-top: 20px;">
        If your columns have different names, you can select the appropriate columns in the sidebar after loading your data.
    </p>
    """)