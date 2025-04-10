import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calculate_rfm(df, customer_id_col='CustomerID', transaction_date_col='TransactionDate', 
                 transaction_id_col='TransactionID', amount_col='AmountSpent', reference_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics from transaction data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction data
    customer_id_col : str
        Name of the customer ID column
    transaction_date_col : str
        Name of the transaction date column
    transaction_id_col : str
        Name of the transaction ID column
    amount_col : str
        Name of the amount/monetary value column
    reference_date : datetime.date or None
        Reference date for recency calculation. If None, max date + 1 day is used.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing RFM metrics for each customer
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
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

def create_rfm_scores(rfm, num_quantiles=5):
    """
    Create RFM scores by dividing each metric into quantiles.
    
    Parameters:
    -----------
    rfm : pandas.DataFrame
        DataFrame containing RFM metrics
    num_quantiles : int
        Number of quantiles to use for scoring (usually 5)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with the original RFM metrics and their scores
    """
    rfm_scored = rfm.copy()
    
    # Calculate quantiles for each metric
    quantiles = np.arange(0, 1.01, 1/num_quantiles)
    
    # Recency is better when lower, so we reverse the quantile labels
    r_labels = list(range(num_quantiles, 0, -1))
    f_labels = list(range(1, num_quantiles+1))
    m_labels = list(range(1, num_quantiles+1))
    
    # Create the score for each metric
    rfm_scored['R_Score'] = pd.qcut(rfm_scored['Recency'], q=quantiles, labels=r_labels, duplicates='drop')
    rfm_scored['F_Score'] = pd.qcut(rfm_scored['Frequency'], q=quantiles, labels=f_labels, duplicates='drop')
    rfm_scored['M_Score'] = pd.qcut(rfm_scored['Monetary'], q=quantiles, labels=m_labels, duplicates='drop')
    
    # Convert scores to integers
    rfm_scored['R_Score'] = rfm_scored['R_Score'].astype(int)
    rfm_scored['F_Score'] = rfm_scored['F_Score'].astype(int)
    rfm_scored['M_Score'] = rfm_scored['M_Score'].astype(int)
    
    # Calculate combined RFM score
    rfm_scored['RFM_Score'] = rfm_scored['R_Score'] * 100 + rfm_scored['F_Score'] * 10 + rfm_scored['M_Score']
    
    return rfm_scored

def apply_kmeans_clustering(rfm_data, n_clusters=4, features=None):
    """
    Apply K-means clustering to RFM data.
    
    Parameters:
    -----------
    rfm_data : pandas.DataFrame
        DataFrame containing RFM metrics
    n_clusters : int
        Number of clusters (segments) to create
    features : list or None
        List of feature columns to use for clustering. If None, uses ['Recency', 'Frequency', 'Monetary']
        
    Returns:
    --------
    tuple
        (rfm_data with Segment column added, cluster centers DataFrame, scaled features)
    """
    if features is None:
        features = ['Recency', 'Frequency', 'Monetary']
    
    # Create a copy to avoid modifying the original
    rfm_data = rfm_data.copy()
    
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

def create_segment_profiles(rfm_segmented, cluster_centers):
    """
    Create customer segment profiles based on RFM characteristics.
    
    Parameters:
    -----------
    rfm_segmented : pandas.DataFrame
        DataFrame containing RFM metrics with segment assignment
    cluster_centers : pandas.DataFrame
        DataFrame containing cluster centers
        
    Returns:
    --------
    dict
        Dictionary with segment profiles
    """
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

def get_marketing_recommendations(segment_profile):
    """
    Generate marketing recommendations for a segment.
    
    Parameters:
    -----------
    segment_profile : dict
        Dictionary containing segment profile information
        
    Returns:
    --------
    dict
        Dictionary with marketing recommendations
    """
    segment_name = segment_profile['name']
    recommendations = {
        'strategies': [],
        'channels': [],
        'messages': [],
        'offers': []
    }
    
    if "LEGENDARY PLAYERS" in segment_name:
        recommendations = {
            'strategies': [
                "VIP ACCESS", 
                "PREMIUM REWARDS", 
                "PERSONAL QUESTS", 
                "REFER-A-PLAYER"
            ],
            'channels': [
                "DIRECT MESSAGE",
                "SPECIAL EVENTS",
                "EXCLUSIVE FORUM",
                "PERSONAL OUTREACH"
            ],
            'messages': [
                "YOU'VE UNLOCKED ELITE STATUS!",
                "EXCLUSIVE REWARDS AWAIT!",
                "YOUR LEGENDARY JOURNEY CONTINUES..."
            ],
            'offers': [
                "DOUBLE XP WEEKENDS",
                "EXCLUSIVE CONTENT ACCESS",
                "PREMIUM BONUSES",
                "EARLY ACCESS TO NEW RELEASES"
            ]
        }
    elif "ACTIVE GAMERS" in segment_name:
        recommendations = {
            'strategies': [
                "ENGAGEMENT BOOST",
                "LEVEL-UP BONUS",
                "FEEDBACK QUEST",
                "SOCIAL CONNECT"
            ],
            'channels': [
                "REGULAR UPDATES",
                "SOCIAL CHALLENGES",
                "IN-GAME NOTIFICATIONS",
                "COMMUNITY EVENTS"
            ],
            'messages': [
                "KEEP YOUR STREAK ALIVE!",
                "LEVEL UP YOUR EXPERIENCE!",
                "YOUR NEXT ADVENTURE AWAITS!"
            ],
            'offers': [
                "BONUS REWARDS FOR CONTINUED PLAY",
                "SPECIAL ITEM BUNDLES",
                "REWARD FOR FEEDBACK",
                "FRIEND REFERRAL BONUSES"
            ]
        }
    elif "LOST PLAYERS" in segment_name:
        recommendations = {
            'strategies': [
                "REACTIVATION SPELL",
                "NEW CONTENT ALERT",
                "SURVEY QUEST",
                "REVIVAL POTION"
            ],
            'channels': [
                "EMAIL CAMPAIGN",
                "DIRECT MESSAGE",
                "SPECIAL OFFERS",
                "COMEBACK EVENTS"
            ],
            'messages': [
                "WE MISS YOU! COME BACK AND PLAY!",
                "NEW ADVENTURES AWAIT YOUR RETURN!",
                "SPECIAL COMEBACK BONUS JUST FOR YOU!"
            ],
            'offers': [
                "RETURNING PLAYER BONUS",
                "CATCH-UP PACKAGE",
                "LIMITED-TIME SPECIAL DISCOUNT",
                "FREE PREMIUM ITEMS"
            ]
        }
    elif "CASUAL PLAYERS" in segment_name:
        recommendations = {
            'strategies': [
                "VALUE PROPOSITION",
                "TUTORIAL MODE",
                "SPECIAL ITEMS",
                "TARGET EVALUATION"
            ],
            'channels': [
                "SIMPLE COMMUNICATIONS",
                "BASIC TUTORIALS",
                "PERIODIC REMINDERS",
                "EASY ACCESS OPTIONS"
            ],
            'messages': [
                "QUICK FUN AWAITS!",
                "EASY WAYS TO ENJOY THE GAME!",
                "DISCOVER NEW SIMPLE QUESTS!"
            ],
            'offers': [
                "BEGINNER-FRIENDLY PACKAGES",
                "LOW-COMMITMENT OPTIONS",
                "CASUAL PLAYER SPECIALS",
                "SIMPLIFIED GAME MODES"
            ]
        }
    else:
        # Generic recommendations
        recommendations = {
            'strategies': [
                "TARGETED ENGAGEMENT",
                "PERSONALIZED OFFERS",
                "REGULAR COMMUNICATIONS",
                "SEGMENT ANALYSIS"
            ],
            'channels': [
                "MIXED CHANNEL APPROACH",
                "TESTING DIFFERENT PLATFORMS",
                "SURVEYS FOR PREFERENCES"
            ],
            'messages': [
                "CUSTOM ADVENTURES AWAIT!",
                "NEW CONTENT JUST FOR YOU!",
                "JOIN THE NEXT CHAPTER!"
            ],
            'offers': [
                "PERSONALIZED RECOMMENDATIONS",
                "TARGETED PROMOTIONS",
                "TAILORED EXPERIENCE BUNDLES"
            ]
        }
    
    return recommendations