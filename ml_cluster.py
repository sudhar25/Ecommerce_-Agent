import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def get_db_connection():
    return sqlite3.connect('olist.db')

def generate_rfm_cluster(n_clusters=4):
    """Calculates RFM metrics, applies K-Means clustering, and generates a scatter plot."""
    sns.set_theme(style="darkgrid")
    
    # 1. Extract raw data for RFM calculation
    query = """
    SELECT 
        o.customer_id,
        MAX(o.order_purchase_timestamp) as last_purchase_date,
        COUNT(DISTINCT o.order_id) as frequency,
        SUM(p.payment_value) as monetary
    FROM orders o
    JOIN order_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.customer_id
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)
        
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    max_date = df['last_purchase_date'].max()
    df['recency'] = (max_date - df['last_purchase_date']).dt.days
    
    
    rfm_data = df[['recency', 'frequency', 'monetary']].fillna(0)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    df['cluster_name'] = 'segment' + df['cluster'].astype(str)
    
    
    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    sns.scatterplot(
        data=df,
        x='recency',
        y='monetary',
        hue='cluster_name',
        palette='rocket_r',
        alpha=0.8,
        s=100,
        edgecolor='none',
        ax=ax
    )
    
    ax.set_title(f"Customer Behavioral Segmentation ({n_clusters} Clusters)", fontsize=16, pad=15, color='white', fontweight='bold')
    ax.set_xlabel("Recency (days since last purchase) - lower is better", color='white', fontsize=12)
    ax.set_ylabel("Monetary value (total spend) - higher is better", color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='#e2e8f0')
    
    # Customizing legend for dark mode
    legend = ax.legend(frameon=True, facecolor='#1a1c23', edgecolor='none')
    for text in legend.get_texts():
        text.set_color("white")
    
    chart_path = "rfm_cluster.png"
    plt.savefig(chart_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    
    return chart_path