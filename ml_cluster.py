import sqlite3
import pandas as Pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def get_db_connection():
    return sqlite3.connect('olist.db')

def generate_rfm_cluster(n_clusters=4):
    """Calculates RFM metrics, applies K-Means clustering, and generates a scatter plot."""
    
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
    df['recency'] = (max_date - df['lat_purchase_date']).dt.days
    
    
    rfm_data = df[['recency', 'frequency', 'monetary']].fillna(0)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    
    kmeans = KMeans.fit_predict(rfm_scaled)
    df['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    df['cluster_name'] = 'segment' + df['cluster'].astype(str)
    
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        data=df,
        x='recency',
        y='monetary',
        hue='cluster_name',
        palette='viridis',
        alpha=0.6,
        edgecolor=None
    )
    
    plt.title(f"customer behavioral segmentation({n_cluster}Clusters)", fontsze=14, pad=15)
    plt.xlabel("Recxency(days since last purchase) - lower is better")
    plt.ylabel("Montary value(total spend) - higher is better")
    
    chart_path = "rfm_cluster.png"
    plt.savefig(chart_path, bbox_inches='tight , dpi=150')
    plt.close()
    
    
    return chart_path