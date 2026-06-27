import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_db_connection():
    return sqlite3.connect('olist.db')

def plot_monthly_sales(year=2017):
    """Generates a line chat showing montly sales for a specific year"""
    sns.set_theme(style="darkgrid", palette="rocket")
    query = f"""
    SELECT
    strftime('%m', o.order_purchase_timestamp) as month,
    SUM(p.payment_value) as total_sales
    From orders o
    JOIN order_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    AND strftime('%Y', o.order_purchase_timestamp) = '{year}'
    GROUP BY month
    ORDER BY month;
    """
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)
        
    month_map = {'01':'Jan', '02':'Feb', '03':'Mar', '04':'Apr', '05':'May', '06':'Jun', 
                 '07':'Jul', '08':'Aug', '09':'Sep', '10':'Oct', '11':'Nov', '12':'Dec'}
    df['month_name'] = df['month'].map(month_map)
    
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    sns.lineplot(data=df, x='month_name', y='total_sales', marker='o', linewidth=3, color='#FF416C', ax=ax)
    
    ax.set_title(f"E-commerce Total Revenue Trend ({year})", fontsize=16, pad=15, color='white', fontweight='bold')
    ax.set_xlabel("Month", color='white', fontsize=12)
    ax.set_ylabel('Revenue (R$)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='#e2e8f0')
    
    chart_path ="monthly_sales.png"
    plt.savefig(chart_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return chart_path

def plot_hourly_peaks():
    """Generates a bar chart showing what hours of the day customers buy the most"""
    sns.set_theme(style="darkgrid")
    query = """
    SELECT 
        strftime('%H', order_purchase_timestamp) as hour,
        COUNT(order_id) as total_orders
    FROM orders
    WHERE order_status = 'delivered'
    GROUP BY hour
    ORDER BY hour;
    """
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    sns.barplot(data=df, x='hour', y='total_orders', color='#667eea', ax=ax, width=0.6)
    
    ax.set_title("Hourly Purchase Distribution (Peak Shopping Hours)", fontsize=16, pad=15, color='white', fontweight='bold')
    ax.set_xlabel("Hour of the Day (24h Format)", color='white', fontsize=12)
    ax.set_ylabel("Volume of Placed Orders", color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='#e2e8f0', axis='y')
    
    chart_path = "hourly_peaks.png"
    plt.savefig(chart_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return chart_path
        