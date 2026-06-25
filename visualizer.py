import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_db_connection():
    return sqlite3.connect('olist.db')

def plot_monthly_sales(year=2017):
    """Generates a line chat showing montly sales for a specific year"""
    query = f"""
    SELECT
    strftime('%m'), o.order_purchase_timestamp) as month,
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
    
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='month_name', y='total_sales', marker='o', linewidth=2.5)
    plt.title(f"E-commerce total revenue trend ({year})", fontsize=14, pad=15)
    plt.xlabel("month")
    plt.ylabel('Revenue (R$)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    
    chart_path ="monthly_sales.png"
    plt.savefig(chart_path, bbox_inches='tight', dpi=150)
    plt.close()
    return chart_path

def plot_hourly_peaks():
    """Generates a bar chart showing what hours of the day customers buy the most"""
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
    
    plt.figure(figsize=(12, 5))
    sns.barplot(data=df, x='hour', y='total_orders')
    plt.title("Hourly Purchase Distribution (Peak Shopping Hours)", fontsize=14, pad=15)
    plt.xlabel("Hour of the Day (24h Format)")
    plt.ylabel("Volume of Placed Orders")
    
    chart_path = "hourly_peaks.png"
    plt.savefig(chart_path, bbox_inches='tight', dpi=150)
    plt.close()
    return chart_path
        