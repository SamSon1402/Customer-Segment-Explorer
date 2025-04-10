import pandas as pd
import numpy as np
import datetime
import csv
import os

def generate_sample_data(num_customers=500, num_transactions=3000, output_file=None):
    """
    Generate sample retail transaction data with a gaming theme.
    
    Parameters:
    -----------
    num_customers : int
        Number of unique customers (players)
    num_transactions : int
        Total number of transactions (game sessions)
    output_file : str or None
        Output CSV filename. If None, doesn't save to file.
    
    Returns:
    --------
    DataFrame containing the generated data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create customer IDs
    customer_ids = [f"C{i:05d}" for i in range(1, num_customers + 1)]
    
    # Define customer segments (for more realistic data)
    # Loyal high-value customers (Legendary Players)
    loyal_customers = customer_ids[:int(num_customers * 0.2)]
    # Regular customers (Active Gamers)
    regular_customers = customer_ids[int(num_customers * 0.2):int(num_customers * 0.6)]
    # Occasional customers (Casual Players)
    occasional_customers = customer_ids[int(num_customers * 0.6):int(num_customers * 0.9)]
    # One-time customers (Lost Players)
    one_time_customers = customer_ids[int(num_customers * 0.9):]
    
    # Generate transaction data
    today = datetime.datetime.now().date()
    one_year_ago = today - datetime.timedelta(days=365)
    
    transactions = []
    transaction_id = 10000
    
    # Helper function to generate random date with tendency
    def generate_date(customer_type):
        if customer_type == 'loyal':
            # More recent purchases, more frequent
            days_ago = np.random.beta(1.2, 6.0) * 365
        elif customer_type == 'regular':
            # Somewhat recent, somewhat frequent
            days_ago = np.random.beta(1.5, 2.0) * 365
        elif customer_type == 'occasional':
            # Less recent, less frequent
            days_ago = np.random.beta(2.0, 1.2) * 365
        else:  # one-time
            # Random timing
            days_ago = np.random.uniform(0, 365)
            
        return today - datetime.timedelta(days=int(days_ago))
    
    # Helper function to generate amount
    def generate_amount(customer_type):
        if customer_type == 'loyal':
            # Higher spending
            return np.random.gamma(5.0, 20.0)
        elif customer_type == 'regular':
            # Medium spending
            return np.random.gamma(3.0, 15.0)
        elif customer_type == 'occasional':
            # Lower spending
            return np.random.gamma(2.0, 10.0)
        else:  # one-time
            # Variable spending
            return np.random.gamma(1.5, 20.0)
    
    # Generate transactions for loyal customers (more transactions)
    for customer_id in loyal_customers:
        num_trans = max(1, int(np.random.normal(12, 3)))
        for _ in range(num_trans):
            transaction_date = generate_date('loyal')
            amount = round(generate_amount('loyal'), 2)
            transaction_id += 1
            
            transactions.append({
                'CustomerID': customer_id,
                'TransactionDate': transaction_date,
                'TransactionID': f"T{transaction_id}",
                'AmountSpent': amount
            })
    
    # Generate transactions for regular customers
    for customer_id in regular_customers:
        num_trans = max(1, int(np.random.normal(6, 2)))
        for _ in range(num_trans):
            transaction_date = generate_date('regular')
            amount = round(generate_amount('regular'), 2)
            transaction_id += 1
            
            transactions.append({
                'CustomerID': customer_id,
                'TransactionDate': transaction_date,
                'TransactionID': f"T{transaction_id}",
                'AmountSpent': amount
            })
    
    # Generate transactions for occasional customers
    for customer_id in occasional_customers:
        num_trans = max(1, int(np.random.normal(3, 1)))
        for _ in range(num_trans):
            transaction_date = generate_date('occasional')
            amount = round(generate_amount('occasional'), 2)
            transaction_id += 1
            
            transactions.append({
                'CustomerID': customer_id,
                'TransactionDate': transaction_date,
                'TransactionID': f"T{transaction_id}",
                'AmountSpent': amount
            })
    
    # Generate transactions for one-time customers
    for customer_id in one_time_customers:
        transaction_date = generate_date('one-time')
        amount = round(generate_amount('one-time'), 2)
        transaction_id += 1
        
        transactions.append({
            'CustomerID': customer_id,
            'TransactionDate': transaction_date,
            'TransactionID': f"T{transaction_id}",
            'AmountSpent': amount
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Sort by transaction date
    df = df.sort_values('TransactionDate')
    
    # Reset the transaction IDs to be sequential by date
    df = df.reset_index(drop=True)
    df['TransactionID'] = [f"T{i+10001}" for i in range(len(df))]
    
    # Ensure all data types are correct
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['AmountSpent'] = df['AmountSpent'].astype(float)
    
    # Save to CSV
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    
    return df

def generate_gaming_themed_data(num_players=500, num_sessions=3000, output_file=None):
    """
    Generate sample gaming transaction data with gaming terminology.
    
    Parameters:
    -----------
    num_players : int
        Number of unique players
    num_sessions : int
        Total number of game sessions
    output_file : str or None
        Output CSV filename. If None, doesn't save to file.
    
    Returns:
    --------
    DataFrame containing the generated data
    """
    # Create standard retail data
    df = generate_sample_data(num_players, num_sessions, None)
    
    # Rename columns to gaming terminology
    df = df.rename(columns={
        'CustomerID': 'PlayerID',
        'TransactionDate': 'PlayDate',
        'TransactionID': 'SessionID',
        'AmountSpent': 'CoinsSpent'
    })
    
    # Save to CSV if output file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Gaming data saved to {output_file}")
        
    return df

if __name__ == "__main__":
    # Generate sample data when run directly
    print("Generating sample retail data...")
    retail_data = generate_sample_data(num_customers=500, num_transactions=3000, 
                                    output_file="retail_transactions.csv")
    
    print("Generating gaming-themed data...")
    gaming_data = generate_gaming_themed_data(num_players=500, num_sessions=3000,
                                           output_file="gaming_sessions.csv")
    
    print("Done!")