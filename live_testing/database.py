"""
Database connection for live testing
Connects to MySQL database on Raspberry Pi
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime, timedelta
import time

class CryptoDatabase:
    """Database connection for crypto trading data."""
    
    def __init__(self, host='127.0.0.1', database='crypto', 
                 user='dbuser', password='dbuser_pw_01'):
        """Initialize database connection."""
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.last_timestamp = None
        
    def connect(self):
        """Connect to database."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                autocommit=True,
                connection_timeout=10
            )
            if self.connection.is_connected():
                return True
        except Error as e:
            print(f"Database connection error: {e}")
            return False
        return False
    
    def disconnect(self):
        """Disconnect from database."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
    
    def get_latest_history(self, limit=1000, since_timestamp=None, include_current=False):
        """
        Get latest assets_history records.
        
        Args:
            limit: Maximum number of records to return
            since_timestamp: Only get records after this timestamp
            include_current: Also include current snapshot from assets table
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
        
        try:
            query = """
                SELECT 
                    asth_id, asth_symbol, asth_market, asth_symbolValue,
                    asth_changedPercentage, asth_symbolValue_USD, asth_symbolValue_BTC,
                    asth_available, asth_inOrder, asth_actualValue,
                    asth_bidPrice, asth_askPrice, asth_bidSize, asth_askSize,
                    asth_spread, asth_pricePrecision, asth_ticketCount,
                    asth_lastPriceWeight, asth_lastPriceCount, asth_hide,
                    changed_by, changed_cnt, changed_time
                FROM assets_history
                WHERE asth_bidPrice > 0 AND asth_askPrice > 0
            """
            
            if since_timestamp:
                # Format timestamp properly for MySQL
                if isinstance(since_timestamp, str):
                    query += f" AND changed_time > '{since_timestamp}'"
                else:
                    query += f" AND changed_time > '{since_timestamp.strftime('%Y-%m-%d %H:%M:%S')}'"
            
            query += " ORDER BY changed_time ASC LIMIT %s"  # ASC to get oldest first
            
            # Use cursor to avoid pandas warning
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            
            if not rows:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rows)
            
            # Convert numeric columns to float (MySQL Decimal -> float)
            numeric_cols = ['asth_bidPrice', 'asth_askPrice', 'asth_bidSize', 'asth_askSize', 
                            'asth_ticketCount', 'asth_lastPriceWeight', 'asth_spread',
                            'asth_symbolValue', 'asth_changedPercentage', 'asth_symbolValue_USD',
                            'asth_symbolValue_BTC', 'asth_available', 'asth_inOrder', 'asth_actualValue',
                            'asth_pricePrecision', 'asth_lastPriceCount']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
            # Sort by time ascending
            if 'changed_time' in df.columns:
                df['changed_time'] = pd.to_datetime(df['changed_time'], errors='coerce')
                df = df.sort_values('changed_time').reset_index(drop=True)
            
            return df
            
        except Error as e:
            print(f"Query error: {e}")
            return None
    
    def get_current_assets(self):
        """Get current assets snapshot."""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
        
        try:
            query = """
                SELECT 
                    ast_id, ast_symbol, ast_market, ast_symbolValue,
                    ast_changedPercentage, ast_symbolValue_USD, ast_symbolValue_BTC,
                    ast_available, ast_inOrder, ast_actualValue,
                    ast_bidPrice, ast_askPrice, ast_bidSize, ast_askSize,
                    ast_spread, ast_pricePrecision, ast_ticketCount,
                    ast_lastPriceWeight, ast_lastPriceCount, ast_hide,
                    changed_by, changed_cnt, changed_time
                FROM assets
                WHERE ast_bidPrice > 0 AND ast_askPrice > 0
                ORDER BY changed_time DESC
            """
            
            # Use cursor to avoid pandas warning
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows)
            
            # Rename to match history format (ast_* -> asth_*)
            rename_map = {
                'ast_bidPrice': 'asth_bidPrice',
                'ast_askPrice': 'asth_askPrice',
                'ast_bidSize': 'asth_bidSize',
                'ast_askSize': 'asth_askSize',
                'ast_ticketCount': 'asth_ticketCount',
                'ast_lastPriceWeight': 'asth_lastPriceWeight',
                'ast_spread': 'asth_spread',
                'ast_symbol': 'asth_symbol'
            }
            df = df.rename(columns=rename_map)
            
            # Convert numeric columns to float
            numeric_cols = ['asth_bidPrice', 'asth_askPrice', 'asth_bidSize', 'asth_askSize', 
                            'asth_ticketCount', 'asth_lastPriceWeight', 'asth_spread',
                            'ast_symbolValue', 'ast_changedPercentage', 'ast_symbolValue_USD',
                            'ast_symbolValue_BTC', 'ast_available', 'ast_inOrder', 'ast_actualValue',
                            'ast_pricePrecision', 'ast_lastPriceCount']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
            if 'changed_time' in df.columns:
                df['changed_time'] = pd.to_datetime(df['changed_time'], errors='coerce')
            
            return df
            
        except Error as e:
            print(f"Query error: {e}")
            return None
