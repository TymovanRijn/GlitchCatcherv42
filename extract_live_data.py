#!/usr/bin/env python3
"""
Extract and clean asset history data from live database.
Pulls all data from assets_history table, applies cleaning, and exports to CSV.
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Set, List
from live_testing.database import CryptoDatabase

def is_zero_value(value) -> bool:
    """Check if a value is effectively zero."""
    if pd.isna(value):
        return True
    
    # Convert to string for comparison
    value_str = str(value).strip()
    
    if not value_str or value_str == '':
        return True
    
    # Check for various zero representations
    zero_patterns = [
        '0',
        '0.0',
        '0.00',
        '0.00000000',
        '0.0000000000000000',
        '0.00000000000000000',
    ]
    
    if value_str in zero_patterns:
        return True
    
    # Try to parse as float
    try:
        float_val = float(value)
        return abs(float_val) < 1e-10  # Very small threshold for floating point
    except (ValueError, TypeError):
        return False

def count_zero_values(row: pd.Series, numeric_columns: Set[str]) -> int:
    """Count how many numeric columns have zero values."""
    zero_count = 0
    for col in numeric_columns:
        if col in row.index:
            if is_zero_value(row[col]):
                zero_count += 1
    return zero_count

def clean_dataframe(
    df: pd.DataFrame,
    max_zero_threshold: int = 10,
    exclude_columns: List[str] = None
) -> pd.DataFrame:
    """
    Clean DataFrame by removing rows with too many zero values.
    
    Args:
        df: Input DataFrame
        max_zero_threshold: Maximum number of zero numeric values allowed (default: 10)
        exclude_columns: Column names to exclude from zero checking
    
    Returns:
        Cleaned DataFrame
    """
    if exclude_columns is None:
        exclude_columns = [
            'asth_id',
            'asth_symbol',
            'asth_market',
            'changed_by',
            'changed_time',
            'changed_cnt',
            'asth_hide',
            'asth_pricePrecision',
            'asth_ticketCount',
            'asth_lastPriceWeight',
            'asth_lastPriceCount'
        ]
    
    print(f"Cleaning DataFrame...")
    print(f"   Input rows: {len(df):,}")
    print(f"   Max zero threshold: {max_zero_threshold} numeric columns")
    print(f"   Excluding columns: {', '.join(exclude_columns)}")
    
    # Identify numeric columns (exclude specified columns)
    numeric_columns = set(df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns)
    numeric_columns = numeric_columns - set(exclude_columns)
    
    print(f"   Checking {len(numeric_columns)} numeric columns for zero values")
    print(f"   Columns checked: {', '.join(sorted(numeric_columns))}")
    print()
    
    # Count zeros for each row
    print("   Counting zero values per row...")
    zero_counts = df.apply(lambda row: count_zero_values(row, numeric_columns), axis=1)
    
    # Filter rows
    mask = zero_counts < max_zero_threshold
    cleaned_df = df[mask].copy()
    
    rows_removed = len(df) - len(cleaned_df)
    
    print()
    print("=" * 60)
    print("Cleaning complete!")
    print(f"   Total rows: {len(df):,}")
    print(f"   Rows kept: {len(cleaned_df):,} ({len(cleaned_df)/len(df)*100:.1f}%)")
    print(f"   Rows removed: {rows_removed:,} ({rows_removed/len(df)*100:.1f}%)")
    print("=" * 60)
    print()
    
    return cleaned_df

def fetch_all_history(db: CryptoDatabase, batch_size: int = 100000) -> pd.DataFrame:
    """
    Fetch all asset history records from database in batches.
    
    Args:
        db: CryptoDatabase instance
        batch_size: Number of records to fetch per batch
    
    Returns:
        DataFrame with all history records
    """
    print("=" * 60)
    print("📊 EXTRACTING DATA FROM LIVE DATABASE")
    print("=" * 60)
    print()
    
    if not db.connect():
        print("❌ Failed to connect to database!")
        return None
    
    print("✅ Connected to database")
    print()
    
    all_dataframes = []
    offset = 0
    total_fetched = 0
    
    try:
        # First, get total count
        cursor = db.connection.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM assets_history WHERE asth_bidPrice > 0 AND asth_askPrice > 0")
        result = cursor.fetchone()
        total_records = result[0] if result else 0
        cursor.close()
        
        print(f"📈 Total records in database: {total_records:,}")
        print(f"📦 Fetching in batches of {batch_size:,} records...")
        print()
        
        # Fetch data in batches
        while True:
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
                ORDER BY changed_time ASC
                LIMIT %s OFFSET %s
            """
            
            cursor = db.connection.cursor(dictionary=True)
            cursor.execute(query, (batch_size, offset))
            rows = cursor.fetchall()
            cursor.close()
            
            if not rows:
                break
            
            # Convert to DataFrame
            df_batch = pd.DataFrame(rows)
            
            # Convert numeric columns to float
            numeric_cols = ['asth_bidPrice', 'asth_askPrice', 'asth_bidSize', 'asth_askSize', 
                          'asth_ticketCount', 'asth_lastPriceWeight', 'asth_spread',
                          'asth_symbolValue', 'asth_changedPercentage', 'asth_symbolValue_USD',
                          'asth_symbolValue_BTC', 'asth_available', 'asth_inOrder', 'asth_actualValue',
                          'asth_pricePrecision', 'asth_lastPriceCount']
            
            for col in numeric_cols:
                if col in df_batch.columns:
                    df_batch[col] = pd.to_numeric(df_batch[col], errors='coerce').astype(float)
            
            # Convert timestamp
            if 'changed_time' in df_batch.columns:
                df_batch['changed_time'] = pd.to_datetime(df_batch['changed_time'], errors='coerce')
            
            all_dataframes.append(df_batch)
            total_fetched += len(df_batch)
            
            print(f"   Fetched batch: {len(df_batch):,} records (Total: {total_fetched:,}/{total_records:,})")
            
            # If we got fewer records than batch_size, we're done
            if len(df_batch) < batch_size:
                break
            
            offset += batch_size
        
        print()
        print(f"✅ Successfully fetched {total_fetched:,} records")
        
        # Combine all batches
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            print(f"✅ Combined into single DataFrame: {len(combined_df):,} rows")
            return combined_df
        else:
            print("⚠️  No data fetched!")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.disconnect()
        print("✅ Database connection closed")

def main():
    """Main function to extract, clean, and save data."""
    print("=" * 60)
    print("🚀 GLITCHCATCHER - LIVE DATA EXTRACTION")
    print("=" * 60)
    print()
    
    # Configuration
    output_csv = 'assets_history_cleaned_v2.csv'
    max_zero_threshold = 10
    batch_size = 100000
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        output_csv = sys.argv[1]
    if len(sys.argv) > 2:
        max_zero_threshold = int(sys.argv[2])
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])
    
    # Initialize database connection
    db = CryptoDatabase()
    
    # Fetch all data
    df = fetch_all_history(db, batch_size=batch_size)
    
    if df is None or len(df) == 0:
        print("❌ No data to process!")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("📊 DATA SUMMARY")
    print("=" * 60)
    print(f"   Total rows: {len(df):,}")
    print(f"   Unique symbols: {df['asth_symbol'].nunique()}")
    print(f"   Date range: {df['changed_time'].min()} to {df['changed_time'].max()}")
    print(f"   Columns: {', '.join(df.columns.tolist())}")
    print("=" * 60)
    print()
    
    # Clean the data
    print("🧹 Cleaning data...")
    print()
    cleaned_df = clean_dataframe(df, max_zero_threshold=max_zero_threshold)
    
    # Sort by symbol and time
    print("📋 Sorting data by symbol and time...")
    cleaned_df = cleaned_df.sort_values(['asth_symbol', 'changed_time']).reset_index(drop=True)
    
    # Save to CSV
    print(f"💾 Saving cleaned data to: {output_csv}")
    cleaned_df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(cleaned_df):,} rows to {output_csv}")
    print()
    
    # Final summary
    print("=" * 60)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"   Output file: {output_csv}")
    print(f"   Total rows: {len(cleaned_df):,}")
    print(f"   Unique symbols: {cleaned_df['asth_symbol'].nunique()}")
    print(f"   File size: {Path(output_csv).stat().st_size / (1024*1024):.2f} MB")
    print("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
