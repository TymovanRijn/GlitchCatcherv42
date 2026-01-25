#!/usr/bin/env python3
"""
Clean CSV data by removing rows with too many zero values.
Filters out rows where numeric columns are mostly zeros.
"""

import csv
import sys
from pathlib import Path
from typing import Set, List

def is_zero_value(value: str) -> bool:
    """Check if a value is effectively zero."""
    if not value or value.strip() == '':
        return True
    
    value = value.strip()
    
    # Check for various zero representations
    zero_patterns = [
        '0',
        '0.0',
        '0.00',
        '0.00000000',
        '0.0000000000000000',
        '0.00000000000000000',
    ]
    
    # Try to parse as float
    try:
        float_val = float(value)
        return abs(float_val) < 1e-10  # Very small threshold for floating point
    except (ValueError, TypeError):
        return value in zero_patterns

def count_zero_values(row: List[str], numeric_indices: Set[int]) -> int:
    """Count how many numeric columns have zero values."""
    zero_count = 0
    for idx in numeric_indices:
        if idx < len(row):
            if is_zero_value(row[idx]):
                zero_count += 1
    return zero_count

def clean_csv(
    input_file: str,
    output_file: str,
    max_zero_threshold: int = 10,
    exclude_columns: List[str] = None
):
    """
    Clean CSV by removing rows with too many zero values.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output cleaned CSV file
        max_zero_threshold: Maximum number of zero numeric values allowed (default: 10)
        exclude_columns: Column names to exclude from zero checking (e.g., IDs, timestamps)
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
    
    print(f"Cleaning CSV file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Max zero threshold: {max_zero_threshold} numeric columns")
    print(f"Excluding columns from zero check: {', '.join(exclude_columns)}")
    print()
    
    rows_read = 0
    rows_kept = 0
    rows_removed = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read header
        header = next(reader)
        writer.writerow(header)
        
        # Identify numeric column indices (exclude specified columns)
        exclude_indices = {i for i, col in enumerate(header) if col in exclude_columns}
        numeric_indices = {i for i in range(len(header)) if i not in exclude_indices}
        
        print(f"Checking {len(numeric_indices)} numeric columns for zero values")
        print(f"Columns checked: {', '.join([header[i] for i in sorted(numeric_indices)])}")
        print()
        
        # Process rows
        for row in reader:
            rows_read += 1
            
            if rows_read % 100000 == 0:
                print(f"Processed {rows_read:,} rows... (kept: {rows_kept:,}, removed: {rows_removed:,})")
            
            # Count zero values in numeric columns
            zero_count = count_zero_values(row, numeric_indices)
            
            # Keep row if it doesn't exceed the threshold
            if zero_count < max_zero_threshold:
                writer.writerow(row)
                rows_kept += 1
            else:
                rows_removed += 1
    
    print()
    print("=" * 60)
    print("Cleaning complete!")
    print(f"Total rows read: {rows_read:,}")
    print(f"Rows kept: {rows_kept:,} ({rows_kept/rows_read*100:.1f}%)")
    print(f"Rows removed: {rows_removed:,} ({rows_removed/rows_read*100:.1f}%)")
    print(f"Cleaned file saved to: {output_file}")
    print("=" * 60)

if __name__ == '__main__':
    # Default paths
    input_file = '../assets_history.csv'
    output_file = '../assets_history_cleaned.csv'
    max_zeros = 10
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        max_zeros = int(sys.argv[3])
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    input_path = (script_dir / input_file).resolve()
    output_path = (script_dir / output_file).resolve()
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        clean_csv(str(input_path), str(output_path), max_zeros)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
