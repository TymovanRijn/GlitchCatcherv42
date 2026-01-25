#!/usr/bin/env python3
"""
Extract data from SQL INSERT statements and convert to CSV.
Handles large files efficiently by processing line by line.
"""

import re
import csv
import sys
from typing import List, Optional

def parse_sql_value(value: str) -> str:
    """Parse a single SQL value, handling NULL, strings, numbers, etc."""
    value = value.strip()
    
    # Handle NULL
    if value.upper() == 'NULL':
        return ''
    
    # Handle strings (remove quotes and unescape)
    if value.startswith("'") and value.endswith("'"):
        # Remove outer quotes and handle escaped quotes
        value = value[1:-1].replace("''", "'").replace("\\'", "'")
        return value
    
    # Handle numbers and other values (return as-is)
    return value

def parse_insert_row(row_str: str) -> Optional[List[str]]:
    """Parse a single row from INSERT statement like (val1, val2, ...)"""
    # Remove leading/trailing whitespace and parentheses
    row_str = row_str.strip().rstrip(',').strip()
    
    if not row_str.startswith('(') or not row_str.endswith(')'):
        return None
    
    # Remove outer parentheses
    row_str = row_str[1:-1]
    
    # Split by comma, but handle commas inside quoted strings
    values = []
    current_value = ''
    in_quotes = False
    quote_char = None
    
    i = 0
    while i < len(row_str):
        char = row_str[i]
        
        if char in ("'", '"') and (i == 0 or row_str[i-1] != '\\'):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                # Check if it's an escaped quote (two quotes in a row)
                if i + 1 < len(row_str) and row_str[i+1] == quote_char:
                    current_value += char
                    i += 1  # Skip the next quote
                else:
                    in_quotes = False
                    quote_char = None
            else:
                current_value += char
        elif char == ',' and not in_quotes:
            values.append(parse_sql_value(current_value))
            current_value = ''
        else:
            current_value += char
        
        i += 1
    
    # Add the last value
    if current_value or len(values) > 0:
        values.append(parse_sql_value(current_value))
    
    return values if values else None

def extract_sql_to_csv(sql_file: str, csv_file: str):
    """Extract data from SQL INSERT statements and write to CSV."""
    
    # Column names from the INSERT statement
    columns = [
        'asth_id', 'asth_symbol', 'asth_market', 'asth_symbolValue',
        'asth_changedPercentage', 'asth_symbolValue_USD', 'asth_symbolValue_BTC',
        'asth_available', 'asth_inOrder', 'asth_actualValue', 'asth_bidPrice',
        'asth_askPrice', 'asth_bidSize', 'asth_askSize', 'asth_spread',
        'asth_pricePrecision', 'asth_ticketCount', 'asth_lastPriceWeight',
        'asth_lastPriceCount', 'asth_hide', 'changed_by', 'changed_cnt', 'changed_time'
    ]
    
    row_count = 0
    current_row_buffer = ''
    in_insert = False
    
    print(f"Processing {sql_file}...")
    print(f"Output will be written to {csv_file}")
    
    with open(sql_file, 'r', encoding='utf-8') as infile, \
         open(csv_file, 'w', encoding='utf-8', newline='') as outfile:
        
        writer = csv.writer(outfile)
        writer.writerow(columns)  # Write header
        
        for line_num, line in enumerate(infile, 1):
            # Check if this is an INSERT statement line
            if 'INSERT INTO' in line.upper() and 'VALUES' in line.upper():
                in_insert = True
                continue
            
            if not in_insert:
                continue
            
            # Skip comment lines and empty lines
            line = line.strip()
            if not line or line.startswith('--') or line.startswith('/*'):
                continue
            
            # Check if we've reached the end of INSERT statements
            if line.upper().startswith('ALTER TABLE') or \
               line.upper().startswith('COMMIT') or \
               line.upper().startswith('SET '):
                break
            
            # Accumulate the line (INSERT rows can span multiple lines)
            current_row_buffer += line
            
            # Check if this line ends a row (ends with ), or ),)
            if current_row_buffer.rstrip().endswith('),') or \
               (current_row_buffer.rstrip().endswith(')') and 
                not current_row_buffer.rstrip().endswith('),')):
                
                # Remove trailing comma if present
                row_str = current_row_buffer.rstrip().rstrip(',')
                
                # Parse the row
                row_data = parse_insert_row(row_str)
                
                if row_data and len(row_data) == len(columns):
                    writer.writerow(row_data)
                    row_count += 1
                    
                    if row_count % 100000 == 0:
                        print(f"Processed {row_count:,} rows...")
                
                # Reset buffer
                current_row_buffer = ''
    
    print(f"\nConversion complete!")
    print(f"Total rows extracted: {row_count:,}")
    print(f"CSV file saved to: {csv_file}")

if __name__ == '__main__':
    sql_file = 'assets_history.sql'
    csv_file = 'assets_history.csv'
    
    if len(sys.argv) > 1:
        sql_file = sys.argv[1]
    if len(sys.argv) > 2:
        csv_file = sys.argv[2]
    
    try:
        extract_sql_to_csv(sql_file, csv_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
