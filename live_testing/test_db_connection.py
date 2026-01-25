#!/usr/bin/env python3
"""
Test database connection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import CryptoDatabase

print("Testing database connection...")
print("=" * 40)

db = CryptoDatabase(
    host='127.0.0.1',
    database='crypto',
    user='dbuser',
    password='dbuser_pw_01'
)

print(f"Host: {db.host}")
print(f"Database: {db.database}")
print(f"User: {db.user}")
print(f"Password: {'*' * len(db.password)}")
print()

if db.connect():
    print("✓ Connection successful!")
    
    # Test query
    print("\nTesting query...")
    df = db.get_latest_history(limit=5)
    
    if df is not None and len(df) > 0:
        print(f"✓ Query successful! Got {len(df)} rows")
        print("\nSample data:")
        print(df[['asth_symbol', 'asth_bidPrice', 'asth_askPrice', 'changed_time']].head())
    else:
        print("⚠ Query returned no data (this might be normal if table is empty)")
    
    db.disconnect()
    print("\n✓ Disconnected")
else:
    print("✗ Connection failed!")
    print("\nTroubleshooting:")
    print("1. Check if MySQL is running: sudo systemctl status mysql")
    print("2. Verify credentials in database.py")
    print("3. Check MySQL user permissions:")
    print("   mysql -u root -p")
    print("   SELECT user, host FROM mysql.user WHERE user='dbuser';")
    print("   SHOW GRANTS FOR 'dbuser'@'localhost';")
