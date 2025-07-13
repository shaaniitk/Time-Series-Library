#!/usr/bin/env python3
"""
Script to check the earliest available data for Gold (GC=F) from yfinance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def check_gold_data_range():
    """Check the earliest available data for Gold futures"""
    
    print("GOLD DATA AVAILABILITY CHECK")
    print("=" * 50)
    
    # Gold futures symbol
    gold_symbol = "GC=F"
    
    try:
        # Create ticker object
        gold = yf.Ticker(gold_symbol)
        
        # Try to get maximum available data
        print(f"Checking data availability for {gold_symbol}...")
        
        # Get data with maximum period
        hist_max = gold.history(period="max")
        
        if not hist_max.empty:
            earliest_date = hist_max.index.min()
            latest_date = hist_max.index.max()
            total_days = len(hist_max)
            
            print(f"\n✅ MAXIMUM AVAILABLE DATA:")
            print(f"   Earliest Date: {earliest_date.strftime('%Y-%m-%d')}")
            print(f"   Latest Date: {latest_date.strftime('%Y-%m-%d')}")
            print(f"   Total Days: {total_days:,}")
            print(f"   Data Span: {(latest_date - earliest_date).days:,} calendar days")
            
            # Show first few records
            print(f"\n📊 EARLIEST RECORDS:")
            print(hist_max.head(5).round(2))
            
            # Show latest few records
            print(f"\n📊 LATEST RECORDS:")
            print(hist_max.tail(5).round(2))
            
            # Check data quality
            print(f"\n📈 DATA QUALITY:")
            print(f"   Missing values: {hist_max.isnull().sum().sum()}")
            print(f"   Price range: ${hist_max['Close'].min():.2f} - ${hist_max['Close'].max():.2f}")
            print(f"   Average volume: {hist_max['Volume'].mean():.0f}")
            
        else:
            print("❌ No historical data found!")
            
        # Try different time periods to see what's available
        periods = ["1y", "2y", "5y", "10y", "ytd", "max"]
        print(f"\n📅 DATA AVAILABILITY BY PERIOD:")
        
        for period in periods:
            try:
                data = gold.history(period=period)
                if not data.empty:
                    start_date = data.index.min()
                    end_date = data.index.max()
                    days = len(data)
                    print(f"   {period:>3}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days:,} days)")
                else:
                    print(f"   {period:>3}: No data")
            except Exception as e:
                print(f"   {period:>3}: Error - {str(e)}")
        
        # Check info about the ticker
        print(f"\n🏷️  TICKER INFO:")
        try:
            info = gold.info
            if info:
                relevant_keys = ['longName', 'currency', 'exchange', 'quoteType', 'symbol']
                for key in relevant_keys:
                    if key in info:
                        print(f"   {key}: {info[key]}")
        except Exception as e:
            print(f"   Could not retrieve ticker info: {str(e)}")
            
        # Try to get very old data by specifying start date
        print(f"\n🔍 TESTING HISTORICAL RANGE:")
        test_dates = ["1980-01-01", "1990-01-01", "2000-01-01", "2010-01-01"]
        
        for start_date in test_dates:
            try:
                old_data = gold.history(start=start_date, end="2025-01-01")
                if not old_data.empty:
                    actual_start = old_data.index.min()
                    days = len(old_data)
                    print(f"   From {start_date}: ✅ Got data from {actual_start.strftime('%Y-%m-%d')} ({days:,} days)")
                else:
                    print(f"   From {start_date}: ❌ No data")
            except Exception as e:
                print(f"   From {start_date}: ❌ Error - {str(e)}")
                
    except Exception as e:
        print(f"❌ Error checking gold data: {str(e)}")

if __name__ == "__main__":
    check_gold_data_range()
