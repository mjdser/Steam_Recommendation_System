"""
Basic Inspection Script for Steam API Collected Data
Run this after collecting data to understand what you have
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set display options for better viewing - NO TRUNCATION
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 20)       # Show up to 20 rows
pd.set_option('display.width', 1000)        # Wide display
pd.set_option('display.max_colwidth', None) # Show full text in columns
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def inspect_api_data(file_path):
    """
    Comprehensive inspection of collected API data
    """
    
    print("=" * 120)
    print("STEAM API DATA INSPECTION")
    print("=" * 120)
    print(f"File: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns")
    print()
    
    # 1. Basic Overview
    print("-" * 120)
    print("1. BASIC OVERVIEW")
    print("-" * 120)
    print("\nData types:")
    print(df.dtypes.value_counts())
    print()
    
    print("\nFULL FIRST 5 ROWS (ALL COLUMNS):")
    print("=" * 120)
    # Convert to string to ensure full display
    for idx, row in df.head(5).iterrows():
        print(f"\nRow {idx}:")
        for col in df.columns:
            value = row[col]
            # Truncate very long strings for readability but show full if short
            if isinstance(value, str) and len(value) > 200:
                print(f"  {col}: {value[:200]}... (truncated, {len(value)} chars)")
            else:
                print(f"  {col}: {value}")
        print("-" * 80)
    print()
    
    # Show using pandas display (non-truncated)
    print("\nUsing pandas display (all columns):")
    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', None):
        print(df.head())
    print()
    
    print("\nLast 5 rows:")
    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', None):
        print(df.tail())
    print()
    
    # 2. Missing Values
    print("-" * 120)
    print("2. MISSING VALUES")
    print("-" * 120)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    }).sort_values('Missing', ascending=False)
    missing_df = missing_df[missing_df['Missing'] > 0]
    
    if len(missing_df) > 0:
        print("\nColumns with missing values:")
        print(missing_df)
    else:
        print("\nNo missing values found!")
    print()
    
    # 3. Data Statistics
    print("-" * 120)
    print("3. NUMERICAL STATISTICS")
    print("-" * 120)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nNumerical columns:")
        print(df[numeric_cols].describe())
        print()
    else:
        print("No numerical columns found")
    
    # 4. Categorical Statistics
    print("-" * 120)
    print("4. CATEGORICAL STATISTICS")
    print("-" * 120)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols[:5]:  # Show first 5 categorical columns
            print(f"\n{col} (Top 10 values):")
            print(df[col].value_counts().head(10))
            print(f"Unique values: {df[col].nunique()}")
    else:
        print("No categorical columns found")
    
    # 5. Profile Privacy Analysis
    print("-" * 120)
    print("5. PROFILE PRIVACY ANALYSIS")
    print("-" * 120)
    
    if 'community_visibility' in df.columns:
        visibility_counts = df['community_visibility'].value_counts()
        print("\nCommunity visibility states:")
        print(visibility_counts)
        print("\n(3 = Public, 1 = Private, 2 = Friends Only)")
        
        # Calculate percentages
        print(f"\nPublic profiles: {(df['community_visibility'] == 3).mean() * 100:.1f}%")
        print(f"Private/Friends-only: {(df['community_visibility'] != 3).mean() * 100:.1f}%")
    
    # 6. Game Library Analysis
    print("-" * 120)
    print("6. GAME LIBRARY ANALYSIS")
    print("-" * 120)
    
    if 'game_count_total' in df.columns:
        print(f"\nGame count statistics:")
        print(df['game_count_total'].describe())
        
        print(f"\nUsers with more than 100 games: {len(df[df['game_count_total'] > 100])}")
        print(f"Users with more than 500 games: {len(df[df['game_count_total'] > 500])}")
        print(f"Users with more than 1000 games: {len(df[df['game_count_total'] > 1000])}")
    
    # 7. Playtime Analysis
    print("-" * 120)
    print("7. PLAYTIME ANALYSIS")
    print("-" * 120)
    
    if 'total_playtime_hours' in df.columns:
        playtime = df['total_playtime_hours']
        print("\nTotal playtime statistics (hours):")
        print(playtime.describe())
        
        print(f"\nUsers with >100 hours: {len(df[playtime > 100])}")
        print(f"Users with >500 hours: {len(df[playtime > 500])}")
        print(f"Users with >1000 hours: {len(df[playtime > 1000])}")
        print(f"Users with >5000 hours: {len(df[playtime > 5000])}")
        
        # Playtime distribution
        print(f"\nPlaytime quantiles:")
        print(playtime.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    
    # 8. Recent Activity
    print("-" * 120)
    print("8. RECENT ACTIVITY (2 WEEKS)")
    print("-" * 120)
    
    if 'total_playtime_2weeks_hours' in df.columns:
        recent = df['total_playtime_2weeks_hours']
        active_users = len(df[recent > 0])
        print(f"\nActive users (played in last 2 weeks): {active_users} ({active_users/len(df)*100:.1f}%)")
        
        if active_users > 0:
            print(f"\nRecent playtime statistics (among active users):")
            print(recent[recent > 0].describe())
    
    # 9. Country Distribution
    print("-" * 120)
    print("9. COUNTRY DISTRIBUTION")
    print("-" * 120)
    
    if 'country_code' in df.columns:
        country_counts = df['country_code'].value_counts().head(10)
        print(f"\nTop 10 countries:")
        print(country_counts)
        print(f"\nUnique countries: {df['country_code'].nunique()}")
    
    # 10. Top Games Analysis
    print("-" * 120)
    print("10. TOP GAMES ANALYSIS")
    print("-" * 120)
    
    top_game_cols = [col for col in df.columns if col.startswith('top_') and col.endswith('_name')]
    if top_game_cols:
        top_games = df[top_game_cols]
        all_top_games = pd.Series(top_games.values.flatten()).dropna()
        most_played_games = all_top_games.value_counts().head(10)
        print("\nMost common top-played games:")
        print(most_played_games)
    
    # 11. Achievement Analysis
    print("-" * 120)
    print("11. ACHIEVEMENT ANALYSIS")
    print("-" * 120)
    
    if 'total_achievements' in df.columns:
        print(f"\nAchievement statistics:")
        print(df['total_achievements'].describe())
        
        print(f"\nUsers with achievements: {len(df[df['total_achievements'] > 0])}")
        print(f"Users with >100 achievements: {len(df[df['total_achievements'] > 100])}")
    
    # 12. Data Quality Summary
    print("-" * 120)
    print("12. DATA QUALITY SUMMARY")
    print("-" * 120)
    
    print(f"\nTotal users collected: {len(df)}")
    print(f"Unique user IDs: {df['steamid'].nunique()}")
    print(f"Total unique countries: {df['country_code'].nunique() if 'country_code' in df.columns else 'N/A'}")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['steamid']).sum()
    print(f"Duplicate steamids: {duplicates} (should be 0)")
    
    # Check sample size
    if len(df) > 0:
        print(f"\nData collection date range:")
        if 'member_since' in df.columns:
            member_dates = pd.to_datetime(df['member_since'], errors='coerce')
            print(f"  Earliest member: {member_dates.min()}")
            print(f"  Latest member: {member_dates.max()}")
    
    # 13. Column Names and Types
    print("-" * 120)
    print("13. COLUMN NAMES AND TYPES")
    print("-" * 120)
    
    print("\nAll columns:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        print(f"  {i:2d}. {col:30s} {str(dtype):15s} Non-null: {non_null:6d}  Null: {null_count:6d}")
    
    return df


def visualize_api_data(df, output_dir='api_visualizations'):
    """
    Create visualizations of the API data
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 120)
    print("GENERATING VISUALIZATIONS")
    print("=" * 120)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Playtime Distribution
    if 'total_playtime_hours' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Full distribution (capped)
        playtime_capped = df['total_playtime_hours'].clip(0, 5000)
        axes[0].hist(playtime_capped, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Total Playtime (hours)')
        axes[0].set_ylabel('Number of Users')
        axes[0].set_title('Total Playtime Distribution (capped at 5000h)')
        
        # Log scale
        axes[1].hist(np.log1p(df['total_playtime_hours']), bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Log(Total Playtime + 1)')
        axes[1].set_ylabel('Number of Users')
        axes[1].set_title('Playtime Distribution (Log Scale)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/playtime_distribution.png', dpi=150)
        plt.close()
        print("✓ Playtime distribution saved")
    
    # 2. Game Count Distribution
    if 'game_count_total' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Full distribution (capped)
        game_count_capped = df['game_count_total'].clip(0, 500)
        axes[0].hist(game_count_capped, bins=50, color='green', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Number of Games Owned')
        axes[0].set_ylabel('Number of Users')
        axes[0].set_title('Game Count Distribution (capped at 500)')
        
        # Log scale
        axes[1].hist(np.log1p(df['game_count_total']), bins=50, color='purple', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Log(Games Owned + 1)')
        axes[1].set_ylabel('Number of Users')
        axes[1].set_title('Game Count Distribution (Log Scale)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/game_count_distribution.png', dpi=150)
        plt.close()
        print("✓ Game count distribution saved")
    
    # 3. Recent Activity
    if 'total_playtime_2weeks_hours' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter to active users
        recent_df = df[df['total_playtime_2weeks_hours'] > 0]
        if len(recent_df) > 0:
            recent_capped = recent_df['total_playtime_2weeks_hours'].clip(0, 100)
            ax.hist(recent_capped, bins=40, color='orange', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Playtime in Last 2 Weeks (hours)')
            ax.set_ylabel('Number of Users')
            ax.set_title(f'Recent Playtime Distribution (n={len(recent_df)} active users)')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/recent_playtime_distribution.png', dpi=150)
            plt.close()
            print("✓ Recent playtime distribution saved")
    
    # 4. Country Distribution
    if 'country_code' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        country_counts = df['country_code'].value_counts().head(15)
        bars = ax.bar(country_counts.index, country_counts.values, color='teal', edgecolor='black')
        ax.set_xlabel('Country Code')
        ax.set_ylabel('Number of Users')
        ax.set_title('Top 15 Countries')
        ax.set_xticklabels(country_counts.index, rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, country_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   str(value), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/country_distribution.png', dpi=150)
        plt.close()
        print("✓ Country distribution saved")
    
    # 5. Playtime vs Game Count
    if 'total_playtime_hours' in df.columns and 'game_count_total' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot with alpha for density
        ax.scatter(df['game_count_total'], df['total_playtime_hours'], 
                  alpha=0.1, s=10, c='blue')
        ax.set_xlabel('Number of Games Owned')
        ax.set_ylabel('Total Playtime (hours)')
        ax.set_title('Playtime vs Games Owned')
        
        # Add trend line
        z = np.polyfit(df['game_count_total'], df['total_playtime_hours'], 1)
        p = np.poly1d(z)
        ax.plot(df['game_count_total'].sort_values(), 
                p(df['game_count_total'].sort_values()), 
                "r-", linewidth=2, label='Trend line')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/playtime_vs_games.png', dpi=150)
        plt.close()
        print("✓ Playtime vs Games scatter plot saved")
    
    # 6. Correlation Heatmap
    print("\nGenerating correlation heatmap...")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        # Select key numeric columns
        key_cols = [col for col in numeric_df.columns if col in [
            'game_count_total', 'total_playtime_hours', 'avg_playtime_hours',
            'max_playtime_hours', 'games_with_achievements', 'total_achievements',
            'total_playtime_2weeks_hours', 'active_games_2weeks'
        ]]
        
        if key_cols:
            corr_matrix = numeric_df[key_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                       linewidths=0.5, ax=ax, vmin=-1, vmax=1)
            ax.set_title('Correlation Heatmap (Key Features)')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=150)
            plt.close()
            print("✓ Correlation heatmap saved")
    
    print(f"\nAll visualizations saved to: {output_dir}/")


def print_summary_report(df):
    """
    Generate a text summary report
    """
    
    print("\n" + "=" * 120)
    print("SUMMARY REPORT")
    print("=" * 120)
    
    with open('api_data_summary.txt', 'w') as f:
        f.write("STEAM API DATA COLLECTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Users Collected: {len(df)}\n")
        f.write(f"Total Features: {df.shape[1]}\n\n")
        
        # User statistics
        f.write("USER STATISTICS\n")
        f.write("-" * 30 + "\n")
        if 'game_count_total' in df.columns:
            f.write(f"Average games per user: {df['game_count_total'].mean():.1f}\n")
            f.write(f"Median games per user: {df['game_count_total'].median():.1f}\n")
            f.write(f"Max games: {df['game_count_total'].max()}\n")
        
        if 'total_playtime_hours' in df.columns:
            f.write(f"\nAverage total playtime: {df['total_playtime_hours'].mean():.1f} hours\n")
            f.write(f"Median total playtime: {df['total_playtime_hours'].median():.1f} hours\n")
            f.write(f"Max total playtime: {df['total_playtime_hours'].max():.1f} hours\n")
        
        # Privacy statistics
        if 'community_visibility' in df.columns:
            public_pct = (df['community_visibility'] == 3).mean() * 100
            f.write(f"\nPublic profiles: {public_pct:.1f}%\n")
        
        # Country statistics
        if 'country_code' in df.columns:
            f.write(f"\nUnique countries: {df['country_code'].nunique()}\n")
            f.write(f"Top 5 countries:\n")
            for country, count in df['country_code'].value_counts().head(5).items():
                f.write(f"  - {country}: {count} users\n")
        
        # Recent activity
        if 'total_playtime_2weeks_hours' in df.columns:
            active = (df['total_playtime_2weeks_hours'] > 0).sum()
            f.write(f"\nActive users (last 2 weeks): {active} ({active/len(df)*100:.1f}%)\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("END OF REPORT\n")
    
    print("✓ Summary report saved to: api_data_summary.txt")


def main():
    """
    Main inspection function
    """
    # Path to your API data
    API_DATA_PATH = "data/steam_user_api_data.csv"
    
    # Check if file exists
    if not os.path.exists(API_DATA_PATH):
        print(f"Error: File not found at {API_DATA_PATH}")
        print("Please update the path to your API data file.")
        return
    
    # Run inspection
    df = inspect_api_data(API_DATA_PATH)
    
    # Generate visualizations
    visualize_api_data(df)
    
    # Generate summary report
    print_summary_report(df)

    print("non-zero achievement users: ", len(df[df['total_achievements_unlocked'] > 0]))

    print("\n" + "=" * 120)
    print("INSPECTION COMPLETE!")
    print("=" * 120)
    print("Files generated:")
    print("  - api_data_summary.txt (text summary)")
    print("  - api_visualizations/ (visualization images)")
    print("  - Inspection output above (console)")
    print()


if __name__ == "__main__":
    main()