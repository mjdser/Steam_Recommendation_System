import os
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
import gc

# --- Configuration ---
REVIEWS_DIR = Path("data/steam_reviews")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "steam_reviews_merged.csv"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def get_sorted_review_files():
    """
    Get all review files sorted by their numeric ranges.
    Files are named like: reviews-1-115.csv, reviews-115-1230.csv, etc.
    """
    # Get all CSV files in the reviews directory
    all_files = list(REVIEWS_DIR.glob("reviews-*.csv"))
    
    if not all_files:
        print(f"Error: No review files found in {REVIEWS_DIR}")
        return []
    
    # Sort files by their first number (the start of the range)
    def extract_start_number(filepath):
        # Extract the first number from filename (e.g., "1" from "reviews-1-115.csv")
        filename = filepath.stem  # removes .csv
        parts = filename.split('-')
        if len(parts) >= 2:
            try:
                return int(parts[1])  # returns the first number
            except ValueError:
                return 0
        return 0
    
    sorted_files = sorted(all_files, key=extract_start_number)
    
    print(f"Found {len(sorted_files)} review files:")
    for f in sorted_files:
        print(f"  - {f.name}")
    
    return sorted_files


def merge_review_files_with_chunking(chunk_size=100000):
    """
    Merge all review files into one CSV file using chunking to handle large data.
    
    Args:
        chunk_size: Number of rows to process at a time (for memory efficiency)
    """
    print("\n" + "="*60)
    print("MERGING STEAM REVIEW FILES")
    print("="*60)
    
    # Get sorted files
    sorted_files = get_sorted_review_files()
    if not sorted_files:
        return False
    
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Chunk size: {chunk_size:,} rows")
    print("\nStarting merge process...")
    
    # First, check if we need to use chunking or can load all at once
    total_rows_estimate = 0
    for f in sorted_files:
        try:
            # Get row count using pandas (fast)
            df_sample = pd.read_csv(f, nrows=1000)
            if len(df_sample) > 0:
                # Estimate total rows by file size
                file_size = f.stat().st_size
                row_estimate = file_size / 1000 * 1000  # rough estimate
                total_rows_estimate += row_estimate
        except:
            pass
    
    print(f"Estimated total rows: {total_rows_estimate:,.0f}")
    
    # Method 1: If files are small enough, merge all at once
    total_size_gb = sum(f.stat().st_size for f in sorted_files) / (1024**3)
    
    if total_size_gb < 2:  # If less than 2GB, try to load all at once
        print(f"Total size: {total_size_gb:.2f} GB - Attempting to load all at once...")
        try:
            all_dfs = []
            for file in tqdm(sorted_files, desc="Reading files"):
                df = pd.read_csv(file)
                all_dfs.append(df)
                print(f"  Loaded: {file.name} ({len(df):,} rows)")
            
            # Merge all DataFrames
            merged_df = pd.concat(all_dfs, ignore_index=True)
            
            # Save to CSV
            print(f"\nSaving merged file ({len(merged_df):,} rows)...")
            merged_df.to_csv(OUTPUT_FILE, index=False)
            
            # Clean up
            del all_dfs, merged_df
            gc.collect()
            
            print(f"Successfully merged all files into {OUTPUT_FILE}")
            return True
            
        except MemoryError:
            print("Warning: Memory error occurred. Falling back to chunked approach...")
            # Fall through to chunked approach
    
    # Method 2: Chunked approach (more memory efficient)
    print(f"Using chunked approach to merge {len(sorted_files)} files...")
    
    # Open output file for writing
    first_file = True
    total_rows_written = 0
    
    try:
        for file_idx, file_path in enumerate(tqdm(sorted_files, desc="Processing files")):
            print(f"\nProcessing {file_path.name}...")
            
            # Read the file in chunks
            chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
            
            for chunk_idx, chunk in enumerate(chunk_iter):
                # Write header only for the first chunk of the first file
                header = (file_idx == 0 and chunk_idx == 0)
                
                # Append to CSV
                mode = 'w' if header else 'a'
                chunk.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
                
                total_rows_written += len(chunk)
                
                # Print progress every 5 chunks
                if chunk_idx % 5 == 0:
                    print(f"  Wrote {total_rows_written:,} rows so far...")
            
            # Force garbage collection after each file
            gc.collect()
        
        # Verify the output
        if OUTPUT_FILE.exists():
            file_size_gb = OUTPUT_FILE.stat().st_size / (1024**3)
            print(f"\nSuccessfully merged all files!")
            print(f"   Output file: {OUTPUT_FILE}")
            print(f"   Total rows written: {total_rows_written:,}")
            print(f"   File size: {file_size_gb:.2f} GB")
            return True
        else:
            print("Failed to create output file")
            return False
            
    except Exception as e:
        print(f"Error during merge: {e}")
        return False


def verify_merged_file(sample_size=10000):
    """
    Verify the merged file by loading a sample and checking its structure.
    """
    print("\n" + "="*60)
    print("VERIFYING MERGED FILE")
    print("="*60)
    
    try:
        if not OUTPUT_FILE.exists():
            print(f"Output file not found: {OUTPUT_FILE}")
            return False
        
        # Get total row count (fast)
        with open(OUTPUT_FILE, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # subtract header
            
        print(f"Total rows in merged file: {total_lines:,}")
        
        # Load a sample to verify structure
        sample_df = pd.read_csv(OUTPUT_FILE, nrows=sample_size)
        
        print(f"\nSample preview ({sample_size} rows):")
        print(f"Columns: {', '.join(sample_df.columns[:10])}")
        if len(sample_df.columns) > 10:
            print(f"  ... and {len(sample_df.columns) - 10} more columns")
        
        print(f"\nData types:")
        print(sample_df.dtypes[:5])
        if len(sample_df.dtypes) > 5:
            print(f"  ... and {len(sample_df.dtypes) - 5} more columns")
        
        print("\nFirst 3 rows:")
        print(sample_df.head(3).to_string())
        
        return True
        
    except Exception as e:
        print(f"Error verifying merged file: {e}")
        return False


def main():
    """
    Main function to merge all review files.
    """
    # Check if reviews directory exists
    if not REVIEWS_DIR.exists():
        print(f"Reviews directory not found: {REVIEWS_DIR}")
        print("Please run the download script first to get the data.")
        return
    
    # Merge the files
    success = merge_review_files_with_chunking(chunk_size=100000)
    
    if success:
        # Verify the merged file
        verify_merged_file()
        
        # Provide usage examples
        print("\n" + "="*60)
        print("HOW TO USE THE MERGED FILE")
        print("="*60)
        print("\n# Load the entire merged file (requires ~15GB RAM):")
        print("import pandas as pd")
        print("df = pd.read_csv('data/steam_reviews_merged.csv')")
        print()
        print("# Load in chunks (recommended for large datasets):")
        print("chunk_iter = pd.read_csv('data/steam_reviews_merged.csv', chunksize=100000)")
        print("for chunk in chunk_iter:")
        print("    # Process each chunk")
        print("    print(f'Processing {len(chunk)} rows')")
        print()
        print("# Load specific columns (memory efficient):")
        print("columns = ['appid', 'steamid', 'review', 'voted_up', 'playtime_forever']")
        print("df = pd.read_csv('data/steam_reviews_merged.csv', usecols=columns)")
    else:
        print("\nFailed to merge review files. Please check for errors above.")
        print("You might need to:")
        print("1. Make sure the reviews directory contains all CSV files")
        print("2. Ensure you have enough disk space (~5GB)")
        print("3. Try reducing the chunk_size parameter")


if __name__ == "__main__":
    # Check for required packages
    try:
        import pandas as pd
        from tqdm import tqdm
    except ImportError as e:
        print("Missing required package. Please install:")
        print("pip install pandas tqdm")
        exit(1)
    
    main()