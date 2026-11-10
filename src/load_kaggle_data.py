import os
import shutil
import pandas as pd
from pathlib import Path
import kagglehub
from tqdm import tqdm

# --- Configuration ---
# Define the data directory
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Define dataset paths
DATASETS = {
    "steam_games": "artermiloff/steam-games-dataset",
    "steam_reviews": "forgemaster/steam-reviews-dataset",
}

# --- Helper Functions ---

def download_steam_games():
    """
    Download the Steam Games dataset.
    This dataset is a single CSV file.
    """
    print("\n" + "="*60)
    print("DOWNLOADING STEAM GAMES DATASET")
    print("="*60)
    
    try:
        # Download the dataset
        print("Downloading from Kaggle...")
        downloaded_path = kagglehub.dataset_download(DATASETS["steam_games"])
        print(f"Downloaded to: {downloaded_path}")
        
        # Find the CSV file
        csv_files = list(Path(downloaded_path).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the downloaded dataset")
        
        # Use the first CSV file found
        source_file = csv_files[0]
        dest_file = DATA_DIR / "steam_games.csv"
        
        # Copy the file
        print(f"Copying {source_file.name} to {dest_file}...")
        shutil.copy2(source_file, dest_file)
        
        # Verify the file was saved
        if dest_file.exists():
            file_size = dest_file.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"Successfully saved: {dest_file} ({file_size:.2f} MB)")
            return str(dest_file)
        else:
            raise Exception("File was not saved successfully")
            
    except Exception as e:
        print(f"Error downloading Steam Games dataset: {e}")
        return None


def download_steam_reviews():
    """
    Download all files from the Steam Reviews dataset.
    This dataset consists of 11 CSV files.
    """
    print("\n" + "="*60)
    print("DOWNLOADING STEAM REVIEWS DATASET")
    print("="*60)
    
    try:
        # Create a subdirectory for reviews
        reviews_dir = DATA_DIR / "steam_reviews"
        reviews_dir.mkdir(exist_ok=True)
        
        # Download the dataset
        print("Downloading from Kaggle...")
        downloaded_path = kagglehub.dataset_download(DATASETS["steam_reviews"])
        print(f"Downloaded to: {downloaded_path}")
        
        # Get all CSV files
        csv_files = list(Path(downloaded_path).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the downloaded dataset")
        
        print(f"Found {len(csv_files)} CSV files to download")
        
        # Copy each file with progress bar
        saved_files = []
        for source_file in tqdm(csv_files, desc="Copying files", unit="file"):
            dest_file = reviews_dir / source_file.name
            shutil.copy2(source_file, dest_file)
            saved_files.append(str(dest_file))
            
            # Print size for each file
            file_size = dest_file.stat().st_size / (1024 * 1024)
            print(f"Saved: {source_file.name} ({file_size:.2f} MB)")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in reviews_dir.glob("*.csv")) / (1024 * 1024 * 1024)
        print(f"\nSuccessfully saved {len(saved_files)} files to {reviews_dir}")
        print(f"   Total size: {total_size:.2f} GB")
        
        return saved_files
        
    except Exception as e:
        print(f"Error downloading Steam Reviews dataset: {e}")
        return None


def load_and_preview(file_path, n_rows=5):
    """
    Load a CSV file and display basic information and preview.
    """
    try:
        print(f"\nPreviewing: {Path(file_path).name}")
        print("-" * 50)
        
        # Read the first few rows to check the file
        df = pd.read_csv(file_path, nrows=n_rows)
        
        print(f"Shape: {df.shape[0]} rows (preview), {df.shape[1]} columns")
        print(f"Columns: {', '.join(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"  ... and {len(df.columns) - 10} more columns")
        
        print("\nFirst 2 rows:")
        print(df.head(2).to_string())
        print("-" * 50)
        
    except Exception as e:
        print(f"Could not preview {file_path}: {e}")


def main():
    """
    Main function to download both datasets.
    """
    print("\n" + "="*60)
    print("STEAM DATASET DOWNLOADER")
    print("="*60)
    print(f"Data will be saved to: {DATA_DIR.absolute()}")
    print("="*60)
    
    # Download Steam Games dataset
    games_file = download_steam_games()
    if games_file:
        load_and_preview(games_file)
    
    # Download Steam Reviews dataset
    review_files = download_steam_reviews()
    if review_files:
        # Preview the first file
        load_and_preview(review_files[0])
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    if games_file:
        print(f"Steam Games: {games_file}")
    else:
        print("Steam Games: Failed to download")
    
    if review_files:
        print(f"Steam Reviews: {len(review_files)} files saved in {DATA_DIR / 'steam_reviews'}")
        print("   Files:")
        for f in sorted(review_files):
            size = Path(f).stat().st_size / (1024 * 1024)
            print(f"   - {Path(f).name} ({size:.2f} MB)")
    else:
        print("Steam Reviews: Failed to download")
    
    print("\n" + "="*60)
    print("Download process completed!")
    print("="*60)
    
    # Provide loading examples
    print("\n To load the datasets in your code:")
    print("-" * 50)
    if games_file:
        print(f"games_df = pd.read_csv('{games_file}')")
    if review_files:
        print(f"reviews_df = pd.read_csv('{review_files[0]}')  # For the first file")
        print("# To load all review files:")
        print("import glob")
        print("all_reviews = []")
        print("for f in glob.glob('data/steam_reviews/*.csv'):")
        print("    all_reviews.append(pd.read_csv(f))")
        print("reviews_df = pd.concat(all_reviews, ignore_index=True)")


if __name__ == "__main__":
    # Install required packages if not present
    try:
        import kagglehub
        import pandas as pd
        from tqdm import tqdm
    except ImportError as e:
        print("Missing required package. Please install:")
        print("pip install kagglehub pandas tqdm")
        exit(1)
    
    main()