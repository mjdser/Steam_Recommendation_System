"""
Steam API Data Collection Script - Clean Version
Only stores public profiles with actual game data
Returns None for any user that should be skipped
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set
import logging
from tqdm import tqdm
from pathlib import Path

# Create data directory if it doesn't exist
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Configure logging
log_file = DATA_DIR / "steam_api_fetch.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SteamAPIClient:
    """Steam API Client - returns None for any user that should be skipped"""
    
    def __init__(self, api_key: str, rate_limit_delay: float = 1.0):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://api.steampowered.com"
        self.last_request_time = 0
        self.session = requests.Session()
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        """Make API request with error handling"""
        self._rate_limit()
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self._make_request(url, params)
            
            if response.status_code == 403:
                logger.warning(f"Access denied (403) for {url}")
                return None
            
            response.raise_for_status()
            
            if not response.text:
                return None
            
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error for {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
    
    def get_player_summary(self, steamid: int) -> Optional[Dict]:
        """Get player profile information"""
        url = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v2/"
        params = {
            'key': self.api_key,
            'steamids': steamid
        }
        
        response = self._make_request(url, params)
        
        if response and isinstance(response, dict) and 'response' in response:
            players = response['response'].get('players', [])
            if players:
                return players[0]
        
        return None
    
    def get_owned_games(self, steamid: int) -> Optional[Dict]:
        """Get list of owned games with playtime"""
        url = f"{self.base_url}/IPlayerService/GetOwnedGames/v0001/"
        params = {
            'key': self.api_key,
            'steamid': steamid,
            'include_appinfo': True,
            'include_played_free_games': True
        }
        
        response = self._make_request(url, params)
        
        if response and isinstance(response, dict) and 'response' in response:
            if not response['response']:
                return None
            
            games = response['response'].get('games', [])
            return {
                'game_count': response['response'].get('game_count', len(games)),
                'games': games
            }
        
        return None
    
    def get_player_achievements(self, steamid: int, appid: int) -> Optional[Dict]:
        """Get achievements for a specific user and game"""
        url = f"{self.base_url}/ISteamUserStats/GetPlayerAchievements/v1/"
        params = {
            'key': self.api_key,
            'steamid': steamid,
            'appid': appid
        }
        
        response = self._make_request(url, params)
        
        if response and isinstance(response, dict) and 'playerstats' in response:
            stats = response['playerstats']
            if stats and stats.get('success'):
                return {
                    'steamid': stats.get('steamID', ''),
                    'game_name': stats.get('gameName', ''),
                    'achievements': stats.get('achievements', []),
                    'success': stats.get('success', False)
                }
        
        return None
    
    def get_recently_played_games(self, steamid: int, count: int = 10) -> Optional[Dict]:
        """Get recently played games"""
        url = f"{self.base_url}/IPlayerService/GetRecentlyPlayedGames/v1/"
        params = {
            'key': self.api_key,
            'steamid': steamid,
            'count': count
        }
        
        response = self._make_request(url, params)
        
        if response and isinstance(response, dict) and 'response' in response:
            if not response['response']:
                return None
            
            games = response['response'].get('games', [])
            return {
                'total_count': response['response'].get('total_count', len(games)),
                'games': games
            }
        
        return None
    
    def get_user_data(self, steamid: int, top_n_games: int = 5) -> Optional[Dict]:
        """
        Get all user data. Returns None for any user that should be skipped.
        Only returns a dict for valid public profiles with game data.
        """
        # Get player summary
        summary = self.get_player_summary(steamid)
        if summary is None:
            logger.debug(f"User {steamid}: No profile found - skipping")
            return None
        
        # Check if profile is public
        community_visibility = summary.get('communityvisibilitystate', 0)
        if community_visibility != 3:
            logger.debug(f"User {steamid}: Profile is private/friends-only (visibility: {community_visibility}) - skipping")
            return None
        
        # Get owned games
        owned_games = self.get_owned_games(steamid)
        if owned_games is None:
            logger.debug(f"User {steamid}: Could not fetch owned games - skipping")
            return None
        
        # Build user data only for valid public profiles
        user_data = {
            'steamid': steamid,
            'summary': summary,
            'owned_games': owned_games,
            'recently_played': None,
            'achievements_by_game': {}
        }
        
        # Get achievements for top N most-played games
        games = owned_games.get('games', [])
        if games:
            sorted_games = sorted(games, key=lambda x: x.get('playtime_forever', 0), reverse=True)
            top_games = sorted_games[:top_n_games]
            
            achievement_data = {}
            for game in top_games:
                appid = game.get('appid')
                if appid:
                    ach_response = self.get_player_achievements(steamid, appid)
                    if ach_response and ach_response.get('success'):
                        achievements = ach_response.get('achievements', [])
                        unlocked = [a for a in achievements if a.get('achieved') == 1]
                        achievement_data[appid] = {
                            'game_name': ach_response.get('game_name', ''),
                            'total_achievements': len(achievements),
                            'unlocked_achievements': len(unlocked),
                            'unlocked_percentage': (len(unlocked) / len(achievements) * 100) if achievements else 0,
                            'achievement_list': achievements
                        }
            
            user_data['achievements_by_game'] = achievement_data
        
        # Get recently played games
        recently_played = self.get_recently_played_games(steamid)
        if recently_played is not None:
            user_data['recently_played'] = recently_played
        
        return user_data


class SteamDataCollector:
    """Main collector - only stores public profiles with valid data"""
    
    def __init__(self, api_key: str, data_folder: str = 'data', top_n_games: int = 5):
        self.api_key = api_key
        self.data_folder = data_folder
        self.top_n_games = top_n_games
        self.client = SteamAPIClient(api_key, rate_limit_delay=1.0)
        self.user_data_list = []
        self.processed_steamids = set()
        
        os.makedirs(self.data_folder, exist_ok=True)
        
        self.stats = {
            'total_attempted': 0,
            'public_found': 0,
            'private_found': 0,
            'errors': 0,
            'achievement_requests': 0,
            'skipped_existing': 0
        }
    
    def load_processed_ids(self):
        """Load processed Steam IDs from the tracking file (source of truth)"""
        ids_path = os.path.join(self.data_folder, 'processed_steamids.txt')
        if os.path.exists(ids_path):
            try:
                with open(ids_path, 'r') as f:
                    for line in f:
                        sid = line.strip()
                        if sid:
                            self.processed_steamids.add(sid)
                logger.info(f"Loaded {len(self.processed_steamids)} processed IDs from {ids_path}")
            except Exception as e:
                logger.warning(f"Could not load processed IDs: {e}")
        else:
            logger.info(f"No processed IDs file found. Starting fresh.")
    
    def load_users_from_reviews(self, reviews_path: str, n_users: Optional[int] = None) -> List[int]:
        """Load unique Steam IDs from reviews dataset"""
        logger.info(f"Loading Steam IDs from {reviews_path}")
        
        try:
            df = pd.read_csv(reviews_path, usecols=['steamid'])
            unique_users = df['steamid'].unique()
            logger.info(f"Found {len(unique_users)} unique users")
            
            # Filter out already processed users
            unprocessed = [uid for uid in unique_users if str(uid) not in self.processed_steamids]
            logger.info(f"Found {len(unprocessed)} unprocessed users")
            
            if n_users and n_users < len(unprocessed):
                # Take the first N unprocessed users
                sampled_users = unprocessed[:n_users]
                logger.info(f"Taking first {len(sampled_users)} unprocessed users")
                return sampled_users
            
            return unprocessed
            
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            return []
    
    def _save_processed_ids(self):
        """Save processed Steam IDs to the tracking file"""
        ids_path = os.path.join(self.data_folder, 'processed_steamids.txt')
        with open(ids_path, 'w') as f:
            for sid in sorted(self.processed_steamids):
                f.write(f"{sid}\n")
        logger.debug(f"Saved {len(self.processed_steamids)} processed IDs to {ids_path}")
    
    def _save_output_csv(self, output_path: str):
        """Save current data to CSV"""
        if not self.user_data_list:
            return
        
        df = self._create_dataframe_from_list()
        
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['steamid'])
            combined_df.to_csv(output_path, index=False)
            logger.debug(f"Appended to {output_path} ({len(combined_df)} users)")
        else:
            df.to_csv(output_path, index=False)
            logger.debug(f"Created {output_path} with {len(df)} users")
    
    def _save_collection_stats(self):
        """Save collection statistics to a file"""
        stats_path = os.path.join(self.data_folder, 'collection_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("STEAM API DATA COLLECTION STATISTICS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total users attempted: {self.stats['total_attempted']}\n")
            f.write(f"Public profiles stored: {self.stats['public_found']}\n")
            f.write(f"Private/friends-only profiles: {self.stats['private_found']}\n")
            f.write(f"Errors: {self.stats['errors']}\n")
            f.write(f"Achievement API calls made: {self.stats['achievement_requests']}\n")
        logger.info(f"Collection stats saved to {stats_path}")
    
    def _log_stats(self):
        """Log current statistics"""
        logger.info(f"""
        Current Stats:
        Total attempted: {self.stats['total_attempted']}
        Public profiles found: {self.stats['public_found']}
        Private/friends-only: {self.stats['private_found']}
        Errors: {self.stats['errors']}
        Achievement requests: {self.stats['achievement_requests']}
        """)
    
    def _create_dataframe_from_list(self) -> pd.DataFrame:
        """Create DataFrame from collected user data"""
        if not self.user_data_list:
            return pd.DataFrame()
        
        rows = []
        for user_data in self.user_data_list:
            try:
                row = self._extract_features(user_data)
                if row:
                    rows.append(row)
            except Exception as e:
                logger.error(f"Error extracting features for user {user_data.get('steamid', 'unknown')}: {e}")
                continue
        
        return pd.DataFrame(rows)
    
    def _extract_features(self, user_data: Dict) -> Dict:
        """
        Extract features from valid user data.
        Assumes all data is present (no None values).
        """
        steamid = user_data['steamid']
        summary = user_data['summary']
        owned_games_data = user_data['owned_games']
        recently_played_data = user_data.get('recently_played')
        achievements_by_game = user_data.get('achievements_by_game', {})
        
        # Get top_n_games from the collector instance
        top_n_games = self.top_n_games
        
        # Basic profile features
        features = {
            'steamid': steamid,
            'profile_name': summary.get('personaname', ''),
            'real_name': summary.get('realname', ''),
            'profile_state': summary.get('personastate', 0),
            'community_visibility': summary.get('communityvisibilitystate', 0),
            'country_code': summary.get('loccountrycode', ''),
        }
        
        # Handle timecreated safely
        if summary.get('timecreated'):
            features['member_since'] = datetime.fromtimestamp(summary.get('timecreated')).strftime('%Y-%m-%d')
        else:
            features['member_since'] = None
        
        # Game library features
        games = owned_games_data.get('games', [])
        game_count = owned_games_data.get('game_count', 0)
        
        if games:
            playtimes = [game.get('playtime_forever', 0) for game in games]
            playtime_2weeks = [game.get('playtime_2weeks', 0) for game in games if 'playtime_2weeks' in game]
            
            features.update({
                'game_count_total': game_count,
                'games_count_returned': len(games),
                'total_playtime_hours': sum(playtimes) / 60,
                'avg_playtime_hours': (sum(playtimes) / len(playtimes)) / 60 if playtimes else 0,
                'max_playtime_hours': max(playtimes) / 60 if playtimes else 0,
                'median_playtime_hours': sorted(playtimes)[len(playtimes)//2] / 60 if playtimes else 0,
            })
            
            # 2-week playtime stats
            if playtime_2weeks:
                features.update({
                    'total_playtime_2weeks_hours': sum(playtime_2weeks) / 60,
                    'avg_playtime_2weeks_hours': (sum(playtime_2weeks) / len(playtime_2weeks)) / 60,
                    'active_games_2weeks': len([p for p in playtime_2weeks if p > 0]),
                })
            else:
                features.update({
                    'total_playtime_2weeks_hours': 0,
                    'avg_playtime_2weeks_hours': 0,
                    'active_games_2weeks': 0,
                })
            
            # Most played games - use top_n_games dynamically
            sorted_games = sorted(games, key=lambda x: x.get('playtime_forever', 0), reverse=True)
            for i, game in enumerate(sorted_games[:top_n_games], 1):
                appid = game.get('appid')
                features[f'top_{i}_appid'] = appid
                features[f'top_{i}_playtime_hours'] = game.get('playtime_forever', 0) / 60
                features[f'top_{i}_name'] = game.get('name', '')
                
                # Add achievement data for this game if available
                if appid and appid in achievements_by_game:
                    ach = achievements_by_game[appid]
                    features[f'top_{i}_achievements_total'] = ach.get('total_achievements', 0)
                    features[f'top_{i}_achievements_unlocked'] = ach.get('unlocked_achievements', 0)
                    features[f'top_{i}_achievement_percentage'] = ach.get('unlocked_percentage', 0)
                else:
                    features[f'top_{i}_achievements_total'] = 0
                    features[f'top_{i}_achievements_unlocked'] = 0
                    features[f'top_{i}_achievement_percentage'] = 0
            
            # Aggregate achievement stats from all fetched games
            total_ach_available = 0
            total_ach_unlocked = 0
            completion_rates = []
            
            for appid, ach in achievements_by_game.items():
                total_ach_available += ach.get('total_achievements', 0)
                total_ach_unlocked += ach.get('unlocked_achievements', 0)
                if ach.get('total_achievements', 0) > 0:
                    completion_rates.append(ach.get('unlocked_percentage', 0))
            
            features.update({
                'achievement_games_fetched': len(achievements_by_game),
                'total_achievements_available': total_ach_available,
                'total_achievements_unlocked': total_ach_unlocked,
                'avg_achievement_completion_rate': np.mean(completion_rates) if completion_rates else 0,
                'max_achievement_completion_rate': max(completion_rates) if completion_rates else 0,
            })
        else:
            # Public profile but no games (rare but possible)
            features.update({
                'game_count_total': 0,
                'games_count_returned': 0,
                'total_playtime_hours': 0,
                'avg_playtime_hours': 0,
                'max_playtime_hours': 0,
                'median_playtime_hours': 0,
                'total_playtime_2weeks_hours': 0,
                'avg_playtime_2weeks_hours': 0,
                'active_games_2weeks': 0,
                'achievement_games_fetched': 0,
                'total_achievements_available': 0,
                'total_achievements_unlocked': 0,
                'avg_achievement_completion_rate': 0,
                'max_achievement_completion_rate': 0,
            })
            
            # Top games with zeros - use top_n_games
            for i in range(1, top_n_games + 1):
                features[f'top_{i}_appid'] = None
                features[f'top_{i}_playtime_hours'] = 0
                features[f'top_{i}_name'] = ''
                features[f'top_{i}_achievements_total'] = 0
                features[f'top_{i}_achievements_unlocked'] = 0
                features[f'top_{i}_achievement_percentage'] = 0
        
        # Recently played features
        if recently_played_data:
            recent_games = recently_played_data.get('games', [])
            if recent_games:
                features.update({
                    'recent_games_count': len(recent_games),
                    'recent_total_playtime_hours': sum([g.get('playtime_2weeks', 0) for g in recent_games]) / 60,
                })
                recent_appids = [g.get('appid') for g in recent_games[:5]]
                features['recent_game_appids'] = ','.join(map(str, recent_appids))
            else:
                features.update({
                    'recent_games_count': 0,
                    'recent_total_playtime_hours': 0,
                    'recent_game_appids': '',
                })
        else:
            features.update({
                'recent_games_count': 0,
                'recent_total_playtime_hours': 0,
                'recent_game_appids': '',
            })
        
        return features
    
    def process_users(self, steamids: List[int], output_path: str, 
                      save_interval: int = 50) -> None:
        """Process users and only store public profiles"""
        
        steamids_to_fetch = [
            sid for sid in steamids 
            if str(sid) not in self.processed_steamids
        ]
        
        if not steamids_to_fetch:
            logger.info("All users in this batch already processed.")
            self.stats['skipped_existing'] = len(steamids)
            return
        
        logger.info(f"Processing {len(steamids_to_fetch)} new users")
        logger.info(f"Skipping {len(steamids) - len(steamids_to_fetch)} already processed")
        
        pbar = tqdm(steamids_to_fetch, desc="Fetching user data", unit="users")
        
        for idx, steamid in enumerate(pbar, 1):
            pbar.set_description(f"Processing {steamid}")
            
            self.stats['total_attempted'] += 1
            
            # get_user_data returns None for any user we should skip
            user_data = self.client.get_user_data(steamid, top_n_games=self.top_n_games)
            
            if user_data is not None:
                # Valid public profile with data
                self.stats['public_found'] += 1
                ach_count = len(user_data.get('achievements_by_game', {}))
                self.stats['achievement_requests'] += ach_count
                self.user_data_list.append(user_data)
            
            # Mark as processed regardless of whether data was found
            self.processed_steamids.add(str(steamid))
            
            # Save progress every save_interval users
            if idx % save_interval == 0:
                self._save_processed_ids()
                self._save_output_csv(output_path)
                self._log_stats()
        
        # Final save for this batch
        self._save_processed_ids()
        self._save_output_csv(output_path)
        self._save_collection_stats()
        
        logger.info(f"Batch complete. Added {self.stats['public_found']} new public users")
        logger.info(f"Private/error users skipped: {self.stats['private_found']}")


def main():
    """Main execution function"""
    # Configuration
    API_KEY = "23ABD213199510AE8CB52F1784A00D93" 
    DATA_FOLDER = "data"
    REVIEWS_PATH = os.path.join(DATA_FOLDER, "steam_reviews_merged.csv")
    OUTPUT_PATH = os.path.join(DATA_FOLDER, "steam_user_api_data.csv")
    SAMPLE_SIZE = 200
    TOP_N_GAMES = 10

    if API_KEY == "YOUR_API_KEY_HERE":
        logger.error("Please set your API key in the script!")
        return
    
    if not os.path.exists(REVIEWS_PATH):
        logger.error(f"Reviews file not found: {REVIEWS_PATH}")
        return

    # Create collector once
    collector = SteamDataCollector(API_KEY, data_folder=DATA_FOLDER, top_n_games=TOP_N_GAMES)
    
    # Load processed IDs from the tracking file (source of truth)
    collector.load_processed_ids()
    
    logger.info(f"Starting collection with {len(collector.processed_steamids)} already processed users")
    
    batch_num = 0
    while True:
        batch_num += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH {batch_num} - Loading next set of users...")
        logger.info(f"{'='*60}")
        
        # Load users (respects already processed IDs)
        steamids = collector.load_users_from_reviews(REVIEWS_PATH, n_users=SAMPLE_SIZE)
        
        if not steamids:
            logger.info("No more unprocessed users found. Collection complete!")
            break
        
        # Process the batch
        collector.process_users(
            steamids=steamids,
            output_path=OUTPUT_PATH,
            save_interval=50
        )
        
        # Log progress
        logger.info(f"""
        Batch {batch_num} Complete!
        =====================
        Total users attempted: {collector.stats['total_attempted']}
        Public profiles stored: {collector.stats['public_found']}
        Private/friends-only (skipped): {collector.stats['private_found']}
        Errors: {collector.stats['errors']}
        Achievement API calls made: {collector.stats['achievement_requests']}
        """)
        
        # Add a small delay between batches
        time.sleep(1)
    
    # Print final statistics
    logger.info(f"""
    Collection Complete!
    =====================
    Total batches processed: {batch_num}
    Total users attempted: {collector.stats['total_attempted']}
    Public profiles stored: {collector.stats['public_found']}
    Private/friends-only (skipped): {collector.stats['private_found']}
    Errors: {collector.stats['errors']}
    Achievement API calls made: {collector.stats['achievement_requests']}
    Output saved to: {OUTPUT_PATH}
    """)
    
    # Final achievement statistics
    if os.path.exists(OUTPUT_PATH):
        final_df = pd.read_csv(OUTPUT_PATH)
        if len(final_df) > 0 and 'total_achievements_unlocked' in final_df.columns:
            non_zero_ach = len(final_df[final_df['total_achievements_unlocked'] > 0])
            logger.info(f"""
    Final Achievement Stats:
    =====================
    Total users collected: {len(final_df)}
    Users with achievements: {non_zero_ach} ({non_zero_ach/len(final_df)*100:.1f}%)
    Avg achievements unlocked: {final_df['total_achievements_unlocked'].mean():.1f}
    Max achievements unlocked: {final_df['total_achievements_unlocked'].max()}
    Avg completion rate: {final_df['avg_achievement_completion_rate'].mean():.2%}
    """)


if __name__ == "__main__":
    main()