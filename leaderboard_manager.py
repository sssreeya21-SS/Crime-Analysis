import pandas as pd
import os
from datetime import datetime
from tabulate import tabulate
import hashlib
from config import LEADERBOARD_PATH

class LeaderboardManager:
    def __init__(self):
        self.leaderboard_path = LEADERBOARD_PATH
        self.leaderboard = self.load_leaderboard()
    
    def load_leaderboard(self):
        """Load existing leaderboard or create new one"""
        if os.path.exists(self.leaderboard_path):
            return pd.read_csv(self.leaderboard_path)
        else:
            return pd.DataFrame(columns=[
                'Timestamp', 'User', 'Model', 'Accuracy', 'F1 Score', 
                'Precision', 'Recall', 'User_ID'
            ])
    
    def save_leaderboard(self):
        """Save leaderboard to CSV"""
        self.leaderboard = self.leaderboard.sort_values('F1 Score', ascending=False)
        self.leaderboard.to_csv(self.leaderboard_path, index=False)
        print(f"💾 Leaderboard saved to {self.leaderboard_path}")
    
    def add_entry(self, user_name, model_name, accuracy, f1, precision, recall):
        """Add a new entry to the leaderboard"""
        # Create a unique user ID
        user_id = hashlib.md5(user_name.encode()).hexdigest()[:8]
        
        new_entry = pd.DataFrame([{
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'User': user_name,
            'Model': model_name,
            'Accuracy': round(accuracy, 4),
            'F1 Score': round(f1, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'User_ID': user_id
        }])
        
        self.leaderboard = pd.concat([self.leaderboard, new_entry], ignore_index=True)
        self.save_leaderboard()
        
        # Return rank
        rank = self.get_user_rank(user_name)
        return rank
    
    def get_user_rank(self, user_name):
        """Get the rank of a specific user"""
        sorted_df = self.leaderboard.sort_values('F1 Score', ascending=False)
        user_entries = sorted_df[sorted_df['User'] == user_name]
        if len(user_entries) > 0:
            return user_entries.index[0] + 1
        return None
    
    def get_top_performers(self, n=10):
        """Get top N performers"""
        return self.leaderboard.nlargest(n, 'F1 Score')[['User', 'Model', 'F1 Score', 'Accuracy']]
    
    def display_leaderboard(self):
        """Display the leaderboard in a formatted table"""
        if len(self.leaderboard) == 0:
            print("\n📋 No entries in leaderboard yet. Be the first to submit!")
            return
        
        print("\n" + "="*80)
        print("🏆 LEADERBOARD 🏆")
        print("="*80)
        
        # Prepare display data
        display_df = self.leaderboard.copy()
        display_df = display_df.sort_values('F1 Score', ascending=False)
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        display_df = display_df[['Rank', 'User', 'Model', 'F1 Score', 'Accuracy', 'Precision', 'Recall', 'Timestamp']]
        
        # Truncate long names
        display_df['User'] = display_df['User'].apply(lambda x: x[:15] + '...' if len(str(x)) > 15 else x)
        display_df['Model'] = display_df['Model'].apply(lambda x: x[:20] + '...' if len(str(x)) > 20 else x)
        
        print(tabulate(display_df.head(20), headers='keys', tablefmt='grid', showindex=False))
        
        # Show top performer
        top = self.leaderboard.loc[self.leaderboard['F1 Score'].idxmax()]
        print(f"\n🥇 Current Leader: {top['User']} with {top['Model']}")
        print(f"   F1 Score: {top['F1 Score']:.4f} | Accuracy: {top['Accuracy']:.4f}")
        print(f"   Total participants: {self.leaderboard['User_ID'].nunique()}")
        print(f"   Total submissions: {len(self.leaderboard)}")
    
    def display_user_history(self, user_name):
        """Display all submissions by a specific user"""
        user_entries = self.leaderboard[self.leaderboard['User'] == user_name]
        if len(user_entries) == 0:
            print(f"\n❌ No entries found for user: {user_name}")
            return
        
        print(f"\n📊 Submission history for: {user_name}")
        print("-" * 60)
        display_df = user_entries[['Timestamp', 'Model', 'F1 Score', 'Accuracy']].sort_values('F1 Score', ascending=False)
        print(tabulate(display_df, headers='keys', tablefmt='simple', showindex=False))
    
    def clear_leaderboard(self, confirm=False):
        """Clear the leaderboard (with confirmation)"""
        if confirm:
            self.leaderboard = pd.DataFrame(columns=self.leaderboard.columns)
            self.save_leaderboard()
            print("🗑️ Leaderboard cleared!")
        else:
            print("❌ Leaderboard not cleared (confirmation required)")
    
    def get_statistics(self):
        """Get leaderboard statistics"""
        if len(self.leaderboard) == 0:
            return {"total_submissions": 0, "unique_users": 0}
        
        return {
            "total_submissions": len(self.leaderboard),
            "unique_users": self.leaderboard['User_ID'].nunique(),
            "avg_f1_score": self.leaderboard['F1 Score'].mean(),
            "max_f1_score": self.leaderboard['F1 Score'].max(),
            "best_user": self.leaderboard.loc[self.leaderboard['F1 Score'].idxmax(), 'User'],
            "best_model": self.leaderboard.loc[self.leaderboard['F1 Score'].idxmax(), 'Model']
        }
