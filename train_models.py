import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import time
from colorama import init, Fore, Style
from tabulate import tabulate

init(autoreset=True)

from config import MODELS, MODELS_DIR, MODEL_COMPARISON_PATH
from data_processor import DataProcessor

# Dynamic imports for models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class ModelTrainer:
    def __init__(self):
        self.processor = DataProcessor()
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        
    def get_model_instance(self, model_name, model_config):
        """Create an instance of the specified model"""
        model_class_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'XGBClassifier': xgb.XGBClassifier,
            'LGBMClassifier': lgb.LGBMClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'CatBoostClassifier': CatBoostClassifier,
            'LogisticRegression': LogisticRegression,
            'SVC': SVC
        }
        
        class_name = model_config['class']
        params = model_config['params']
        
        if class_name in model_class_map:
            return model_class_map[class_name](**params)
        else:
            raise ValueError(f"Unknown model class: {class_name}")
    
    def train_and_evaluate(self, model_name, model_config, X_train, X_test, y_train, y_test):
        """Train a single model and return metrics"""
        print(f"\n{Fore.CYAN}🤖 Training {model_name}...{Style.RESET_ALL}")
        start_time = time.time()
        
        model = self.get_model_instance(model_name, model_config)
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except:
            cv_mean = 0
            cv_std = 0
        
        print(f"   ✅ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Time: {train_time:.2f}s")
        
        return {
            'model': model,
            'name': model_name,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'train_time': train_time
        }
    
    def run_all_models(self):
        """Train and evaluate all models"""
        print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🚀 STARTING MODEL TRAINING{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        
        # Process data
        X_train, X_test, y_train, y_test, feature_cols = self.processor.run_pipeline()
        
        # Train each model
        for model_name, model_config in MODELS.items():
            result = self.train_and_evaluate(
                model_name, model_config, X_train, X_test, y_train, y_test
            )
            self.results.append(result)
            
            if result['accuracy'] > self.best_accuracy:
                self.best_accuracy = result['accuracy']
                self.best_model = result['model']
                self.best_model_name = result['name']
        
        # Save best model
        joblib.dump(self.best_model, f'{MODELS_DIR}/best_model.pkl')
        print(f"\n{Fore.GREEN}💾 Best model saved: {self.best_model_name}{Style.RESET_ALL}")
        
        # Save all models
        for result in self.results:
            joblib.dump(result['model'], f"{MODELS_DIR}/{result['name'].replace(' ', '_')}.pkl")
        
        return self.results
    
    def display_results(self):
        """Display model comparison results"""
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 MODEL COMPARISON RESULTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.results)
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        comparison_df = comparison_df[['name', 'accuracy', 'f1', 'precision', 'recall', 'cv_mean', 'train_time']]
        comparison_df.columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'CV Score', 'Time (s)']
        
        print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".4f"))
        
        # Save comparison
        comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
        print(f"\n💾 Comparison saved to {MODEL_COMPARISON_PATH}")
        
        # Highlight best model
        best = comparison_df.iloc[0]
        print(f"\n{Fore.GREEN}🏆 BEST MODEL: {best['Model']}{Style.RESET_ALL}")
        print(f"   Accuracy: {best['Accuracy']:.4f}")
        print(f"   F1 Score: {best['F1 Score']:.4f}")
        
        return comparison_df
    
    def get_detailed_report(self, model_name, X_test, y_test):
        """Get detailed classification report for a specific model"""
        model_path = f"{MODELS_DIR}/{model_name.replace(' ', '_')}.pkl"
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        print(f"\n{Fore.CYAN}📋 Detailed Report for {model_name}{Style.RESET_ALL}")
        print("-" * 50)
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return classification_report(y_test, y_pred, output_dict=True)
