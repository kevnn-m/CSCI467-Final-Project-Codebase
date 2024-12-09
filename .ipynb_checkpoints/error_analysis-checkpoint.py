import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from RF import train_and_evaluate_tree_model
from SVM import train_and_evaluate_svm
from LR import train_and_evaluate
from MLP import train_and_evaluate_mlp

def check_class_imbalance(y):
    """Check and print class distribution information"""
    class_counts = y.value_counts()
    total_count = len(y)
    print("Class Distribution:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} samples ({count / total_count * 100:.2f}%)")

def plot_feature_distributions(df, feature_cols):
    """Create separate visualizations of feature distributions for successful vs failed predictions"""
    plt.style.use('seaborn')
    
    for feature in feature_cols:
        if feature != 'FUEL':  
            plt.figure(figsize=(12, 7))
            
            sns.kdeplot(data=df[df['STATUS'] == 1][feature], 
                       label='Extinguished', 
                       color='green',
                       linewidth=2)
            sns.kdeplot(data=df[df['STATUS'] == 0][feature], 
                       label='Not Extinguished', 
                       color='red',
                       linewidth=2)
            
            plt.title(f'{feature} Distribution by Fire Status', pad=20, size=14)
            plt.xlabel(feature, size=12)
            plt.ylabel('Density', size=12)
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            stats_text = (
                f"Mean (Extinguished): {df[df['STATUS'] == 1][feature].mean():.2f}\n"
                f"Mean (Not Extinguished): {df[df['STATUS'] == 0][feature].mean():.2f}\n"
                f"Std (Extinguished): {df[df['STATUS'] == 1][feature].std():.2f}\n"
                f"Std (Not Extinguished): {df[df['STATUS'] == 0][feature].std():.2f}\n"
                f"Success Rate: {(len(df[df['STATUS'] == 1]) / len(df) * 100):.1f}%"
            )
            
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', 
                            facecolor='white', 
                            alpha=0.9,
                            edgecolor='gray'),
                    fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f'Output/{feature.lower()}_distribution.png', 
                       bbox_inches='tight',
                       dpi=300)
            plt.close()
    
    if 'FUEL' in feature_cols:
        plt.figure(figsize=(12, 7))
        
        fuel_proportions = df.groupby(['FUEL', 'STATUS']).size().unstack()
        fuel_proportions_pct = fuel_proportions.div(fuel_proportions.sum(axis=1), axis=0) * 100
        
        ax = fuel_proportions_pct.plot(kind='bar', 
                                     stacked=True,
                                     color=['#ff7f7f', '#90EE90'])  
        

        plt.title('Fire Status Distribution by Fuel Type', pad=20, size=14)
        plt.xlabel('Fuel Type', size=12)
        plt.ylabel('Percentage', size=12)
        plt.legend(['Not Extinguished', 'Extinguished'], 
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')
        plt.xticks(rotation=45)
        
        total_samples = fuel_proportions.sum(axis=1)
        y_offset = 5  
        
        for i, total in enumerate(total_samples):
            plt.text(i, 100 + y_offset, 
                    f'n={int(total):,}',
                    ha='center',
                    va='bottom',
                    fontsize=10)
            
            cumsum = 0
            for status in [0, 1]:  
                height = fuel_proportions_pct.iloc[i, status]
                plt.text(i, cumsum + height/2,
                        f'{height:.1f}%',
                        ha='center',
                        va='center',
                        color='black' if height > 20 else 'white',
                        fontsize=10)
                cumsum += height
        
        plt.ylim(0, 115)
        
        plt.tight_layout()
        plt.savefig('Output/fuel_distribution.png', 
                   bbox_inches='tight',
                   dpi=300)
        plt.close()

def plot_model_error_distributions(df, feature_cols, model_name, model, X_test, y_test):
    """Create visualizations of feature distributions for correct vs incorrect predictions"""
    y_pred = get_predictions(model, X_test)
    
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df['true_label'] = y_test
    test_df['predicted'] = y_pred
    test_df['correct'] = y_test == y_pred

    plt.style.use('seaborn')
    
    for feature in feature_cols:
        if feature != 'FUEL':  
            plt.figure(figsize=(12, 7))
            
            sns.kdeplot(data=test_df[test_df['correct']][feature], 
                       label='Correct Predictions', 
                       color='green',
                       linewidth=2)
            sns.kdeplot(data=test_df[~test_df['correct']][feature], 
                       label='Incorrect Predictions', 
                       color='red',
                       linewidth=2)
            
            # Formatting
            plt.title(f'{feature} Distribution by Prediction Correctness\n{model_name}', 
                     pad=20, size=14)
            plt.xlabel(feature, size=12)
            plt.ylabel('Density', size=12)
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add statistics
            stats_text = (
                f"Mean (Correct): {test_df[test_df['correct']][feature].mean():.2f}\n"
                f"Mean (Incorrect): {test_df[~test_df['correct']][feature].mean():.2f}\n"
                f"Std (Correct): {test_df[test_df['correct']][feature].std():.2f}\n"
                f"Std (Incorrect): {test_df[~test_df['correct']][feature].std():.2f}\n"
                f"Accuracy: {(len(test_df[test_df['correct']]) / len(test_df) * 100):.1f}%"
            )
            
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', 
                            facecolor='white', 
                            alpha=0.9,
                            edgecolor='gray'),
                    fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f'Output/{model_name.lower()}_{feature.lower()}_errors.png', 
                       bbox_inches='tight',
                       dpi=300)
            plt.close()

def analyze_model_errors(df, feature_cols):
    X = df[feature_cols].copy()
    y = df['STATUS']
    
    if 'FUEL' in feature_cols:
        label_encoder = LabelEncoder()
        X.loc[:, 'FUEL'] = label_encoder.fit_transform(X['FUEL'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': train_and_evaluate_tree_model,
        'SVM': train_and_evaluate_svm,
        'Logistic Regression': train_and_evaluate,
        'MLP': train_and_evaluate_mlp
    }
    
    for model_name, model_func in models.items():
        print(f"\nAnalyzing {model_name}...")
        model, metrics = model_func(df, feature_cols)
        plot_model_error_distributions(df, feature_cols, model_name, model, X_test, y_test)
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")

def get_predictions(model, X_test):
    """Get predictions from either sklearn or PyTorch model"""
    import torch
    import torch.nn as nn
    
    if isinstance(model, nn.Module):  
        model.eval()  
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test.values)
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()
    else:  
        return model.predict(X_test)

def main():
    print("Loading data...")
    df = pd.read_excel("Acoustic_Extinguisher_Fire_Dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx")
    feature_cols = ['SIZE', 'FUEL', 'DISTANCE', 'DESIBEL', 'AIRFLOW', 'FREQUENCY']
    
    print("Starting Error Analysis...")
    
    print("\n=== Feature Distribution Analysis ===")
    plot_feature_distributions(df, feature_cols)
    
    print("\n=== Model Error Analysis ===")
    analyze_model_errors(df, feature_cols)

if __name__ == "__main__":
    main()