"""
Create a synthetic credit risk dataset for the project.
Features mimic real German Credit dataset structure.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    # Numeric features
    'Age': np.random.normal(35, 10, n_samples).clip(18, 70).astype(int),
    'Income': np.random.lognormal(10.5, 0.5, n_samples).astype(int),  # Annual income
    'LoanAmount': np.random.lognormal(9.5, 0.6, n_samples).astype(int),  # Loan amount requested
    'LoanDuration': np.random.choice([12, 24, 36, 48, 60, 72], n_samples),  # Months
    'CreditScore': np.random.normal(650, 100, n_samples).clip(300, 850).astype(int),
    'ExistingCredits': np.random.poisson(1, n_samples).clip(0, 4),
    'Dependents': np.random.poisson(0.5, n_samples).clip(0, 3),
    
    # Categorical features
    'CheckingStatus': np.random.choice(['no_checking', 'lt_0', '0_to_200', 'gt_200'], n_samples, 
                                        p=[0.3, 0.25, 0.35, 0.1]),
    'CreditHistory': np.random.choice(['critical', 'good', 'perfect', 'delayed', 'no_credits'], n_samples,
                                      p=[0.3, 0.35, 0.15, 0.15, 0.05]),
    'Purpose': np.random.choice(['car_new', 'car_used', 'furniture', 'radio_tv', 'education', 
                                  'repairs', 'business', 'domestic'], n_samples),
    'SavingsStatus': np.random.choice(['no_savings', 'lt_100', '100_to_500', '500_to_1000', 'gt_1000'], n_samples,
                                      p=[0.4, 0.3, 0.15, 0.1, 0.05]),
    'Employment': np.random.choice(['unemployed', 'lt_1', '1_to_4', '4_to_7', 'gt_7'], n_samples,
                                   p=[0.05, 0.15, 0.35, 0.30, 0.15]),
    'PersonalStatus': np.random.choice(['male_single', 'female_divorced', 'male_married', 'female_single'], n_samples),
    'Housing': np.random.choice(['own', 'rent', 'free'], n_samples, p=[0.5, 0.35, 0.15]),
    'JobType': np.random.choice(['unemployed', 'unskilled', 'skilled', 'management'], n_samples,
                               p=[0.05, 0.2, 0.55, 0.2]),
    'Telephone': np.random.choice(['none', 'yes'], n_samples, p=[0.4, 0.6]),
    'ForeignWorker': np.random.choice(['yes', 'no'], n_samples, p=[0.1, 0.9]),
}

df = pd.DataFrame(data)

# Create target variable with realistic relationships
def calculate_default_prob(row):
    prob = 0.15  # Base default rate
    
    # Age: younger people have higher default risk
    if row['Age'] < 25:
        prob += 0.10
    elif row['Age'] < 35:
        prob += 0.05
    
    # Credit Score: lower scores = higher risk
    if row['CreditScore'] < 500:
        prob += 0.20
    elif row['CreditScore'] < 600:
        prob += 0.10
    elif row['CreditScore'] < 650:
        prob += 0.05
    
    # Income: lower income = higher risk
    if row['Income'] < 20000:
        prob += 0.10
    elif row['Income'] < 30000:
        prob += 0.05
    
    # Loan to Income ratio
    ratio = row['LoanAmount'] / row['Income']
    if ratio > 2:
        prob += 0.15
    elif ratio > 1.5:
        prob += 0.08
    
    # Credit History
    if row['CreditHistory'] == 'critical':
        prob += 0.15
    elif row['CreditHistory'] == 'delayed':
        prob += 0.08
    elif row['CreditHistory'] == 'no_credits':
        prob += 0.03
    
    # Employment status
    if row['Employment'] == 'unemployed':
        prob += 0.15
    elif row['Employment'] == 'lt_1':
        prob += 0.05
    
    # Checking status (no checking account = slightly riskier)
    if row['CheckingStatus'] == 'no_checking':
        prob += 0.05
    elif row['CheckingStatus'] == 'lt_0':
        prob += 0.08
    
    # Savings status
    if row['SavingsStatus'] == 'no_savings':
        prob += 0.05
    
    # Housing
    if row['Housing'] == 'free':
        prob -= 0.05
    elif row['Housing'] == 'rent':
        prob += 0.03
    
    # Job type
    if row['JobType'] == 'unemployed':
        prob += 0.10
    elif row['JobType'] == 'unskilled':
        prob += 0.05
    elif row['JobType'] == 'management':
        prob -= 0.05
    
    # Existing credits
    if row['ExistingCredits'] >= 3:
        prob += 0.08
    
    # Cap probability between 0.02 and 0.95
    return min(max(prob, 0.02), 0.95)

# Calculate probabilities and generate target
default_probs = df.apply(calculate_default_prob, axis=1)
df['Default'] = (np.random.random(n_samples) < default_probs).astype(int)

# Introduce some missing values (realistic scenario)
missing_indices = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
df.loc[missing_indices[:15], 'Age'] = np.nan
missing_indices2 = np.random.choice(df.index, size=int(0.015 * n_samples), replace=False)
df.loc[missing_indices2[:10], 'CreditScore'] = np.nan
missing_indices3 = np.random.choice(df.index, size=int(0.01 * n_samples), replace=False)
df.loc[missing_indices3[:8], 'CheckingStatus'] = np.nan

print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['Default'].mean():.2%}")
print(f"Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Save to CSV
df.to_csv('/Users/muzamilirfan/Desktop/SUP PROJECT/credit.csv', index=False)
print("\nDataset saved to credit.csv")
