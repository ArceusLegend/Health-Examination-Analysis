import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('numpy_pandas\data\medical_examination.csv')

# Add 'overweight' column
# Use np.where
df['overweight'] = np.where(df['weight']/pow(df['height']/100, 2) > 25, True, False)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df.loc[:, ['cholesterol', 'gluc']] = np.where(df.loc[:, ['cholesterol', 'gluc']] > 1, 1, 0)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 
    # 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars=['cardio'],
                    value_vars=['cholesterol', 'gluc', 'smoke', 
                                'alco', 'active', 'overweight'])
    

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['variable'].count().reset_index(name='total')
    # Convert the 'value' column to an object for hue parameter in catplot() to work (dtype was originally int64)
    df_cat['value'] = df_cat['value'].astype(str)

    # Draw the catplot with 'sns.catplot()'
    # Get the figure for the output
    fig = sns.catplot(data=df_cat, x='variable', y='total', col='cardio',
             kind='bar', errorbar=None, hue='value')  

    # Save and export the figure as catplot.png
    fig.savefig('catplot.png')
    return fig

draw_cat_plot()

# Draw Heat Map
def draw_heat_map():
    # Clean the data

    # 0. Copy original df to new dataframe
    df_heat = df.copy()

    # Create index mask to select and filter out rows based on following conditions:
    # 1. Segments where diastolic pressure (ap_lo) is higher than systolic (ap_hi)
    # 2. Segments where height is less than the 2.5th percentile or greater than the 97.5th percentile
    # 3. Segments where weight is less than the 2.5th percentile or greater than the 97.5th percentile
    cont = df_heat.loc[
        (df['ap_lo'] >= df['ap_hi']) |
        (df['height'] < df['height'].quantile(0.025)) | (df['height'] > df['height'].quantile(0.975)) |
        (df['weight'] < df['weight'].quantile(0.025)) | (df['weight'] > df['weight'].quantile(0.975))
    ].index

    # Remove the rows selected by cont
    df_heat = df_heat.drop(cont)

    # Calculate the correlation matrix
    # Uses the default method ('pearson')
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))



    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", ax=ax)


    #Export the figure as heatmap.png
    fig.savefig('heatmap.png')
    return fig

draw_heat_map()
