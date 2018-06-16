"""
Functions used to clean dataframes used for post processing.

Tausif S., 2018
"""

import numpy as np
import pandas as pd
import re

def cleanText(rawDF):
    """
    Takes in a raw text dataframe and returns a cleaned dataframe.
    Useful for tweet related cleanDFs. Text must be in 'text' column.

    Input:
        rawDF   : Raw panda dataframe with a 'text' column.
    
    Output:
        cleanDF : Cleaned text panda dataframe. 
    """
    
    # Creating a copy of the original dataframe.
    cleanDF = rawDF.copy()

    # Removing html tags and attributes.
    htmlTagsRgx = r'<[^>]+>'
    cleanDF['text'] = cleanDF['text'].str.replace(htmlTagsRgx, '')
    
    # Replacing html character codes.
    htmlCharDict = {'&amp;': '&', '&gt;': '>', '&lt;': '<', '&quot;': '\"'}
    
    for key, value in htmlCharDict.items():
        cleanDF['text'] = cleanDF['text'].str.replace(key, htmlCharDict[key])

    # Replacing literal \' to literal apostrophes.
    litAposRgx = r'\\+\''
    cleanDF['text'] = cleanDF['text'].str.replace(litAposRgx, '\'')

    # Removing all urls.
    urlRgx = r'(?:\S+(?=\.[a-zA-Z])\S+)'
    cleanDF['text'] = cleanDF['text'].str.replace(urlRgx, '')

    # Removing RT tags.
    cleanDF['text'] = cleanDF['text'].str.replace("RT", '')

    # Removing Emojis
    #emojiRgx = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    #cleanDF['text'] = cleanDF['text'].str.replace(emojiRgx, '')

    # Forcing all string characters to be lower case.
    cleanDF['text'] = cleanDF['text'].str.lower()

    # Removing twitter handles.
    handle_rgx = r'@\S+'
    cleanDF['text'] = cleanDF['text'].str.replace(handle_rgx, '')

    return cleanDF


def cleanNumerical(rawDF):
    """
    Takes in the raw dataframe, and returns a cleaned numerical dataframe.
    Fills empty continuous features with its mean and empty categorical
    features with the feature's mode.
    
    Input:
        rawDF   : Raw M x 15 panda dataframe.
    
    Output:
        cleanDF : Cleaned numerical M x 15 panda dataframe.
    """
    
    # Creating a copy of the original dataframe.
    cleanDF = rawDF.copy()
    
    # Running through each feature of the dataframe.
    for i in range(0, cleanDF.shape[1]):
        # Checking to see if the feature is non-numerical, i.e. a categorical feature. 
        if(cleanDF.iloc[:, i].dtypes == 'object'):
            # Converting object dtype to category.
            cleanDF.iloc[:, i] = cleanDF.iloc[:, i].astype('category')
            # Filling empty values with the feature's mode. 
            cleanDF.iloc[:, i] = cleanDF.iloc[:, i].fillna(cleanDF.iloc[:, i].mode()[0])
            # Encoding to numerical values.
            cleanDF.iloc[:, i] = cleanDF.iloc[:, i].cat.codes
        else:
            # Filling empty numerical feature values with its mean. 
            cleanDF.iloc[:, i] = cleanDF.iloc[:, i].fillna(round(cleanDF.iloc[:, i].mean()))
    
    return cleanDF