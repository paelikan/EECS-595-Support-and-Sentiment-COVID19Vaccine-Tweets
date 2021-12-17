import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import re
from nltk.corpus import stopwords
import emoji
import contractions


SAVE = False

data = pd.read_json(f'dataids.jsonl', lines=True, chunksize=10000, encoding='utf-16')
for i,c in enumerate(data):
    if i == 0:
        tweets = c.drop(['source','in_reply_to_status_id',
       'in_reply_to_status_id_str', 'in_reply_to_user_id',
       'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'possibly_sensitive', 'extended_entities', 'quoted_status_id',
       'quoted_status_id_str', 'quoted_status_permalink', 'quoted_status', 'contributors', 'is_quote_status','source', 'in_reply_to_status_id', 'in_reply_to_status_id_str',
       'in_reply_to_user_id', 'in_reply_to_user_id_str',
       'in_reply_to_screen_name', 'user', 'contributors', 'is_quote_status',
       'possibly_sensitive', 'extended_entities', 'quoted_status_id',
       'quoted_status_id_str', 'quoted_status_permalink', 'quoted_status',
       'retweet_count', 'favorite_count'], axis=1)
    else:
        print(i)
        c.drop(['source','in_reply_to_status_id',
       'in_reply_to_status_id_str', 'in_reply_to_user_id',
       'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'possibly_sensitive', 'extended_entities', 'quoted_status_id',
       'quoted_status_id_str', 'quoted_status_permalink', 'quoted_status', 'contributors', 'is_quote_status','source', 'in_reply_to_status_id', 'in_reply_to_status_id_str',
       'in_reply_to_user_id', 'in_reply_to_user_id_str',
       'in_reply_to_screen_name', 'user', 'contributors', 'is_quote_status',
       'possibly_sensitive', 'extended_entities', 'quoted_status_id',
       'quoted_status_id_str', 'quoted_status_permalink', 'quoted_status','retweet_count', 'favorite_count'], axis=1)
        tweets = pd.concat([tweets, c], ignore_index=True)

# data filtering
# 1. lower tweets
#tweets['full_text'] = tweets['full_text'].map(lambda x: x.lower())
# 2. filter boris johnson (total 14 tweets)
def BORIS(X):
    regex = '(^|\s|\#)(boris johnson)($|\s)'
    return re.sub(regex, " ",X, flags=re.IGNORECASE)
tweets['full_text'] = tweets['full_text'].map(lambda x: BORIS(x))
# filter by vaccine and location in us
def VACCINE(X):
    vaccine = '(^|\s|\#)(vaccination|vaccinations|vaccine|vaccines|immunization|vaccinate|vaccinated|vaccin)($|\s)'
    brands = '(^|\s|\#)(phizer|biontech|moderna|astrzeneca|sinovac|sinopharm|j&j|johnson&johnson|johnson & johnson|johnson|johnsonjohnson|johnsonandjohnson|janssen|sanofi|curevac|sputnik-v|valneva|casinobio)($|\s)'
    regex=f'({vaccine}|{brands})'
    return bool(re.search(regex, X, flags=re.IGNORECASE))
tweets['vaccineincluded'] = tweets['full_text'].map(lambda x: VACCINE(x))
tweets['inus'] = tweets['place'].map(lambda x: x['country_code'] == 'US' if x is not None else False)
usvtweets = tweets[tweets['inus'] & tweets['vaccineincluded']]


usvtweets['monthyear'] = usvtweets['created_at'].map(lambda x: str(x.year)+'/'+str(x.month))
usvtweets['cityname'] = usvtweets['place'].map(lambda x: x['full_name'] if x is not None else False)

usvtweets.groupby('monthyear').count()

usvtweets['cords'] = usvtweets['coordinates'].map(lambda x: x['coordinates'])
usvtweets['lat'] = usvtweets['cords'].map(lambda x: x[1])
usvtweets['long'] = usvtweets['cords'].map(lambda x: x[0])


# save filtered data to pickle
if SAVE:
    usvtweets.to_pickle('reduced_data.pkl')
    
#### Data preprcoessing VADER

# remove web address
def WEB_ADDRESS(X):
    X = re.sub(r'(((https?|http?)://)|(www.?))(?:[-\w./]|(?:%[\da-zA-Z]))+', " ",X)
    return X

usvtweets['full_text'] = usvtweets['full_text'].map(lambda x: WEB_ADDRESS(x))

# save filtered data to pickle
if SAVE:
    usvtweets.to_pickle('reduced_data.pkl')
