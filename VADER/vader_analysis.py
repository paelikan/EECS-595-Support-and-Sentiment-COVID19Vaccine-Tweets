from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import geopandas as gpd
import pandas as pd
import emoji
import contractions
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np

gpd.options.use_pygeos = False
# color mapping
cmap = ListedColormap(["red", "gold", "green"])

# load data
usvtweets = pd.read_pickle("reduced_data.pkl")
ground_truth = pd.read_csv('tweetstoscore.tsv', sep='\t', header=0, encoding='utf-16')
vaccination_data = pd.read_csv('us_state_vaccinations.csv') # from https://ourworldindata.org/us-states-vaccinations

# change encoding to uft-8
usvtweets["full_text"] = usvtweets["full_text"].str.encode("utf-8").str.decode("utf-8")
#usvtweets['reduced_text'].encode("utf-8").decode("utf-8")
ground_truth["text"] = ground_truth["text"].str.encode("utf-8").str.decode("utf-8")
print('statistics of groundtruth data', '\npos: ', sum(ground_truth['label'] == 1)/200, '\nneu: ', sum(ground_truth['label'] == 0)/200, '\nneg: ', sum(ground_truth['label'] == -1)/200)


# remove contractions (may've --> may have, didn't --> did not)
def remove_contractions(X):
    filtered_words = []
    for w in X.split():
        filtered_words.append(contractions.fix(w))
    return ' '.join(filtered_words)

usvtweets['reduced_text'] = usvtweets['full_text'].map(lambda x: remove_contractions(x))

def remove_smileys(X):
    filtered_words = []
    for w in X.split():
        filtered_words.append(emoji.demojize(w))
    return ' '.join(filtered_words)

usvtweets['reduced_text'] = usvtweets['reduced_text'].map(lambda x: remove_smileys(x))


ground_truth = pd.merge(usvtweets, ground_truth, how='inner', on=['id'])


sid_obj = SentimentIntensityAnalyzer()

predictions = [sid_obj.polarity_scores(tweet) for tweet in ground_truth['full_text']]
pred_labels = []
for prediction in predictions:
    value = prediction['compound']
    if value > 0.05:
        pred_labels.append(1)
    elif value < -0.05:
        pred_labels.append(-1)
    else:
        pred_labels.append(0)

print(f'F1 score: {f1_score(ground_truth["label"], pred_labels, average=None)} (pos / neu / neg)')
print(f'Mean F1 score: {np.mean(f1_score(ground_truth["label"], pred_labels, average=None))}')
print('statistics of prediction', '\npos: ', sum(np.array(pred_labels)==1)/200, '\nneu: ', sum(np.array(pred_labels)==0)/200, '\nneg: ', sum(np.array(pred_labels)==-1)/200)


predictions = [sid_obj.polarity_scores(tweet) for tweet in usvtweets['full_text']]
pred_labels = []
for prediction in predictions:
    value = prediction['compound']
    if value > 0.05:
        pred_labels.append(1)
    elif value < -0.05:
        pred_labels.append(-1)
    else:
        pred_labels.append(0)

usvtweets['label'] = pred_labels
usvtweets['monthyear'] = usvtweets['monthyear'].astype('datetime64[ns]')
# plot sentiment over time
#months = ['2020/3', '2020/4', '2020/5', '2020/6', '2020/7', '2020/8', '2020/9', '2020/10', '2020/11', '2020/12', '2021/1', '2021/2', '2021/3', '2021/4', '2021/5', '2021/6', '2021/7', '2021/8', '2021/9', '2021/10', '2021/11']
#pos = [sum(usvtweets[usvtweets['monthyear']==month]['label'] == 1) / sum(usvtweets['monthyear']==month) for month in months]
pos = usvtweets[usvtweets['label']==1].groupby('monthyear').count()['label']/usvtweets.groupby('monthyear').count()['id']
pos = pos.rename('positive')
#neu = [sum(usvtweets[usvtweets['monthyear']==month]['label'] == 0)  / sum(usvtweets['monthyear']==month) for month in months]
neu = usvtweets[usvtweets['label']==0].groupby('monthyear').count()['label']/usvtweets.groupby('monthyear').count()['id']
neu = neu.rename('neutral')
#neg = [sum(usvtweets[usvtweets['monthyear']==month]['label'] == -1) / sum(usvtweets['monthyear']==month)  for month in months]
neg = usvtweets[usvtweets['label']==-1].groupby('monthyear').count()['label']/usvtweets.groupby('monthyear').count()['id']
neg = neg.rename('negative')

results = pd.concat([neg, neu, pos], axis=1)


total_num = usvtweets.groupby('monthyear').count()['id']
total_num = total_num.cumsum()/total_num.sum()
total_num = total_num.rename('number of tweets')

print('statistics of predicted labels', '\npos: ', sum(usvtweets['label'] == 1)/len(usvtweets), '\nneu: ', sum(usvtweets['label'] == 0)/len(usvtweets), '\nneg: ', sum(usvtweets['label'] == -1)/len(usvtweets))


fig, ax = plt.subplots(figsize=(6.4, 4.8))
results.plot(cmap=cmap, ax=ax)
total_num.plot(ax=ax, color='black')
ax.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Percent Sentiment', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
#plt.savefig('images/SentimentOverTime.png')
plt.show()


### bar plot for months
neg_month = usvtweets[usvtweets['label'] == -1].groupby('monthyear').count()['label'].rename('negative')
neu_month = usvtweets[usvtweets['label'] == 0].groupby('monthyear').count()['label'].rename('neutral')
pos_month = usvtweets[usvtweets['label'] == 1].groupby('monthyear').count()['label'].rename('positive')
results_month = pd.concat([neg_month, neu_month, pos_month], axis=1)

def bar_plot(month, save=False):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    results_month.loc[month].plot.bar(color=['red', 'gold', 'green'],rot=0, ax=ax)
    plt.xlabel('Sentiment class', fontsize=16)
    plt.ylabel('Number of Tweets', fontsize=16)
    if month == '2020-11-01':
        plt.yticks(np.arange(0, 80, step=10))
    elif month == '2021-05-01':
        plt.yticks(np.arange(0, 1750 + 250, step=250))
    elif month == '2021-08-01':
        plt.yticks(np.arange(0, 400, step=50))
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(f'images/barplot_sentiment_{month}.png')
    plt.show()
bar_plot('2020-11-01')
bar_plot('2021-05-01')
bar_plot('2021-08-01')

### geoplot
def map_plot(data, month, save=False):
    geovax = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.long, data.lat))
    fig, ax = plt.subplots()
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world[world.name == 'United States of America'].plot(color='white', edgecolor='black', ax=ax)
    # We can now plot our ``GeoDataFrame``.
    geovax[(data['monthyear']==month) & (data['label'] == -1)].plot(ax=ax, color='red', label='negative', markersize=2)
    geovax[(data['monthyear']==month) & (data['label'] == 0)].plot(ax=ax, color='gold', label='neutral',  markersize=2)
    geovax[(data['monthyear']==month) & (data['label'] == 1)].plot(ax=ax, color='green', label='positve',  markersize=2)
    plt.xlabel('Latitude [Degree]', fontsize=16)
    plt.ylabel('Longitude [Degree]', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14, markerscale=4)
    plt.tight_layout()
    if save:
        plt.savefig(f'images/map_sentiment_{month}.png')
    plt.show()

#map_plot(usvtweets, "2020-11-01")
map_plot(usvtweets, "2021-05-01")

### state plot
usa = gpd.read_file('./maps/states_21basic/states.shp')
geovax = gpd.GeoDataFrame(usvtweets, geometry=gpd.points_from_xy(usvtweets.long, usvtweets.lat))
def state_polygons(y):
    if type(y) != Polygon:
        return [x for x in y.geoms]
    return [y]

usa['polys'] = usa.geometry.map(lambda x: state_polygons(x))
geovax[usa.columns] = None
for i, row in geovax.iterrows():
    point = Point(row['long'], row['lat'])
    if len(usa[usa.apply(lambda x: point.within(x['geometry']), axis=1)])!=0:
        geovax.loc[i, usa.columns] = usa[usa.apply(lambda x: point.within(x['geometry']), axis=1)].reset_index(drop=True).iloc[0]
    else:
        geovax.loc[i, usa.columns] = usa.iloc[usa.polys.map(lambda x: min([geo.exterior.distance(point) for geo in x])).argmin()]
def regionalTweets(geovax, region):
    regiontweets = geovax[geovax['SUB_REGION']==region]
    d1 = regiontweets[regiontweets['label']==-1].groupby('monthyear').count()['id']/regiontweets.groupby('monthyear').count()['id']
    d2 = regiontweets[regiontweets['label']==0].groupby('monthyear').count()['id']/regiontweets.groupby('monthyear').count()['id']
    d3 = regiontweets[regiontweets['label']==1].groupby('monthyear').count()['id']/regiontweets.groupby('monthyear').count()['id']

    # fill missing dates with NaN
    if len(d1) != len(geovax['monthyear'].unique()):
        d1 = pd.concat([geovax.groupby('monthyear').count()['id'], d1], axis=1)
        d1.columns=['ID', 'id']
        d1 = d1['id']
    if len(d2) != len(geovax['monthyear'].unique()):
        d2 = pd.concat([geovax.groupby('monthyear').count()['id'], d2], axis=1)
        d2.columns=['ID', 'id']
        d2 = d2['id']
    if len(d3) != len(geovax['monthyear'].unique()):
        d3 = pd.concat([geovax.groupby('monthyear').count()['id'], d3], axis=1)
        d3.columns=['ID', 'id']
        d3 = d3['id']

    d1 = d1.reset_index().rename({'id':'negative'}, axis=1)[['monthyear', 'negative']]
    d2 = d2.reset_index().rename({'id':'neutral'}, axis=1)[['monthyear', 'neutral']]
    d3 = d3.reset_index().rename({'id':'positive'}, axis=1)[['monthyear', 'positive']]

    d1 = d1.merge(d2, on='monthyear', how='inner').merge(d3, on='monthyear', how='inner').fillna(0)
    return d1

regions = {x:regionalTweets(geovax, x) for x in geovax['SUB_REGION'].unique()}
print({x:regions[x]['negative'].mean() for x in regions.keys()})
print({x:regions[x]['neutral'].mean() for x in regions.keys()})
print({x:regions[x]['positive'].mean() for x in regions.keys()})


#states = gpd.read_file('geopandas-tutorial/data/usa-states-census-2014.shp') # https://github.com/joncutrer/geopandas-tutorial/blob/master/data/usa-states-census-2014.shx


# vaccination rate analysis

# remove non us states
def filter_rows_by_values(X, col, values):
    return X[~X[col].isin(values)]

non_usstates = ['American Samoa', 'Bureau of Prisons', 'Dept of Defense', 'Federated States of Micronesia', 'Guam', 'Indian Health Svc',
 'Long Term Care', 'Marshall Islands', 'Northern Mariana Islands', 'Republic of Palau', 'Veterans Health', 'Virgin Islands']
vaccination_data = filter_rows_by_values(vaccination_data, 'location', non_usstates)
# set New York state label equal to usa geopandas labels
vaccination_data['location'][vaccination_data['location'] == 'New York State'] = "New York"

end_of_month = ['2021-01-31', '2021-02-28', '2021-03-31', '2021-04-30', '2021-05-31', '2021-06-30', '2021-07-31',
                '2021-08-31', '2021-09-30', '2021-10-31', '2021-11-30']
def increase_of_month(data, month, end_of_month, state):
    if int(month) != 0:
        prev_month = month - 1
        return float(vaccination_data[(vaccination_data['date']==end_of_month[month]) & (vaccination_data['location']==state)]['people_fully_vaccinated'].values-vaccination_data[(vaccination_data['date']==end_of_month[prev_month]) & (vaccination_data['location']==state)]['people_fully_vaccinated'].values)
    elif int(month) == 0:
        return float(data[(data['date']==end_of_month[month]) & (data['location']==state)]['people_fully_vaccinated'].values)
increase_of_month(vaccination_data, 0, end_of_month, 'United States')

vaccination_data[usa.columns] = np.nan
for i, row in usa.iterrows():
    for j, row2 in vaccination_data[vaccination_data['location'] == str(row['STATE_NAME'])].iterrows():
        vaccination_data.loc[j, usa.columns] = row

def regionalVaccination(geovax, region, end_of_month):
    regionvaccination = geovax[geovax['SUB_REGION']==region]
    d1 = regionvaccination[regionvaccination['date'].isin(end_of_month)].fillna(method='ffill').groupby('date').mean()['people_vaccinated_per_hundred']/100
    d1 = d1.reset_index().rename({'date': 'monthyear', 'people_vaccinated_per_hundred': 'vaccination rate'}, axis=1)[['monthyear', 'vaccination rate']]
    d1['monthyear'] = d1['monthyear'].astype('datetime64[ns]')
    return d1

regions_vax = {x:regionalVaccination(vaccination_data, x, end_of_month) for x in vaccination_data['SUB_REGION'].unique()}

# regional plots
for x in regions.keys():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    regions[x].groupby('monthyear').mean().plot(cmap=cmap, ax=ax)
    regions_vax[x].groupby('monthyear').mean().plot(ax=ax, color='black')
    #ax.set_title(f'Distribution of Vaccine Sentiment in the {x} Region')
    ax.set_ylabel('Percent Sentiment', fontsize=16)
    ax.set_xlabel('Time', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    #plt.savefig(f'images/Sentiment_{x}.png')
    plt.show()

