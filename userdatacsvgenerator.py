
import pandas as pd
import requests
from google.colab import files

response = requests.get('https://random-data-api.com/api/v2/users?size=5')

# Creating a DataFrame
df = pd.DataFrame()

if response.status_code == 200:
	for i in range(1, 400):
		response = requests.get('https://api.themoviedb.org/3/\
	movie/top_rated?api_key=aaa7de53dcab3a19afed86880\
	f364e54&language=en-US&page={}'.format(i))
		temp_df = pd.DataFrame(response.json()['results'])[['id',
					'title', 'overview', 'release_date', 'popularity',
					'vote_average', 'vote_count']]
		df = df.append(temp_df, ignore_index=True)
else:
	print('Error', response.status_code)

