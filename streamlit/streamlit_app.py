import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
from moviepy.editor import ImageSequenceClip


def show_player_image(index, col):
	result = requests.get(f"http://backend:8000/image/{index}")
	if result.status_code != 200:
		st.error(f'There was an error in backend: {result.json()}')
	else:
		result = result.json()
		data = np.asarray(json.loads(result["image"]))
		with col:
			st.image(data, width=128)

def show_interpolation(index1, index2, interp_size = 100, col = None):
		data = {
			"Image1": str(index1),
			"Image2": str(index2),
			"size": str(interp_size)
		}

		result = requests.post(f"http://backend:8000/interpolation", json = data)
		if result.status_code != 200:
			st.error(f'There was an error in backend: {result.json()}')
		else:
			result = result.json()
			data = np.asarray(json.loads(result["images"]))
			
			clip = ImageSequenceClip(list(data * 255), fps=20)
			clip.write_gif('local.gif', fps=20)
			with col:
				st.image("local.gif", width=128)

def show_random_player(col):
	result = requests.get(f"http://backend:8000/random")
	if result.status_code != 200:
		st.error(f'There was an error in backend: {result.json()}')
	else:
		result = result.json()
		data = np.asarray(json.loads(result["image"]))
		with col:
			st.image(data, width=128)

players = pd.read_csv('data.csv')


def main():
	st.title("No puedo, tengo fulbo")
	
	col1, col2, col3 = st.columns(3)
	
	with col1:
		player1_name = st.text_input('Player1', 'L. Messi')
	with col3:
		player2_name = st.text_input('Player2', 'E. Hazard')

	try:
		player1_index = players[players['Name'].str.contains(player1_name)].iloc[0]["Unnamed: 0"]
	except:
		player1_index = 0

	try:
		player2_index = players[players['Name'].str.contains(player2_name)].iloc[0]["Unnamed: 0"]
	except:
		player2_index = 5

	col1, col2, col3 = st.columns(3)
	if st.button('Search players'):
		show_player_image(player1_index, col1)
		show_player_image(player2_index, col3)
	
	if st.button('Interpolate'):
		show_player_image(player1_index, col1)
		show_player_image(player2_index, col3)
		show_interpolation(player1_index, player2_index, col=col2)

	col1, col2, col3 = st.columns(3)
	if st.button('Random Player'):
		show_random_player(col2)


if __name__ == '__main__':
	main()