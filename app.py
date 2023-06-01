import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import streamlit as st
st.set_page_config(layout="wide")


dataset = pd.read_csv('Datasets/useful_feature.csv')
complete_feature_set = pd.read_csv('Datasets/complete_feature.csv')



scope = 'user-library-read'

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)


def create_playlist_df_that_in_dataset(playlist_id, dataset):
    playlist = pd.DataFrame()

    for ix, i in enumerate(sp.playlist_tracks(playlist_id)['items']):
        playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'track_id'] = i['track']['id']

    playlist = playlist[playlist['track_id'].isin(dataset['track_id'].values)]

    return playlist


def generate_playlist_feature(complete_feature_set, playlist_in_df):
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['track_id'].isin(
        playlist_in_df['track_id'].values)]
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['track_id'].isin(
        playlist_in_df['track_id'].values)]
    complete_feature_set_playlist_final = complete_feature_set_playlist.drop(
        columns="track_id")

    return complete_feature_set_playlist_final.sum(axis=0), complete_feature_set_nonplaylist


def generate_recommendation(dataset, playlist_vector, nonplaylist_features):
    non_playlist_df = dataset[dataset['track_id'].isin(
        nonplaylist_features['track_id'].values)]

    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop(
        'track_id', axis=1).values, playlist_vector.values.reshape(1, -1))[:, 0]

    return non_playlist_df.sort_values('sim', ascending=False)


def get_top_rec(recommend, top):
    res_df = recommend.head(top).copy()
    res_df = res_df['track_id']
    return res_df


def get_recommendation(pID):
    playlist_id = pID

    playlist_in_df = create_playlist_df_that_in_dataset(playlist_id, dataset)

    complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_playlist_feature(
        complete_feature_set, playlist_in_df)

    recommend = generate_recommendation(
        dataset, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)

    res_recommended = get_top_rec(recommend, 6)

    return res_recommended.to_list()


def get_track_name(rec_list):
    track_name_list = []

    for x in rec_list:

        track_name = sp.track(x)['name']
        if track_name == '':
            track_name = 'null'

        track_name_list.append(track_name)
    # print(track_name_list)
    return track_name_list


def get_track_artist(rec_list):
    artist_name_list = []
    for x in rec_list:
        try:
            artist_list = sp.track(x)['artists']
        except:
            artist_list = []
        artist_names = []
        for idx, x in enumerate(artist_list):
            artist_names.append(artist_list[idx]['name'])

        artist_name_list.append(artist_names)

    return artist_name_list


def get_track_image(rec_list):
    track_image = []

    for x in rec_list:
        try:
            url = sp.track(x)['album']['images'][0]['url']
        except:
            url = 'https://img.toolstud.io/240x240/3b5998/fff&text=+640x640+'
        track_image.append(url)

    return track_image


def get_track_url(rec_list):
    track_url = []
    dummy = "https://open.spotify.com/track/"
    for x in rec_list:
        try:
            url = dummy + x
        except:
            url = "https://open.spotify.com/"
        track_url.append(url)

    return track_url


track_name = []
track_artist = []
track_image = []
track_url = []


def getLinkToImage(track_image, track_link):
    markdown_output = "[![Foo](%s)](%s)" % (track_image, track_link)
    return markdown_output


def display():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header(track_name[0])
        # st.image(track_image[0])
        st.markdown(getLinkToImage(track_image[0], track_url[0]))
        st.write(track_artist[0][0])

    with col2:
        st.header(track_name[1])
        # st.image(track_image[1])
        st.markdown(getLinkToImage(track_image[1], track_url[1]))
        st.write(track_artist[1][0])

    with col3:
        st.header(track_name[2])
        # st.image(track_image[2])
        st.markdown(getLinkToImage(track_image[2], track_url[2]))
        st.write(track_artist[2][0])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header(track_name[3])
        # st.image(track_image[3])
        st.markdown(getLinkToImage(track_image[3], track_url[3]))
        st.write(track_artist[3][0])

    with col2:
        st.header(track_name[4])
        # st.image(track_image[4])
        st.markdown(getLinkToImage(track_image[4], track_url[4]))
        st.write(track_artist[4][0])

    with col3:
        st.header(track_name[5])
        # st.image(track_image[5])
        st.markdown(getLinkToImage(track_image[5], track_url[5]))
        st.write(track_artist[5][0])


st.title('MuRec - A playlist based recommendation app')


playlistID = ""
user_query = st.text_input('Playlist ID', '')


if st.button('Recommend') or (playlistID != user_query):
    playlistID = user_query
    recommended_track_id_list = get_recommendation(playlistID)

    track_name = get_track_name(recommended_track_id_list)
    track_artist = get_track_artist(recommended_track_id_list)
    track_image = get_track_image(recommended_track_id_list)
    track_url = get_track_url(recommended_track_id_list)

    display()
