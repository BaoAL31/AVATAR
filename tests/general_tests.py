import pickle
with open('/home/jembo/AVATAR/data/processed/-FaXLcSFjUI_trimmed/cache/tracks.pkl', 'rb') as f:
    tracks = pickle.load(f)

# print first few tracks start/end times
print(type(tracks[0]))
print(tracks[0].keys())
print(tracks[0]['track'].keys())
for i, track in enumerate(tracks[:5]):
    frames = track['track']['frame']
    print(f"Track {i}: frame {frames[0]} to {frames[-1]} = {frames[0]/25:.2f}s to {frames[-1]/25:.2f}s")