from IPython.display import HTML
from base64 import b64encode

# Path to the video file
video_path = '/content/sample_data/mov_bbb.mp4'

# Function to display video
def display_video(file_path):
    video_file = open(file_path, "rb").read()
    video_url = "data:video/mp4;base64," + b64encode(video_file).decode()
    return HTML(f'<video width="640" height="480" controls><source src="{video_url}" type="video/mp4"></video>')

# Display the video
display_video(video_path)
