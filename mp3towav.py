# http://pydub.com

from pydub import AudioSegment
from pydub.utils import mediainfo
import os

file_path = "./music/doremi.mp3"
file_name = os.path.splitext(file_path)[0]
file_name = str(file_name) + ".wav"

sound = AudioSegment.from_mp3(file_path)
sound.export(file_name, format="wav")

print(file_name)
info = mediainfo(file_name)
print(info['sample_rate'])
