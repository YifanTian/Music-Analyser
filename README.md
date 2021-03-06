# Music-Analyser
Midi Music analysis tools and code collection, including statistical music models.

**Requirements**
Python 2.7
Use pip install -r requirements.txt to install packages.

# Tools
1. **play midi** </br>
Run: </br>
python tools.py midi_file.mid --play </br>
For example: </br>
python tools.py anjing.mid --play </br>

2. **plot midi** </br>
Run: </br>
python tools.py midi_file.mid --play </br>
For example: </br>
python tools.py anjing.mid --play </br>

3. **Basic music analysis** </br>
Extract music inforamtion including Unigrams and Bigrams. </br>
Run: </br>
python midi_analyser.py midi_file.mid </br>
For example: </br>
python midi_analyser.py anjing.mid </br>

4. **Generate midi music based on statistical language models** </br>
Run: </br>
python midi_generator.py midi_file.mid </br>
For example: </br>
python midi_generator.py anjing.mid </br>

I will keep updating this project. Will Do: 
- Add chords module. 
- Determine music tonality based on statistcal data.
- Generate midi based on chords as well as traning midi files.
- Determine music verse and chorus.
- Music clustering based on melody, tonality and emotion.

I am a music fan as well as a machine learning researcher. I would like to explore music creation, especially using machine learning to genereate melody in the future. I would like to hear about any suggestion or comments. I am also looking for collcaboration oppertunities or working oppertunity. Thank you!

Example based on anjing.mid: </br>
[Music link](https://www.youtube.com/watch?v=KV64yCc-0Y4) </br>

**Plot of one band of midi** </br>
![alt text](plot/anjing.mid.png)

**Plot of unigrams** </br>
![alt text](plot/anjing_unigrams.png)

**Plot of bigrams** </br>
![alt text](plot/anjing_bigrams.png)





