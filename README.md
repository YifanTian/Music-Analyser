# Music-Analyser
Midi Music analysis tools and code collection, including statistical music models.

**Requirements**
Python 2.7
Use pip install -r requirements.txt to install packages.


# Tools
1. **play midi**
Run: </br>
python tools.py midi_file.mid --play </br>
For example: </br>
python tools.py anjing.mid --play </br>

2. **plot midi**
Run: </br>
python tools.py midi_file.mid --play </br>
For example: </br>
python tools.py anjing.mid --play </br>

3. **Basic music analysis**
Extract music inforamtion including Unigrams and Bigrams. </br>
Run: </br>
python midi_analyser.py midi_file.mid </br>
For example: </br>
python midi_analyser.py anjing.mid </br>

4. **Generate midi music based on statistical language models**
Run: </br>
python midi_generator.py midi_file.mid </br>
For example: </br>
python midi_generator.py anjing.mid </br>

Example based on anjing.mid: </br>
[Music link](https://www.youtube.com/watch?v=KV64yCc-0Y4) </br>


I will keep updating this project. Will Do: Add chords module. Generate midi based on chords as well as traning midi files.

I am a music fan. I would like to explore music creation, especially using machine learning to genereate music in the future. I would like to hear about any suggestion or comments. I am also looking for collcaboration oppertunities or working oppertunity. Thank you!

**Plot of one band of midi** 
![alt text](plot/anjing.mid.png)

**Plot of unigrams**
![alt text](plot/anjing_unigrams.png)


**Plot of bigrams**
![alt text](anjing_bigrams.png)





