import midi
import matplotlib.pyplot as plt
import argparse


def plot_midi(midi_file,show=False,save=True,save_dir='./midi_plots/'):
    """ plot midi file """
    song = midi.read_midifile(midi_file)
    song.make_ticks_abs()
    tracks = []
    for track in song:
        notes = [note for note in track if note.name == 'Note On']
        pitch = [note.pitch for note in notes]
        tick = [note.tick for note in notes]
        tracks += [tick, pitch]

    plt.plot(*tracks)
    if show:
        plt.show()
    if save:
        # plt.savefig(save_dir+midi_file+'.png')
        plt.savefig(midi_file+'.png')
    return

def play_midi(music_file):
    """
    stream music with mixer.music module in blocking manner
    this will stream the sound from disk while playing
    """
    import pygame
    import base64

    # # convert back to a binary midi and save to a file in the working directory
    # fish = base64.b64decode(mid64)
    # fout = open(music_file,"wb")
    # fout.write(fish)
    # fout.close()
    freq = 44100    # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2    # 1 is mono, 2 is stereo
    buffer = 1024    # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)
    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(0.8)

    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print "Music file %s loaded!" % music_file
    except pygame.error:
        print "File %s not found! (%s)" % (music_file, pygame.get_error())
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)

def read_pitchs(midi_file):
    """ read pitchs from midi file """
    song = midi.read_midifile(midi_file)
    song.make_ticks_abs()
    tracks = []
    for track in song:
        notes = [note for note in track if note.name == 'Note On']
        pitch = [note.pitch for note in notes]
        tick = [note.tick for note in notes]
        tracks += [tick, pitch]

    # start = 0
    # song_length = 20
    # ticks = tracks[2][start:start+song_length]
    # pitchs = tracks[3][start:start+song_length]
    ticks = tracks[2]
    pitchs = tracks[3]
    return pitchs


def compose(pitchs, song_name, save_dir):
    start = 0
    song_length = 30
    # ticks = tracks[2][start:start+song_length]
    pitchs = pitchs[start:start+song_length]

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    # start = ticks[0]
    for i in range(len(pitchs)):
        new_tick = 200
        # print(ticks[i], start, new_tick)
        # on = midi.NoteOnEvent(tick=new_tick, velocity=20, pitch=midi.C_6)
        # on = midi.NoteOnEvent(tick=new_tick, velocity=20, pitch=eval('midi.'+str(pitchs[i])))
        on = midi.NoteOnEvent(tick=new_tick, velocity=20, pitch=pitchs[i])
        track.append(on)
    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    midi.write_midifile(song_name, pattern)
    return song_name

if __name__ == '__main__':
    midi_file = "./collection/zhou/shanhuhai1.mid"
    # plot_midi(midi_file)
    pitchs = read_pitchs(midi_file)
    print(pitchs)
    song_name = "compose_test.mid"
    naive_song = compose(pitchs, song_name,'./')
    plot_midi(song_name,show=True,save=False)
    play_midi(naive_song)

