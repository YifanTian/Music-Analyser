
import argparse
from util import compose
from util import play_midi
from util import plot_midi


def play(midi_name):
    print("play: ",midi_name)
    play_midi(midi_name)
    return

def plot(midi_name):
    print("plot: ",midi_name)
    plot_midi(midi_name,show=True,save=False)
    return

# def create(melody):
#     print("compose: ",melody)
#     song_name = "new"
#     naive_song = compose(melody, song_name,'./')
#     return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some midi.')
    parser.add_argument('midi_name', type=str, help='midi name')
    parser.add_argument('--plot', dest='op', action='store_const',
                        const=plot, help='plot operation on midi')
    parser.add_argument('--play', dest='op', action='store_const',
                        const=play, help='play operation on midi')
    # parser.add_argument('--compose', dest='op', action='store_const',
    #                     const=create, help='melody for creation')

    args = parser.parse_args()
    args.op(args.midi_name)
