# Audio layer separation algorithm

## Description

This is a separation algorithm that exploits the "layering" structure at the beginning of songs. 
The basic idea is that a song will sometimes start in layers (e.g. the drums, then the bass a measure later, then
the guitar playing chords, and then the vocals). I try to leverage that information to do source separation.

## Usage
Do:
    python layer_separation.py <audio_file>
To run the algorithm. The audio file can be an .mp3 or a .wav, whatever librosa can accept.

NOTE: I had to edit the non-negative matrix factorization in scikit-learn to get the algorithm to work, so I've included 
my edited version of the library here.
