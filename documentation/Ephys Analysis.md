# Pac-Rat: Ephys Analysis

Stages of Ephys processing (DRAFT)

## Pre-processing

These steps produce a "cleaned" binary file with the original (128-channel) numbering of the InTan raw data

1. Re-referencing
    * Operates on raw amplifier data (Amplifier.bin)
    * Compute per-headstage mean for each sample (ignore bad channels)
    * Subtract headstage mean from all of its channels
    * Compute linear scaling per channel (gain and offset compensation) w.r.t. mean reference
    * Store as "Amplifier_cleaned.bin"

2. (Optional) Remove 50 Hz noise with IIR notch filter

## Intermediates

These steps produce a "frame binned" binary file with the remapped (121-channel) channel order based on the probe map

1. Compute MUA using rectified high-pass (> 250 Hz) and bin "per frame"
2. Compute MUA using threshold crossing counts (> 3 STD) and bin "per frame"
3. Compute spectral power in relevant bands (LFP, MUA, etc.) and bin "per frame"
4. Compute LFP phase for each channel w.r.t to its shank (+ is leading, - is lagging) and bin "per frame"
5. Compute some form of correlation between shafts (?)

## Final

These steps produce final analysis figures

1. Plot event related activity (start, touch, and reward) in surrounding window
2. Average signal maps for each trial phase (waiting, searching, seeking, and homing)
