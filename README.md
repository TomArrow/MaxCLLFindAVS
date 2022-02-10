## MaxCLLFind ##

PQ HDR Analyzer plugin for AVISynth, analyzes MaxCLL and MaxFALL and writes that to a text file after closing the application that is calling AVISynth.

The created textfile's name is "MaxCLLFind_Results0.txt". It will be overwritten if it already exists.

### Usage
```
clip.MaxCLLFind()
```
Load in VirtualDub and click Play. After video is finished playing, close VirtualDub. The plugin also writes the Average FALL (frame average light level) into the text file. If you want this result to be accurate, make sure to not load any frame more than once.

This plugin only accepts RGB inputs. If your HDR clip isn't RGB, convert it first. 

For example, let's say you are loading a HDR HEVC YUV file, do this:
```
clip = clip.ConvertToRGB64(matrix="Rec2020")
clip.MaxCLLFind()
```

Supported are the packed RGB formats RGB24, RGB32, RGB48, RGB64 and the planar RGB formats RGBP8, RGBP10, RGBP12, RGBP14, RGBP16. It is not advisable to use the formats RGB24, RGB32 and RGBP8 for HDR clips. The planar formats are processed about 30% faster than the packed formats, however the default MaxFALL calculation is unsuppored for them.

### Alternate MaxFALL algorithm
The default MaxFALL algorithm uses the SMPTE recommendation of averaging max(R,G,B) across all pixels, meaning the brightest channel of each pixel goes into the average. If you want the average of all channels of all pixels (not the official recommendation) instead, do this:
```
clip.MaxCLLFind(maxFallAlgorithm=1)
```
This is more for your own curiosity and might lead to playback problems like flickering if used as actual HDR metadata, since it typically leads to typically slightly lower average intensity readings and if the TV bases its own dimming on the official recommendation, it might dim the image when it reaches a higher FALL than your calculated MaxFALL, which will almost certainly happen. 

The official MaxFALL algorithm is not supported by the planar formats. If your input is planar either disable the MaxFALL calculation or use the unofficial algorithm or convert your clip to a packed format like RGB48. The MaxFALL calculation is disabled by:

```
clip.MaxCLLFind(maxFallAlgorithm=-1)
```

### Word of caution

This filter is really badly written. Since I'm a C++ noob, I took the Average filter and ditched everything I didn't need and commented out some other things, so there are lots of remnants of the Average filter still in this code. It will have to be cleaned up eventually.

It might also have unexpected bugs and glitches and I cannot guarantee the correctness of the results, since I'm not a colour scientist.

The RGB value to nits conversion algorithm was lifted from the colour-science package of python.

### Contribute

If you feel like improving the code, refactoring or cleaning up, feel free. I might do it someday myself or I might not, I don't know.


## TODO

- Add functionality to import cutlist for dynamic scene-based metadata. I'm not sure how this would be implemented in an encode, but I read that this possibility exists, so it would be nice to have.

## Changelog

*2019-12-30 - Fixed MaxFALL Algorithm to be based on the official SMPTE recommendation. This algorithm computes the frame average brightness based on the average of the brightest channel of each pixel, or in other words, max(R,G,B). The old algorithm was using the average of all channels of all pixels, leading to slightly lower resulting values in the tests I did. The old algorithm can be still optionally used via the parameter maxFallAlgorithm=1*