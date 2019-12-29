## MaxCLLFind ##

PQ HDR Analyzer plugin for AVISynth, analyzes MaxCLL and MaxFALL and writes that to a text file after closing the application that is calling AVISynth.

The created textfile's name is "MaxCLLFind_Results0.txt". It will be overwritten if it already exists.

### Usage
```
clip.MaxCLLFind()
```
Load in VirtualDub and click Play. After video is finished playing, close VirtualDub. The plugin also writes the Average FALL (frame average light level) into the text file. If you want this result to be accurate, make sure to not load any frame more than once.

If your HDR clip isn't RGB64, convert it first. This plugin only accepts RGB64 input. 

For example, let's say you are loading a HDR HEVC YUV file, do this:
```
clip = clip.ConvertToRGB64(matrix="Rec2020")
clip.MaxCLLFind()
```

### Word of caution

This filter is really badly written. Since I'm a C++ noob, I took the Average filter and ditched everything I didn't need and commented out some other things, so there are lots of remnants of the Average filter still in this code. It will have to be cleaned up eventually.

It might also have unexpected bugs and glitches and I cannot guarantee the correctness of the results, since I'm not a colour scientist.

The RGB value to nits conversion algorithm was lifted from the colour-science package of python.

### Contribute

If you feel like improving the code, refactoring or cleaning up, feel free. I might do it someday myself or I might not, I don't know.
