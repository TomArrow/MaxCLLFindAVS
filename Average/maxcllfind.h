#pragma once
#include "avisynth.h"
#include <stdint.h>
#include <string>

enum MaxFallAlgorithm {
    MAXFALL_NONE = -1,
    MAXFALL_OFFICIAL = 0,
    MAXFALL_ALLCHANNELS = 1
};

template<MaxFallAlgorithm maxFallAlgorithm, int components_per_pixel>
class MaxCLLFind : public GenericVideoFilter {
public:

    MaxCLLFind(PClip clip, IScriptEnvironment* env, float* nitArray);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    ~MaxCLLFind();
    
    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

private:
    void writeCLLStats();
    template<typename pixel_t, int bits_per_pixel>
    void dofindmaxcll_c(const PVideoFrame src, int thisFrame);

    decltype(&dofindmaxcll_c<uint8_t, 8>) processor_;
    float const* const nitArray;

    // MaxcLL
    unsigned int highestrawvalue;
    unsigned int highestFrame;
    float highestFloatvalue;
    float highestnits;
    unsigned int highestValueX, highestValueY;

    // MinCLL
    unsigned int lowestrawvalue;
    unsigned int lowestFrame;
    float lowestFloatvalue;
    float lowestnits;
    unsigned int lowestValueX, lowestValueY;

    // FALL Average
    long double FALLSum;
    long framesCounted;

    // MaxFALL
    float MaxFALL;
    unsigned int MaxFALLFrame;

    int fileWriteCounter;
    std::string statsFileName;
};

