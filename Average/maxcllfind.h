#pragma once
#include "avisynth.h"
#include <stdint.h>
#include <string>

class MaxCLLFind : public GenericVideoFilter {
public:
    enum MaxFallAlgorithm {
        MAXFALL_NONE = -1,
        MAXFALL_OFFICIAL = 0,
        MAXFALL_ALLCHANNELS = 1
    };

    MaxCLLFind(PClip clip, IScriptEnvironment* env, MaxFallAlgorithm maxFallAlgorithm);
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
    
    // MaxcLL
    unsigned int highestrawvalue = 0;
    unsigned int highestFrame = 0;
    float highestFloatvalue = 0;
    float highestnits = 0;
    unsigned int highestValueX, highestValueY;

    // MinCLL
    unsigned int lowestrawvalue = -1;
    unsigned int lowestFrame = -1;
    float lowestFloatvalue;
    float lowestnits;
    unsigned int lowestValueX, lowestValueY;

    // FALL Average
    long double FALLSum = 0;
    long framesCounted = 0;

    // MaxFALL
    const int maxFallAlgorithm;
    float MaxFALL = 0;
    unsigned int MaxFALLFrame;

    int fileWriteCounter = 0;
    std::string statsFileName = "";
};

