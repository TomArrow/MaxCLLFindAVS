#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "maxcllfind.h"
#include "avs\alignment.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <emmintrin.h>
#include <vector>


template<int minimum, int maximum>
static __forceinline int static_clip(float val) {
    if (val > maximum) {
        return maximum;
    }
    if (val < minimum) {
        return minimum;
    }
    return (int)val;
}

struct FloatColor
{
	float R, G, B;
};

int outputValueCount = 32;

const int R = 0;
const int G = 1;
const int B = 2;
const int RCORD = 3;
const int GCORD = 4;
const int BCORD = 5;


struct AverageData
{
	double totalR, totalG, totalB;
	float divisor;
};

struct ColorPairData
{
	uint8_t R, G, B, RCORD, GCORD, BCORD;
	uint8_t nearestQuadrantR, nearestQuadrantG, nearestQuadrantB;
};


// Linear when:
// A grows fastest, b grows second fastest, c grows slowest
// For RGB image that would be: channel, x, y
// thus nesting would be ideal: 
// for y : for x : for channel
inline int map3D(int &a, int &b, int&c, int&sizeA, int&sizeB) {
	return a + sizeA*b + sizeA*sizeB*c;
}

/*
ST2084_CONSTANTS = Structure(
	m_1=2610 / 4096 * (1 / 4),
	m_2=2523 / 4096 * 128,
	c_1=3424 / 4096,
	c_2=2413 / 4096 * 32,
	c_3=2392 / 4096 * 32)
*/
const float ST2084_L_P = 10000;
const float ST2084_CONST_M_1 = 2610.0f / 4096.0f * (1.0f / 4.0f);
const float ST2084_CONST_M_2 = 2523.0f / 4096.0f * 128.0f;
const float ST2084_CONST_C_1 = 3424.0f / 4096.0f;
const float ST2084_CONST_C_2 = 2413.0f / 4096.0f * 32.0f;
const float ST2084_CONST_C_3 = 2392.0f / 4096.0f * 32.0f;

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

inline float spow(float base, float exp) {
	return sgn(base) * pow(abs(base),exp);
}


float m_1_d = 1 / ST2084_CONST_M_1;
float m_2_d = 1 / ST2084_CONST_M_2;

inline float eotf_ST2084(float N) {
	//N = to_domain_1(N)


	float V_p = spow(N, m_2_d);

	float n = V_p - ST2084_CONST_C_1;
	// Limiting negative values.
	n = n < 0 ? 0 : n;
    
	float L = spow((n / (ST2084_CONST_C_2 - ST2084_CONST_C_3 * V_p)), m_1_d);
	float C = ST2084_L_P * L;

	return C;//from_range_1(C)
}

template<MaxFallAlgorithm maxFallAlgorithm, int components_per_pixel>
template<typename pixel_t, int bits_per_pixel>
void MaxCLLFind<maxFallAlgorithm, components_per_pixel>::dofindmaxcll_planar_c(const PVideoFrame src, int thisFrame) {

    const int max_pixel_value = (1 << bits_per_pixel) - 1;
    //static const int planes[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    static const int planes[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };

    int height = src->GetHeight(planes[0]);
    int width = src->GetRowSize(planes[0]);

    for (int p = 1; p < 3; p++) {
        if (height != src->GetHeight(planes[p])){
            //env->ThrowError("MaxCLLFind: all planes must have same sizes");
            return;
        }
        if (width != src->GetRowSize(planes[p])) {
            //env->ThrowError("MaxCLLFind: all planes must have same sizes");
            return;
        }
    }
    width /= sizeof(pixel_t);

    long double CLLSum = 0;
    int CLLvalueCount = 0;
    for (int p = 0; p < 3; ++p) {
        const BYTE* ptr = src->GetReadPtr(planes[p]);
        int pitch = src->GetPitch(planes[p]);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                pixel_t currentvalue = reinterpret_cast<const pixel_t*>(ptr)[x];
                //nits = eotf_ST2084(currentvalueFloat);
                float nits = nitArray[(int)currentvalue];

                if (currentvalue > highestrawvalue) {

                    float currentvalueFloat = (float)currentvalue / (float)max_pixel_value;
                    highestnits = nits;
                    highestrawvalue = currentvalue;
                    highestFloatvalue = currentvalueFloat;
                    highestValueX = x;
                    highestValueY = height - y - 1;
                    highestFrame = thisFrame;
                }
                if (currentvalue < lowestrawvalue) {

                    float currentvalueFloat = (float)currentvalue / (float)max_pixel_value;
                    lowestnits = nits;
                    lowestrawvalue = currentvalue;
                    lowestFloatvalue = currentvalueFloat;
                    lowestValueX = x;
                    lowestValueY = height - y - 1;
                    lowestFrame = thisFrame;
                }
                switch (maxFallAlgorithm) {
                case MAXFALL_NONE:
                    break;
                case MAXFALL_ALLCHANNELS:
                    CLLSum += nits;
                    CLLvalueCount++;
                    break;
                }
            }
            ptr += pitch;
        }
    }
    if (CLLvalueCount > 0) {
        float FALL = CLLSum / (float)CLLvalueCount;
        FALLSum += FALL;
        if (FALL > MaxFALL) {
            MaxFALL = FALL;
            MaxFALLFrame = thisFrame;
        }
    }
}

template<MaxFallAlgorithm maxFallAlgorithm, int components_per_pixel>
template<typename pixel_t, int bits_per_pixel>
void MaxCLLFind<maxFallAlgorithm, components_per_pixel>::dofindmaxcll_packed_c(const PVideoFrame src, int thisFrame) {
    
    const int max_pixel_value = (1 << bits_per_pixel) - 1;
    const BYTE* ptr = src->GetReadPtr();
    int pitch = src->GetPitch();
    int height = src->GetHeight();
    int width = src->GetRowSize();
    width /= sizeof(pixel_t);

    long double CLLSum = 0;
    float channelNits[3];
    int CLLvalueCount = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += components_per_pixel) {
            for (int c = 0; c < 3; c++) {
                pixel_t currentvalue = reinterpret_cast<const pixel_t*>(ptr)[x+c];
                //nits = eotf_ST2084(currentvalueFloat);
                float nits = nitArray[(int)currentvalue];

                if (currentvalue > highestrawvalue) {

                    float currentvalueFloat = (float)currentvalue / (float)max_pixel_value;
                    highestnits = nits;
                    highestrawvalue = currentvalue;
                    highestFloatvalue = currentvalueFloat;
                    highestValueX = x / components_per_pixel;
                    highestValueY = height - y - 1;
                    highestFrame = thisFrame;
                }
                if (currentvalue < lowestrawvalue) {

                    float currentvalueFloat = (float)currentvalue / (float)max_pixel_value;
                    lowestnits = nits;
                    lowestrawvalue = currentvalue;
                    lowestFloatvalue = currentvalueFloat;
                    lowestValueX = x / components_per_pixel;
                    lowestValueY = height - y - 1;
                    lowestFrame = thisFrame;
                }

                switch (maxFallAlgorithm) {
                case MAXFALL_NONE:
                    break;
                case MAXFALL_ALLCHANNELS:
                    CLLSum += nits;
                    CLLvalueCount++;
                    break;
                case MAXFALL_OFFICIAL:
                    channelNits[c] = nits;
                }
            }

            if (maxFallAlgorithm == MAXFALL_OFFICIAL) {
                // we passed through R, G and B and populated channelNits, so we can now calculate their max and use it for CLLSum which in turn gets used for MaxFALL. 
                float maxChannelNits = std::max(channelNits[0], channelNits[1]);
                maxChannelNits = std::max(maxChannelNits, channelNits[2]);
                CLLSum += maxChannelNits;
                CLLvalueCount++;
            }
        }
        ptr += pitch;
    }

    if (CLLvalueCount > 0) {
        float FALL = CLLSum / (float)CLLvalueCount;
        FALLSum += FALL;
        if (FALL > MaxFALL) {
            MaxFALL = FALL;
            MaxFALLFrame = thisFrame;
        }
    }
}

template<MaxFallAlgorithm maxFallAlgorithm, int components_per_pixel>
MaxCLLFind<maxFallAlgorithm, components_per_pixel>::MaxCLLFind(PClip clip, IScriptEnvironment* env, float* nitArray)
    : GenericVideoFilter(clip)
    , nitArray(nitArray)
    , highestrawvalue(0)
    , highestFrame(0)
    , highestFloatvalue(0)
    , highestnits(0)
    , highestValueX(0)
    , highestValueY(0)
    , lowestrawvalue(-1)
    , lowestFrame(-1)
    , lowestFloatvalue(0)
    , lowestnits(0)
    , lowestValueX(0)
    , lowestValueY(0)
    , FALLSum(0)
    , framesCounted(0)
    , MaxFALL(0)
    , MaxFALLFrame(0)
    , fileWriteCounter(0)
    , statsFileName("") {

    int pixelsize = vi.ComponentSize();
    int bits_per_pixel = vi.BitsPerComponent();
    int planar = vi.IsPlanar();

    //bool avx = !!(env->GetCPUFlags() & CPUF_AVX);
    // we don't know the alignment here. avisynth+: 32 bytes, classic: 16
    // decide later (processor_, processor_32aligned)

    /*if (env->GetCPUFlags() & CPUF_SSE2) {
      bool use_weighted_average_f = false;
      if (pixelsize == 1) {
        if (frames_count == 2)
          processor_ = &weighted_average_int_sse2<2>;
        else if (frames_count == 3)
          processor_ = &weighted_average_int_sse2<3>;
        else
          processor_ = &weighted_average_int_sse2<0>;
        processor_32aligned_ = processor_;
        for (const auto& clip : clips) {
          if (std::abs(clip.weight) > 1) {
            use_weighted_average_f = true;
            break;
          }
        }
        if (clips.size() > 255) {
          // too many clips, may overflow
          use_weighted_average_f = true;
        }
      }
      else {
        // uint16 and float: float mode internally
        use_weighted_average_f = true;
      }

      if (use_weighted_average_f) {
        switch(bits_per_pixel) {
        case 8:
          processor_ = &weighted_average_sse2<uint8_t, 8, false>;
          processor_32aligned_ = avx ? &weighted_average_avx<uint8_t, 8> : &weighted_average_sse2<uint8_t, 8, false>;
          break;
        case 10:
          processor_ = &weighted_average_sse2<uint16_t, 10, false>;
          processor_32aligned_ = avx ? &weighted_average_avx<uint16_t, 10> : &weighted_average_sse2<uint16_t, 10, false>;
          break;
        case 12:
          processor_ = &weighted_average_sse2<uint16_t, 12, false>;
          processor_32aligned_ = avx ? &weighted_average_avx<uint16_t, 12> : &weighted_average_sse2<uint16_t, 12, false>;
          break;
        case 14:
          processor_ = &weighted_average_sse2<uint16_t, 14, false>;
          processor_32aligned_ = avx ? &weighted_average_avx<uint16_t, 14> : &weighted_average_sse2<uint16_t, 14, false>;
          break;
        case 16:
          if(env->GetCPUFlags() & CPUF_SSE4_1)
            processor_ = &weighted_average_sse2<uint16_t, 16, true>;
          else
            processor_ = &weighted_average_sse2<uint16_t, 16, false>;
          processor_32aligned_ = avx ? &weighted_average_avx<uint16_t, 16> : processor_;
          break;
        case 32:
          processor_ = &weighted_average_f_sse2;
          processor_32aligned_ = avx ? &weighted_average_f_avx : &weighted_average_f_sse2;
          break;
        }
      }
    }
    else {*/
    if (planar) {
        if (maxFallAlgorithm == MAXFALL_OFFICIAL) {
            env->ThrowError("MaxCLLFind: official maxFall algorithm not supported for planar formats. Use a packed format like RGB48, or use another maxFall algorithm.");
        }
    }
    switch (bits_per_pixel) {
    case 8:
        if (planar) processor_ = &MaxCLLFind::dofindmaxcll_planar_c<uint8_t, 8>;
        else processor_ = &MaxCLLFind::dofindmaxcll_packed_c<uint8_t, 8>;
        break;
    case 10:
        processor_ = &MaxCLLFind::dofindmaxcll_planar_c<uint16_t, 10>;
        break;
    case 12:
        processor_ = &MaxCLLFind::dofindmaxcll_planar_c<uint16_t, 12>;
        break;
    case 14:
        processor_ = &MaxCLLFind::dofindmaxcll_planar_c<uint16_t, 14>;
        break;
    case 16:
        if (planar) processor_ = &MaxCLLFind::dofindmaxcll_planar_c<uint16_t, 16>;
        else processor_ = &MaxCLLFind::dofindmaxcll_packed_c<uint16_t, 16>;
        break;
    }
    /*processor_32aligned_ = processor_;
  }*/
}

template<MaxFallAlgorithm maxFallAlgorithm, int components_per_pixel>
void MaxCLLFind<maxFallAlgorithm, components_per_pixel>::writeCLLStats() {

	if (statsFileName == "") {
		FILE* test;
		int counter = 0;
		while (test = fopen((statsFileName = "MaxCLLFind_Results" + std::to_string(counter) + ".txt").c_str() , "r")) {
			fclose(test);
			counter++;
		}
		//statsFileName = "MaxCLLFind_Results" + std::to_string(counter) + ".txt";
	}

	float FALLAverage = FALLSum / (float)framesCounted;

	std::ofstream myfile;
	myfile.open(statsFileName, std::ios::out | std::ios::app);
	myfile << "Stats at frame " << framesCounted << ":\n";
	myfile << "MaxCLL: " << highestnits << ", raw value: " << highestrawvalue << " " << highestFloatvalue << " at X " << highestValueX << " Y " << highestValueY << " at frame " << highestFrame << /*" byte depth: " << sizeof(pixel_t) <<*/ "\n";
	myfile << "MinCLL: " << lowestnits << ", raw value: " << lowestrawvalue << " " << lowestFloatvalue << " at X " << lowestValueX << " Y " << lowestValueY << " at frame " << lowestFrame << /*" byte depth: " << sizeof(pixel_t) <<*/ "\n";
    if (maxFallAlgorithm != MAXFALL_NONE) {
        myfile << "MaxFALL: " << MaxFALL << " at frame " << MaxFALLFrame << "\n";
        myfile << "FALL Average: " << FALLAverage << " across " << framesCounted << " frames.\n";
    }
    //myfile << "Dims: " << width << "x" << height;
    myfile << "\n";
	myfile.close();
}

template<MaxFallAlgorithm maxFallAlgorithm, int components_per_pixel>
MaxCLLFind<maxFallAlgorithm, components_per_pixel>::~MaxCLLFind() {

	delete[] nitArray;
	writeCLLStats();
}

template<MaxFallAlgorithm maxFallAlgorithm, int components_per_pixel>
PVideoFrame MaxCLLFind<maxFallAlgorithm, components_per_pixel>::GetFrame(int n, IScriptEnvironment *env) {
    PVideoFrame src = child->GetFrame(n, env);

    /*int planes_y[4] = {PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A};
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) ? planes_r : planes_y;
    
    bool hasAlpha = vi.IsPlanarRGBA() || vi.IsYUVA();

    for (int pid = 0; pid < (vi.IsY() ? 1 : (hasAlpha ? 4 : 3)); pid++) {
        int plane = planes[pid];
        int width = src->GetRowSize(plane);
        int height = src->GetHeight(plane);
        const BYTE* ptr = src->GetReadPtr(plane);
        int pitch = src->GetPitch(plane);

        bool allSrc32aligned = true;
        if (!IsPtrAligned(ptr, 32))
            allSrc32aligned = false;
        if (pitch & 0x1F)
            allSrc32aligned = false;

        if (allSrc32aligned) {
            (*this.*processor_32aligned_)(ptr, pitch, width, height, n);
        }
        else {*/
            (*this.*processor_)(src, n);
        /* }
    }*/
	if (framesCounted++ % 100 == 0) {
		writeCLLStats();
	}

    return src;
}

AVSValue __cdecl create_maxcllfind(AVSValue args, void* user_data, IScriptEnvironment* env) {

    // Algorithms for MaxFALL frame averaging (default 0)
    //-1 = No maxFALL calculation (is slightly faster)
    // 0 = SMPTE2084 recommendation (average of highest channels of all pixels)
    // 1 = Average of all channels of all pixels
    MaxFallAlgorithm maxFallAlgorithm = (MaxFallAlgorithm)args[1].AsInt(MaxFallAlgorithm::MAXFALL_OFFICIAL);

    auto clip = args[0].AsClip();
    auto vi = clip->GetVideoInfo();

    if (!vi.IsRGB() && !vi.IsPlanarRGB() && !vi.IsPlanarRGBA()) {
        env->ThrowError("MaxCLLFind: clip MUST be packed or planar RGB. Use ConvertToRGB48 for example.");
    }

    int possibleValues = 1 << vi.BitsPerComponent(); //pow(2, 16)
    float* nitArray = new float[possibleValues]; // 16 bit value array for nit values corresponding to 16 bit RGB values

    for (int i = 0; i < possibleValues; i++) {
        nitArray[i] = eotf_ST2084((float)i / ((float)possibleValues - 1));
    }

    bool hasAlpha = vi.IsRGB64() || vi.IsRGB32() || vi.IsPlanarRGBA();
    if (maxFallAlgorithm == MAXFALL_NONE && !hasAlpha) {
        return new MaxCLLFind<MAXFALL_NONE, 3>(clip, env, nitArray);
    }
    if (maxFallAlgorithm == MAXFALL_NONE && hasAlpha) {
        return new MaxCLLFind<MAXFALL_NONE, 4>(clip, env, nitArray);
    }

    if (maxFallAlgorithm == MAXFALL_OFFICIAL && !hasAlpha) {
        return new MaxCLLFind<MAXFALL_OFFICIAL, 3>(clip, env, nitArray);
    }
    if (maxFallAlgorithm == MAXFALL_OFFICIAL && hasAlpha) {
        return new MaxCLLFind<MAXFALL_OFFICIAL, 4>(clip, env, nitArray);
    }

    if (maxFallAlgorithm == MAXFALL_ALLCHANNELS && !hasAlpha) {
        return new MaxCLLFind<MAXFALL_ALLCHANNELS, 3>(clip, env, nitArray);
    }
    if (maxFallAlgorithm == MAXFALL_ALLCHANNELS && hasAlpha) {
        return new MaxCLLFind<MAXFALL_ALLCHANNELS, 4>(clip, env, nitArray);
    }
}

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
	// Arguments: 
	// 0 - clip to be regraded
	// 1 - reference clip
	// 2 - testclip (for example a downsized version of the clip to be regraded. this one will be actually used for the regrading and comparing to reference.
	// 3+ - additional reference clips (not supported atm, will be ignored)
    env->AddFunction("maxcllfind", "c[maxFallAlgorithm]i", create_maxcllfind, 0);
    return "Mind your sugar level";
}