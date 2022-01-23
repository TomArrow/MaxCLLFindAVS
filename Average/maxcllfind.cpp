#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "avisynth.h"
#include "avs\alignment.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <emmintrin.h>
#include <vector>
#include <string>


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

float* nitArray;

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

int fileWriteCounter = 0;

// MaxcLL
int highestrawvalue = 0;
int highestFrame = 0;
float highestFloatvalue = 0;
float highestnits = 0;
int highestValueX, highestValueY;

// MinCLL
int lowestrawvalue = INT_MAX;
int lowestFrame = INT_MAX;
float lowestFloatvalue;
float lowestnits;
int lowestValueX, lowestValueY;

// FALL Average
long double FALLSum = 0;
long framesCounted = 0;

// MaxFALL
float MaxFALL = 0;
int MaxFALLFrame;


template<typename pixel_t, int bits_per_pixel>
static inline void dofindmaxcll_c(const BYTE* ptr, int pitch, int width, int height, int thisFrame) {
    // width is rowsize
    const int max_pixel_value = (sizeof(pixel_t) == 1) ? 255 : ((1 << bits_per_pixel) - 1);

    width /= sizeof(pixel_t);

    float nits;
    pixel_t currentvalue;
    float currentvalueFloat;
    long double CLLSum = 0;
    float channelNits[3];
    float maxChannelNits;
    int CLLvalueCount = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            pixel_t currentvalue = reinterpret_cast<const pixel_t*>(ptr)[x];
            //nits = eotf_ST2084(currentvalueFloat);
            nits = nitArray[(int)currentvalue];

            if (maxFallAlgorithm == MAXFALL_ALLCHANNELS) {

                CLLSum += nits;
                CLLvalueCount++;
            }
            else {

                channelNits[x % 3] = nits;
            }
            if (currentvalue > highestrawvalue) {

                currentvalueFloat = (float)currentvalue / (float)max_pixel_value;
                highestnits = nits;
                highestrawvalue = currentvalue;
                highestFloatvalue = currentvalueFloat;
                highestValueX = x / 3;
                highestValueY = height - y - 1;
                highestFrame = thisFrame;
            }
            if (currentvalue < lowestrawvalue) {

                currentvalueFloat = (float)currentvalue / (float)max_pixel_value;
                lowestnits = nits;
                lowestrawvalue = currentvalue;
                lowestFloatvalue = currentvalueFloat;
                lowestValueX = x / 3;
                lowestValueY = height - y - 1;
                lowestFrame = thisFrame;
            }
            if ((x % 3 == 2) && (maxFallAlgorithm == MAXFALL_OFFICIAL)) {
                // we passed through R, G and B and populated channelNits, so we can now calculate their max and use it for CLLSum which in turn gets used for MaxFALL. 
                // This is how it's officially to be done according to SMPTE, even if it's questionable in my eyes logically, but hey.

                maxChannelNits = std::max(channelNits[0], channelNits[1]);
                maxChannelNits = std::max(maxChannelNits, channelNits[2]);

                CLLSum += maxChannelNits;
                CLLvalueCount++;
            }
        }
        ptr += pitch;
    }

    if (CLLvalueCount > 0) {
        float FALL = CLLSum / (long double)CLLvalueCount;
        FALLSum += FALL;
        if (FALL > MaxFALL) {
            MaxFALL = FALL;
            MaxFALLFrame = thisFrame;
        }
    }
}


class MaxCLLFind : public GenericVideoFilter {
public:
  MaxCLLFind(PClip clip, IScriptEnvironment* env)
    : GenericVideoFilter(clip) {

    int pixelsize = vi.ComponentSize();
    int bits_per_pixel = vi.BitsPerComponent();

    bool avx = !!(env->GetCPUFlags() & CPUF_AVX);
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
      switch (bits_per_pixel) {
      case 8:
        processor_ = &dofindmaxcll_c<uint8_t, 8>;
        break;
      case 10:
        processor_ = &dofindmaxcll_c<uint16_t, 10>;
        break;
      case 12:
        processor_ = &dofindmaxcll_c<uint16_t, 12>;
        break;
      case 14:
        processor_ = &dofindmaxcll_c<uint16_t, 14>;
        break;
      case 16:
        processor_ = &dofindmaxcll_c<uint16_t, 16>;
        break;
      case 32:
        processor_ = &dofindmaxcll_c<float, 1>; // bits_per_pixel n/a
        break;
      }
      processor_32aligned_ = processor_;
    //}
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);

  ~MaxCLLFind();

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  decltype(&dofindmaxcll_c<uint8_t, 8>) processor_;
  decltype(&dofindmaxcll_c<uint8_t, 8>) processor_32aligned_;
};

std::string statsFileName = "";

void writeCLLStats() {

	if (statsFileName == "") {
		FILE* test;
		int counter = 0;
		while (test = fopen((statsFileName = "MaxCLLFind_Results" + std::to_string(counter) + ".txt").c_str() , "r")) {
			fclose(test);
			counter++;
		}
		//statsFileName = "MaxCLLFind_Results" + std::to_string(counter) + ".txt";
	}

	float FALLAverage = FALLSum / (long double)framesCounted;

	std::ofstream myfile;
	myfile.open(statsFileName, std::ios::out | std::ios::app);
	myfile << "Stats at frame " << framesCounted << ":\n";
	myfile << "MaxCLL: " << highestnits << ", raw value: " << highestrawvalue << " " << highestFloatvalue << " at X " << highestValueX << " Y " << highestValueY << " at frame " << highestFrame << /*" byte depth: " << sizeof(pixel_t) <<*/ "\n";
	myfile << "MinCLL: " << lowestnits << ", raw value: " << lowestrawvalue << " " << lowestFloatvalue << " at X " << lowestValueX << " Y " << lowestValueY << " at frame " << lowestFrame << /*" byte depth: " << sizeof(pixel_t) <<*/ "\n";
	myfile << "MaxFALL: " << MaxFALL << " at frame " << MaxFALLFrame << "\n";
	myfile << "FALL Average: " << FALLAverage << " across " << framesCounted << " frames.\n\n";
	//myfile << "Dims: " << width << "x" << height;
	myfile.close();
}


MaxCLLFind::~MaxCLLFind() {

	delete[] nitArray;

	writeCLLStats();
}

PVideoFrame MaxCLLFind::GetFrame(int n, IScriptEnvironment *env) {
    PVideoFrame src = child->GetFrame(n, env);

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
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
            processor_32aligned_(ptr, pitch, width, height, n);
        }
        else {
            processor_(ptr, pitch, width, height, n);
        }
    }
	if (framesCounted++ % 100 == 0) {
		writeCLLStats();
	}

    return src;
}

const int MAXFALL_OFFICIAL = 0;
const int MAXFALL_ALLCHANNELS = 1;

int maxFallAlgorithm = 0;

AVSValue __cdecl create_maxcllfind(AVSValue args, void* user_data, IScriptEnvironment* env) {

	// Algorithms for MaxFALL frame averaging (default 0)
	// 0 = SMPTE2084 recommendation (average of highest channels of all pixels)
	// 1 = Average of all channels of all pixels
	maxFallAlgorithm = args[1].AsInt(MAXFALL_OFFICIAL);

	auto clip = args[0].AsClip();
	auto vi = clip->GetVideoInfo();

	if (!vi.IsRGB48()) {
		env->ThrowError("MaxCLLFind: clip MUST be RGB48, sorry. Use ConvertToRGB48 for example.");
	}

	int possibleValues = pow(2, 16);
	nitArray = new float[possibleValues]; // 16 bit value array for nit values corresponding to 16 bit RGB values

	for (int i = 0; i < possibleValues; i++) {
		nitArray[i] = eotf_ST2084((float)i / (float)possibleValues);
	}

    return new MaxCLLFind(clip, env);
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