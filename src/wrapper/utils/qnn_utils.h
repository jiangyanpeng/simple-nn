#ifndef SIMPLE_NN_QNN_UTILS_H_
#define SIMPLE_NN_QNN_UTILS_H_

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#if (defined __ARM_NEON)
#include <arm_neon.h>
#endif

#include <log.h>

namespace nn {
namespace wrap {
#if (defined __ARM_NEON) && ((defined __arm64__) || (defined __aarch64__))

    void transpose_ufixed_quant_3xN_neon(float* src, uint8_t* tgt, int n, int offset, float scale);
    void transpose_ufixed_quant_3xN_neon(float* src, uint16_t* tgt, int n, int offset, float scale);
    void transpose_ufixed_quant_4x4_neon(float* src,
                                         uint8_t* tgt,
                                         int m,
                                         int n,
                                         int offset,
                                         float scale);
    void transpose_ufixed_quant_4x4_neon(float* src,
                                         uint16_t* tgt,
                                         int m,
                                         int n,
                                         int offset,
                                         float scale);
    void transpose_ufixed_quant_8x8_neon(float* src,
                                         uint8_t* tgt,
                                         int m,
                                         int n,
                                         int offset,
                                         float scale);
    void transpose_ufixed_quant_8x8_neon(float* src,
                                         uint16_t* tgt,
                                         int m,
                                         int n,
                                         int offset,
                                         float scale);

    template <typename T_QuantType,
              typename = typename std::enable_if<std::is_unsigned<T_QuantType>::value>::type>
    void
    transpose_ufixed_quant(float* src, T_QuantType* tgt, int m, int n, int offset, float scale) {
        if (3 == m) {
            transpose_ufixed_quant_3xN_neon(src, tgt, n, offset, scale);
            return;
        }

        float max_value = static_cast<float>((1 << (8 * sizeof(T_QuantType))) - 1);
        int m0 = m / 8 * 8, m1 = m % 8, n0 = n / 8 * 8, n1 = n % 8;

        for (size_t i = 0; i < n0; i += 8) {
            for (size_t j = 0; j < m0; j += 8) {
                transpose_ufixed_quant_8x8_neon(
                    src + j * n + i, tgt + i * m + j, m, n, offset, scale);
            }
        }

        if (n1 >= 4) {
            for (size_t i = 0; i < m0; i += 4) {
                transpose_ufixed_quant_4x4_neon(
                    src + n0 + i * n, tgt + n0 * m + i, m, n, offset, scale);
            }
            n0 += 4;
        }

        for (size_t i = n0; i < n; ++i) {
            for (size_t j = 0; j < m0; ++j) {
                float t = round(src[n * j + i] / scale - offset);
                if (t < 0.f) {
                    t = 0.f;
                } else if (t > max_value) {
                    t = max_value;
                };
                tgt[m * i + j] = static_cast<T_QuantType>(t);
            }
        }

        if (m1 >= 4) {
            for (size_t i = 0; i < n0; i += 4) {
                transpose_ufixed_quant_4x4_neon(
                    src + m0 * n + i, tgt + m0 + i * m, m, n, offset, scale);
            }
            for (size_t i = n0; i < n; ++i) {
                for (size_t j = m0; j < m0 + 4; ++j) {
                    float t = round(src[n * j + i] / scale - offset);
                    if (t < 0.f) {
                        t = 0.f;
                    } else if (t > max_value) {
                        t = max_value;
                    };
                    tgt[m * i + j] = static_cast<T_QuantType>(t);
                }
            }
            m0 += 4;
        }

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = m0; j < m; ++j) {
                float t = round(src[n * j + i] / scale - offset);
                if (t < 0.f) {
                    t = 0.f;
                } else if (t > max_value) {
                    t = max_value;
                };
                tgt[m * i + j] = static_cast<T_QuantType>(t);
            }
        }
    }

    void transpose_ufixed_dequant_4x4_neon(uint8_t* src,
                                           float* tgt,
                                           int m,
                                           int n,
                                           int offset,
                                           float scale);
    void transpose_ufixed_dequant_4x4_neon(uint16_t* src,
                                           float* tgt,
                                           int m,
                                           int n,
                                           int offset,
                                           float scale);
    void transpose_ufixed_dequant_8x8_neon(uint8_t* src,
                                           float* tgt,
                                           int m,
                                           int n,
                                           int offset,
                                           float scale);
    void transpose_ufixed_dequant_8x8_neon(uint16_t* src,
                                           float* tgt,
                                           int m,
                                           int n,
                                           int offset,
                                           float scale);
    void
    transpose_ufixed_dequant_Mx3_neon(uint8_t* src, float* tgt, int m, int offset, float scale);
    void
    transpose_ufixed_dequant_Mx3_neon(uint16_t* src, float* tgt, int m, int offset, float scale);
    void
    transpose_ufixed_dequant_Mx4_neon(uint8_t* src, float* tgt, int m, int offset, float scale);
    void
    transpose_ufixed_dequant_Mx4_neon(uint16_t* src, float* tgt, int m, int offset, float scale);

    template <typename T_QuantType,
              typename = typename std::enable_if<std::is_unsigned<T_QuantType>::value>::type>
    void
    transpose_ufixed_dequant(T_QuantType* src, float* tgt, int m, int n, int offset, float scale) {
        if (3 == n) {
            transpose_ufixed_dequant_Mx3_neon(src, tgt, m, offset, scale);
            return;
        }

        if (4 == n) {
            transpose_ufixed_dequant_Mx4_neon(src, tgt, m, offset, scale);
            return;
        }

        int m0 = m / 8 * 8, m1 = m % 8, n0 = n / 8 * 8, n1 = n % 8;

        for (size_t i = 0; i < n0; i += 8) {
            for (size_t j = 0; j < m0; j += 8) {
                transpose_ufixed_dequant_8x8_neon(
                    src + j * n + i, tgt + i * m + j, m, n, offset, scale);
            }
        }

        if (n1 >= 4) {
            for (size_t i = 0; i < m0; i += 4) {
                transpose_ufixed_dequant_4x4_neon(
                    src + n0 + i * n, tgt + n0 * m + i, m, n, offset, scale);
            }
            n0 += 4;
        }

        for (size_t i = n0; i < n; ++i) {
            for (size_t j = 0; j < m0; ++j) {
                tgt[m * i + j] = static_cast<float>((src[n * j + i] + offset) * scale);
            }
        }

        if (m1 >= 4) {
            for (size_t i = 0; i < n0; i += 4) {
                transpose_ufixed_dequant_4x4_neon(
                    src + m0 * n + i, tgt + m0 + i * m, m, n, offset, scale);
            }
            for (size_t i = n0; i < n; ++i) {
                for (size_t j = m0; j < m0 + 4; ++j) {
                    tgt[m * i + j] = static_cast<float>((src[n * j + i] + offset) * scale);
                }
            }
            m0 += 4;
        }

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = m0; j < m; ++j) {
                tgt[m * i + j] = static_cast<float>((src[n * j + i] + offset) * scale);
            }
        }
    }

#else // (defined __ARM_NEON) && ((defined __arm64__) || (defined __aarch64__))

    template <typename T_QuantType,
              typename = typename std::enable_if<std::is_unsigned<T_QuantType>::value>::type>
    void
    transpose_ufixed_quant(float* src, T_QuantType* tgt, int m, int n, int offset, float scale) {
        float max_value = static_cast<float>((1 << (8 * sizeof(T_QuantType))) - 1);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                float t = round(src[n * j + i] / scale - offset);
                if (t < 0.f) {
                    t = 0.f;
                } else if (t > max_value) {
                    t = max_value;
                };
                tgt[m * i + j] = static_cast<T_QuantType>(t);
            }
        }
    }

    template <typename T_QuantType,
              typename = typename std::enable_if<std::is_unsigned<T_QuantType>::value>::type>
    void
    transpose_ufixed_dequant(T_QuantType* src, float* tgt, int m, int n, int offset, float scale) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                tgt[m * i + j] = static_cast<float>((src[n * j + i] + offset) * scale);
            }
        }
    }

#endif // (defined __ARM_NEON) && ((defined __arm64__) || (defined __aarch64__))

    template <typename T_QuantType,
              typename = typename std::enable_if<std::is_unsigned<T_QuantType>::value>::type>
    bool floatToTfN(T_QuantType* out, float* in, int32_t offset, float scale, size_t numElements) {
        if (nullptr == out || nullptr == in) {
            SIMPLE_LOG_ERROR("Received a nullptr");
            return false;
        }

        float max_value = static_cast<float>((1 << (8 * sizeof(T_QuantType))) - 1);
        for (size_t i = 0; i < numElements; ++i) {
            float t = round(in[i] / scale - offset);
            if (t < 0.f) {
                t = 0.f;
            } else if (t > max_value) {
                t = max_value;
            };
            out[i] = static_cast<T_QuantType>(t);
        }
        return true;
    }

    template <typename T_QuantType,
              typename = typename std::enable_if<std::is_unsigned<T_QuantType>::value>::type>
    bool floatToTfN_NHWC(T_QuantType* out,
                         float* in,
                         int32_t offset,
                         float scale,
                         const std::vector<size_t>& dims) {
        if (nullptr == out || nullptr == in) {
            SIMPLE_LOG_ERROR("Received a nullptr");
            return false;
        }

        size_t numElements =
            std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        if (4 != dims.size()) {
            return floatToTfN(out, in, offset, scale, numElements);
        }

        size_t n = dims[0], h = dims[1], w = dims[2], c = dims[3];
        size_t hw = h * w, chw = c * h * w;

        for (size_t i = 0; i < n; ++i) {
            transpose_ufixed_quant(in + i * chw, out + i * chw, c, hw, offset, scale);
        }

        return true;
    }

    template <typename T_QuantType>
    bool castFromFloat(T_QuantType* out, float* in, size_t numElements) {
        if (nullptr == out || nullptr == in) {
            SIMPLE_LOG_ERROR("Received a nullptr");
            return false;
        }
        for (size_t i = 0; i < numElements; i++) {
            out[i] = static_cast<T_QuantType>(in[i]);
        }
        return true;
    }

    template <typename T_QuantType>
    bool castFromFloat_NHWC(T_QuantType* out, float* in, const std::vector<size_t>& dims) {
        if (nullptr == out || nullptr == in) {
            SIMPLE_LOG_ERROR("Received a nullptr");
            return false;
        }

        size_t numElements =
            std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        if (4 != dims.size()) {
            return castFromFloat(out, in, numElements);
        }

        size_t n = dims[0], h = dims[1], w = dims[2], c = dims[3];
        size_t hw = h * w, chw = c * h * w;

        for (size_t i = 0; i < n; ++i) {
            float* ip       = in + i * chw;
            T_QuantType* op = out + i * chw;
            for (size_t j = 0; j < hw; ++j) {
                for (size_t k = 0; k < c; ++k) {
                    op[j * c + k] = static_cast<T_QuantType>(ip[k * hw + j]);
                }
            }
        }
        return true;
    }

    template <typename T_QuantType,
              typename = typename std::enable_if<std::is_unsigned<T_QuantType>::value>::type>
    bool tfNToFloat(float* out, T_QuantType* in, int32_t offset, float scale, size_t numElements) {
        if (nullptr == out || nullptr == in) {
            SIMPLE_LOG_ERROR("Received a nullptr");
            return false;
        }
        for (size_t i = 0; i < numElements; i++) {
            float quantizedValue = static_cast<float>(in[i]);
            float offsetDouble   = static_cast<float>(offset);
            out[i]               = static_cast<float>((quantizedValue + offsetDouble) * scale);
        }
        return true;
    }

    template <typename T_QuantType,
              typename = typename std::enable_if<std::is_unsigned<T_QuantType>::value>::type>
    bool tfNToFloat_NCHW(float* out,
                         T_QuantType* in,
                         int32_t offset,
                         float scale,
                         const std::vector<size_t>& dims) {
        if (nullptr == out || nullptr == in) {
            SIMPLE_LOG_ERROR("Received a nullptr");
            return false;
        }

        size_t numElements =
            std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        if (4 != dims.size()) {
            return tfNToFloat(out, in, offset, scale, numElements);
        }

        size_t n = dims[0], h = dims[1], w = dims[2], c = dims[3];
        size_t hw = h * w, chw = c * h * w;

        for (size_t i = 0; i < n; ++i) {
            transpose_ufixed_dequant(in + i * chw, out + i * chw, hw, c, offset, scale);
        }

        return true;
    }

    template <typename T_QuantType>
    bool castToFloat(float* out, T_QuantType* in, size_t numElements) {
        if (nullptr == out || nullptr == in) {
            SIMPLE_LOG_ERROR("Received a nullptr");
            return false;
        }
        for (size_t i = 0; i < numElements; i++) {
            out[i] = static_cast<float>(in[i]);
        }
        return true;
    }

    template <typename T_QuantType>
    bool castToFloat_NCHW(float* out, T_QuantType* in, const std::vector<size_t>& dims) {
        if (nullptr == out || nullptr == in) {
            SIMPLE_LOG_ERROR("Received a nullptr");
            return false;
        }

        size_t numElements =
            std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        if (4 != dims.size()) {
            return castToFloat(out, in, numElements);
        }

        size_t n = dims[0], h = dims[1], w = dims[2], c = dims[3];
        size_t hw = h * w, chw = c * h * w;

        for (size_t i = 0; i < n; ++i) {
            T_QuantType* ip = in + i * chw;
            float* op       = out + i * chw;
            for (size_t j = 0; j < c; ++j) {
                for (size_t k = 0; k < hw; ++k) {
                    op[j * hw + k] = static_cast<float>(ip[k * c + j]);
                }
            }
        }

        return true;
    }

#if (defined __ARM_NEON) && ((defined __arm64__) || (defined __aarch64__))

    template <>
    bool floatToTfN(uint8_t* out, float* in, int32_t offset, float scale, size_t numElements) {
        if (nullptr == out || nullptr == in) {
            SIMPLE_LOG_ERROR("Received a nullptr");
            return false;
        }

        int64_t n = static_cast<int64_t>(numElements / 16);

        if (n == 0) {
            for (size_t i = 0; i < numElements; ++i) {
                float t = round(in[i] / scale - offset);
                if (t < 0.f) {
                    t = 0.f;
                } else if (t > 255.f) {
                    t = 255.f;
                };
                out[i] = static_cast<uint8_t>(t);
            }
            return true;
        }

        uint8_t* op              = out;
        float* ip                = in;
        float32x4_t vf32x4Scale  = vdupq_n_f32(1.f / scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile("0:                                 \n"
                     "ld1    {v0.4s-v3.4s}, [%0], #64    \n"
                     "fmul   v0.4s, v0.4s, %2.4s         \n"
                     "fmul   v1.4s, v1.4s, %2.4s         \n"
                     "fmul   v2.4s, v2.4s, %2.4s         \n"
                     "fmul   v3.4s, v3.4s, %2.4s         \n"
                     "fsub   v0.4s, v0.4s, %3.4s         \n"
                     "fsub   v1.4s, v1.4s, %3.4s         \n"
                     "fsub   v2.4s, v2.4s, %3.4s         \n"
                     "fsub   v3.4s, v3.4s, %3.4s         \n"
                     "fcvtau v0.4s, v0.4s                \n"
                     "fcvtau v1.4s, v1.4s                \n"
                     "fcvtau v2.4s, v2.4s                \n"
                     "fcvtau v3.4s, v3.4s                \n"
                     "uqxtn  v0.4h, v0.4s                \n"
                     "uqxtn  v2.4h, v2.4s                \n"
                     "uqxtn2 v0.8h, v1.4s                \n"
                     "uqxtn2 v2.8h, v3.4s                \n"
                     "uqxtn  v0.8b, v0.8h                \n"
                     "uqxtn  v2.8b, v2.8h                \n"
                     "str    d0, [%1], #8                \n"
                     "str    d2, [%1], #8                \n"
                     "subs   %4, %4, #1                  \n"
                     "bgt    0b                          \n"
                     : "=r"(ip), "=r"(op), "=w"(vf32x4Scale), "=w"(vf32x4Offset), "=r"(n)
                     : "0"(ip), "1"(op), "2"(vf32x4Scale), "3"(vf32x4Offset), "4"(n)
                     : "cc", "memory", "v0", "v1", "v2", "v3");

        for (size_t i = numElements / 16 * 16; i < numElements; ++i) {
            float t = round(in[i] / scale - offset);
            if (t < 0.f) {
                t = 0.f;
            } else if (t > 255.f) {
                t = 255.f;
            };
            out[i] = static_cast<uint8_t>(t);
        }

        return true;
    }

    template <>
    bool floatToTfN(uint16_t* out, float* in, int32_t offset, float scale, size_t numElements) {
        if (nullptr == out || nullptr == in) {
            SIMPLE_LOG_ERROR("Received a nullptr");
            return false;
        }

        int64_t n = static_cast<int64_t>(numElements / 16);
        if (n == 0) {
            for (size_t i = 0; i < numElements; ++i) {
                float t = round(in[i] / scale - offset);
                if (t < 0.f) {
                    t = 0.f;
                } else if (t > 65535.f) {
                    t = 65535.f;
                };
                out[i] = static_cast<uint16_t>(t);
            }
            return true;
        }

        uint16_t* op             = out;
        float* ip                = in;
        float32x4_t vf32x4Scale  = vdupq_n_f32(1.f / scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile("0:                                 \n"
                     "ld1    {v0.4s-v3.4s}, [%0], #64    \n"
                     "fmul   v0.4s, v0.4s, %2.4s         \n"
                     "fmul   v1.4s, v1.4s, %2.4s         \n"
                     "fmul   v2.4s, v2.4s, %2.4s         \n"
                     "fmul   v3.4s, v3.4s, %2.4s         \n"
                     "fsub   v0.4s, v0.4s, %3.4s         \n"
                     "fsub   v1.4s, v1.4s, %3.4s         \n"
                     "fsub   v2.4s, v2.4s, %3.4s         \n"
                     "fsub   v3.4s, v3.4s, %3.4s         \n"
                     "fcvtau v0.4s, v0.4s                \n"
                     "fcvtau v1.4s, v1.4s                \n"
                     "fcvtau v2.4s, v2.4s                \n"
                     "fcvtau v3.4s, v3.4s                \n"
                     "uqxtn  v0.4h, v0.4s                \n"
                     "uqxtn  v2.4h, v2.4s                \n"
                     "uqxtn2 v0.8h, v1.4s                \n"
                     "uqxtn2 v2.8h, v3.4s                \n"
                     "st1    {v0.4s}, [%1], #16          \n"
                     "st1    {v2.4s}, [%1], #16          \n"
                     "subs   %4, %4, #1                  \n"
                     "bgt    0b                          \n"
                     : "=r"(ip), "=r"(op), "=w"(vf32x4Scale), "=w"(vf32x4Offset), "=r"(n)
                     : "0"(ip), "1"(op), "2"(vf32x4Scale), "3"(vf32x4Offset), "4"(n)
                     : "cc", "memory", "v0", "v1", "v2", "v3");

        for (size_t i = numElements / 16 * 16; i < numElements; ++i) {
            float t = round(in[i] / scale - offset);
            if (t < 0.f) {
                t = 0.f;
            } else if (t > 65535.f) {
                t = 65535.f;
            };
            out[i] = static_cast<uint16_t>(t);
        }
        return true;
    }

    void transpose_ufixed_quant_3xN_neon(float* src, uint8_t* tgt, int n, int offset, float scale) {
        float* src0   = src;
        float* src1   = src + n;
        float* src2   = src + 2 * n;
        uint8_t* tgt0 = tgt;

        int ii = n / 8;
        if (ii == 0) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    float t = round(src[n * j + i] / scale - offset);
                    if (t < 0.f) {
                        t = 0.f;
                    } else if (t > 255.f) {
                        t = 255.f;
                    };
                    tgt[3 * i + j] = static_cast<uint8_t>(t);
                }
            }
            return;
        }

        float32x4_t vf32x4Scale  = vdupq_n_f32(1.f / scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile("0:                                 \n" // loop 0
                     "ld1    {v0.4s, v1.4s}, [%0], #32   \n" // load src0 to v0
                     "ld1    {v2.4s, v3.4s}, [%1], #32   \n" // load src1 to v1
                     "ld1    {v4.4s, v5.4s}, [%2], #32   \n" // load src2 to v2
                     "fmul   v0.4s, v0.4s, %5.4s         \n" // mul by vf32x4Scale
                     "fmul   v1.4s, v1.4s, %5.4s         \n"
                     "fmul   v2.4s, v2.4s, %5.4s         \n"
                     "fmul   v3.4s, v3.4s, %5.4s         \n"
                     "fmul   v4.4s, v4.4s, %5.4s         \n"
                     "fmul   v5.4s, v5.4s, %5.4s         \n"
                     "fsub   v0.4s, v0.4s, %6.4s         \n" // sub to vf32x4Offset
                     "fsub   v1.4s, v1.4s, %6.4s         \n"
                     "fsub   v2.4s, v2.4s, %6.4s         \n"
                     "fsub   v3.4s, v3.4s, %6.4s         \n"
                     "fsub   v4.4s, v4.4s, %6.4s         \n"
                     "fsub   v5.4s, v5.4s, %6.4s         \n"
                     "fcvtau v0.4s, v0.4s                \n" // float --> uint32
                     "fcvtau v1.4s, v1.4s                \n"
                     "fcvtau v2.4s, v2.4s                \n"
                     "fcvtau v3.4s, v3.4s                \n"
                     "fcvtau v4.4s, v4.4s                \n"
                     "fcvtau v5.4s, v5.4s                \n"
                     "uqxtn  v0.4h, v0.4s                \n" // uint32 --> uint16
                     "uqxtn  v2.4h, v2.4s                \n"
                     "uqxtn  v4.4h, v4.4s                \n"
                     "uqxtn2 v0.8h, v1.4s                \n"
                     "uqxtn2 v2.8h, v3.4s                \n"
                     "uqxtn2 v4.8h, v5.4s                \n"
                     "uqxtn  v0.8b, v0.8h                \n" // uint16 --> uint8
                     "uqxtn  v2.8b, v2.8h                \n"
                     "uqxtn  v4.8b, v4.8h                \n"
                     "mov    v1.8b, v2.8b                \n"
                     "mov    v2.8b, v4.8b                \n"
                     "st3    {v0.8b-v2.8b}, [%3], #24    \n" // save to tgt
                     "subs   %w4, %w4, #1                \n" // ii--
                     "bgt    0b                          \n" // if ii > 0: goto loop 0
                     : "=r"(src0),
                       "=r"(src1),
                       "=r"(src2),
                       "=r"(tgt0),
                       "=r"(ii),
                       "=w"(vf32x4Scale),
                       "=w"(vf32x4Offset)
                     : "0"(src0),
                       "1"(src1),
                       "2"(src2),
                       "3"(tgt0),
                       "4"(ii),
                       "5"(vf32x4Scale),
                       "6"(vf32x4Offset)
                     : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");

        for (size_t i = n / 8 * 8; i < n; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                float t = round(src[n * j + i] / scale - offset);
                if (t < 0.f) {
                    t = 0.f;
                } else if (t > 255.f) {
                    t = 255.f;
                };
                tgt[3 * i + j] = static_cast<uint8_t>(t);
            }
        }
    }

    void
    transpose_ufixed_quant_3xN_neon(float* src, uint16_t* tgt, int n, int offset, float scale) {
        float* src0    = src;
        float* src1    = src + n;
        float* src2    = src + 2 * n;
        uint16_t* tgt0 = tgt;

        int ii = n / 8;
        if (ii == 0) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    float t = round(src[n * j + i] / scale - offset);
                    if (t < 0.f) {
                        t = 0.f;
                    } else if (t > 65535.f) {
                        t = 65535.f;
                    };
                    tgt[3 * i + j] = static_cast<uint16_t>(t);
                }
            }
            return;
        }

        float32x4_t vf32x4Scale  = vdupq_n_f32(1.f / scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile("0:                                 \n" // loop 0
                     "ld1    {v0.4s, v1.4s}, [%0], #32   \n" // load src0 to v0
                     "ld1    {v2.4s, v3.4s}, [%1], #32   \n" // load src1 to v1
                     "ld1    {v4.4s, v5.4s}, [%2], #32   \n" // load src2 to v2
                     "fmul   v0.4s, v0.4s, %5.4s         \n" // mul by vf32x4Scale
                     "fmul   v1.4s, v1.4s, %5.4s         \n"
                     "fmul   v2.4s, v2.4s, %5.4s         \n"
                     "fmul   v3.4s, v3.4s, %5.4s         \n"
                     "fmul   v4.4s, v4.4s, %5.4s         \n"
                     "fmul   v5.4s, v5.4s, %5.4s         \n"
                     "fsub   v0.4s, v0.4s, %6.4s         \n" // sub to vf32x4Offset
                     "fsub   v1.4s, v1.4s, %6.4s         \n"
                     "fsub   v2.4s, v2.4s, %6.4s         \n"
                     "fsub   v3.4s, v3.4s, %6.4s         \n"
                     "fsub   v4.4s, v4.4s, %6.4s         \n"
                     "fsub   v5.4s, v5.4s, %6.4s         \n"
                     "fcvtau v0.4s, v0.4s                \n" // float --> uint32
                     "fcvtau v1.4s, v1.4s                \n"
                     "fcvtau v2.4s, v2.4s                \n"
                     "fcvtau v3.4s, v3.4s                \n"
                     "fcvtau v4.4s, v4.4s                \n"
                     "fcvtau v5.4s, v5.4s                \n"
                     "uqxtn  v0.4h, v0.4s                \n" // uint32 --> uint16
                     "uqxtn  v2.4h, v2.4s                \n"
                     "uqxtn  v4.4h, v4.4s                \n"
                     "uqxtn2 v0.8h, v1.4s                \n"
                     "uqxtn2 v2.8h, v3.4s                \n"
                     "uqxtn2 v4.8h, v5.4s                \n"
                     "mov    v1.8h, v2.8h                \n"
                     "mov    v2.8h, v4.8h                \n"
                     "st3    {v0.8h-v2.8h}, [%3], #48    \n" // save to tgt
                     "subs   %w4, %w4, #1                \n" // ii--
                     "bgt    0b                          \n" // if ii > 0: goto loop 0
                     : "=r"(src0),
                       "=r"(src1),
                       "=r"(src2),
                       "=r"(tgt0),
                       "=r"(ii),
                       "=w"(vf32x4Scale),
                       "=w"(vf32x4Offset)
                     : "0"(src0),
                       "1"(src1),
                       "2"(src2),
                       "3"(tgt0),
                       "4"(ii),
                       "5"(vf32x4Scale),
                       "6"(vf32x4Offset)
                     : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");

        for (size_t i = n / 8 * 8; i < n; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                float t = round(src[n * j + i] / scale - offset);
                if (t < 0.f) {
                    t = 0.f;
                } else if (t > 65535.f) {
                    t = 65535.f;
                };
                tgt[3 * i + j] = static_cast<uint16_t>(t);
            }
        }
    }

    void transpose_ufixed_quant_4x4_neon(float* src,
                                         uint8_t* tgt,
                                         int m,
                                         int n,
                                         int offset,
                                         float scale) {
        long _m                  = sizeof(uint8_t) * m;
        long _n                  = sizeof(float) * n;
        float32x4_t vf32x4Scale  = vdupq_n_f32(1.f / scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile(
            "ld1    {v0.4s}, [%0]       \n" // load src
            "add    %0, %0, %3          \n" // src += _m
            "ld1    {v1.4s}, [%0]       \n"
            "add    %0, %0, %3          \n"
            "ld1    {v2.4s}, [%0]       \n"
            "add    %0, %0, %3          \n"
            "ld1    {v3.4s}, [%0]       \n"
            "trn1   v4.4s, v0.4s, v1.4s \n" // *************************
            "trn1   v5.4s, v2.4s, v3.4s \n" //
            "trn2   v6.4s, v0.4s, v1.4s \n" //  1  2  3  4     1 5  9 13
            "trn2   v7.4s, v2.4s, v3.4s \n" //  5  6  7  8 --> 2 6 10 14
            "trn1   v0.2d, v4.2d, v5.2d \n" //  9 10 11 12     3 7 11 15
            "trn1   v1.2d, v6.2d, v7.2d \n" // 13 14 15 16     4 8 12 16
            "trn2   v2.2d, v4.2d, v5.2d \n" //
            "trn2   v3.2d, v6.2d, v7.2d \n" // *************************
            "fmul   v0.4s, v0.4s, %4.4s \n" // mul by vf32x4Scale
            "fmul   v1.4s, v1.4s, %4.4s \n"
            "fmul   v2.4s, v2.4s, %4.4s \n"
            "fmul   v3.4s, v3.4s, %4.4s \n"
            "fsub   v0.4s, v0.4s, %5.4s \n" // sub to vf32x4Offset
            "fsub   v1.4s, v1.4s, %5.4s \n"
            "fsub   v2.4s, v2.4s, %5.4s \n"
            "fsub   v3.4s, v3.4s, %5.4s \n"
            "fcvtau v0.4s, v0.4s        \n" // float --> uint32
            "fcvtau v1.4s, v1.4s        \n"
            "fcvtau v2.4s, v2.4s        \n"
            "fcvtau v3.4s, v3.4s        \n"
            "uqxtn  v0.4h, v0.4s        \n" // uint32 --> uint16
            "uqxtn  v1.4h, v1.4s        \n"
            "uqxtn  v2.4h, v2.4s        \n"
            "uqxtn  v3.4h, v3.4s        \n"
            "uqxtn  v0.8b, v0.8h        \n" // uint16 --> uint8
            "uqxtn  v1.8b, v1.8h        \n"
            "uqxtn  v2.8b, v2.8h        \n"
            "uqxtn  v3.8b, v3.8h        \n"
            "str    s0, [%1]            \n" // save to tgt
            "add    %1, %1, %2          \n" // tgt += _n
            "str    s1, [%1]            \n"
            "add    %1, %1, %2          \n"
            "str    s2, [%1]            \n"
            "add    %1, %1, %2          \n"
            "str    s3, [%1]            \n"
            : "=r"(src), "=r"(tgt), "=r"(_m), "=r"(_n), "=w"(vf32x4Scale), "=w"(vf32x4Offset)
            : "0"(src), "1"(tgt), "2"(_m), "3"(_n), "4"(vf32x4Scale), "5"(vf32x4Offset)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }

    void transpose_ufixed_quant_4x4_neon(float* src,
                                         uint16_t* tgt,
                                         int m,
                                         int n,
                                         int offset,
                                         float scale) {
        long _m                  = sizeof(uint16_t) * m;
        long _n                  = sizeof(float) * n;
        float32x4_t vf32x4Scale  = vdupq_n_f32(1.f / scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile(
            "ld1    {v0.4s}, [%0]       \n" // load src
            "add    %0, %0, %3          \n" // src += _m
            "ld1    {v1.4s}, [%0]       \n"
            "add    %0, %0, %3          \n"
            "ld1    {v2.4s}, [%0]       \n"
            "add    %0, %0, %3          \n"
            "ld1    {v3.4s}, [%0]       \n"
            "trn1   v4.4s, v0.4s, v1.4s \n" // *************************
            "trn1   v5.4s, v2.4s, v3.4s \n" //
            "trn2   v6.4s, v0.4s, v1.4s \n" //  1  2  3  4     1 5  9 13
            "trn2   v7.4s, v2.4s, v3.4s \n" //  5  6  7  8 --> 2 6 10 14
            "trn1   v0.2d, v4.2d, v5.2d \n" //  9 10 11 12     3 7 11 15
            "trn1   v1.2d, v6.2d, v7.2d \n" // 13 14 15 16     4 8 12 16
            "trn2   v2.2d, v4.2d, v5.2d \n" //
            "trn2   v3.2d, v6.2d, v7.2d \n" // *************************
            "fmul   v0.4s, v0.4s, %4.4s \n" // mul by vf32x4Scale
            "fmul   v1.4s, v1.4s, %4.4s \n"
            "fmul   v2.4s, v2.4s, %4.4s \n"
            "fmul   v3.4s, v3.4s, %4.4s \n"
            "fsub   v0.4s, v0.4s, %5.4s \n" // sub to vf32x4Offset
            "fsub   v1.4s, v1.4s, %5.4s \n"
            "fsub   v2.4s, v2.4s, %5.4s \n"
            "fsub   v3.4s, v3.4s, %5.4s \n"
            "fcvtau v0.4s, v0.4s        \n" // float --> uint32
            "fcvtau v1.4s, v1.4s        \n"
            "fcvtau v2.4s, v2.4s        \n"
            "fcvtau v3.4s, v3.4s        \n"
            "uqxtn  v0.4h, v0.4s        \n" // uint32 --> uint16
            "uqxtn  v1.4h, v1.4s        \n"
            "uqxtn  v2.4h, v2.4s        \n"
            "uqxtn  v3.4h, v3.4s        \n"
            "str    d0, [%1]            \n" // save to tgt
            "add    %1, %1, %2          \n" // tgt += _n
            "str    d1, [%1]            \n"
            "add    %1, %1, %2          \n"
            "str    d2, [%1]            \n"
            "add    %1, %1, %2          \n"
            "str    d3, [%1]            \n"
            : "=r"(src), "=r"(tgt), "=r"(_m), "=r"(_n), "=w"(vf32x4Scale), "=w"(vf32x4Offset)
            : "0"(src), "1"(tgt), "2"(_m), "3"(_n), "4"(vf32x4Scale), "5"(vf32x4Offset)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }

    void transpose_ufixed_quant_8x8_neon(float* src,
                                         uint8_t* tgt,
                                         int m,
                                         int n,
                                         int offset,
                                         float scale) {
        long _m                  = sizeof(uint8_t) * m;
        long _n                  = sizeof(float) * n;
        float32x4_t vf32x4Scale  = vdupq_n_f32(1.f / scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile(
            "ld1    {v0.4s, v1.4s}, [%0]    \n" // load src
            "add    %0, %0, %3              \n"
            "ld1    {v2.4s, v3.4s}, [%0]    \n"
            "add    %0, %0, %3              \n"
            "ld1    {v4.4s, v5.4s}, [%0]    \n"
            "add    %0, %0, %3              \n"
            "ld1    {v6.4s, v7.4s}, [%0]    \n"
            "add    %0, %0, %3              \n"
            "ld1    {v20.4s, v21.4s}, [%0]  \n"
            "add    %0, %0, %3              \n"
            "ld1    {v22.4s, v23.4s}, [%0]  \n"
            "add    %0, %0, %3              \n"
            "ld1    {v24.4s, v25.4s}, [%0]  \n"
            "add    %0, %0, %3              \n"
            "ld1    {v26.4s, v27.4s}, [%0]  \n"
            "trn1   v16.4s, v0.4s, v2.4s    \n" // transpose 8x8
            "trn1   v17.4s, v4.4s, v6.4s    \n"
            "trn2   v18.4s, v0.4s, v2.4s    \n"
            "trn2   v19.4s, v4.4s, v6.4s    \n"
            "trn1   v0.2d, v16.2d, v17.2d   \n"
            "trn1   v2.2d, v18.2d, v19.2d   \n"
            "trn2   v4.2d, v16.2d, v17.2d   \n"
            "trn2   v6.2d, v18.2d, v19.2d   \n"
            "mov    v16.4s, v1.4s           \n"
            "mov    v17.4s, v3.4s           \n"
            "mov    v18.4s, v5.4s           \n"
            "mov    v19.4s, v7.4s           \n"
            "trn1   v1.4s, v16.4s, v17.4s   \n"
            "trn1   v3.4s, v18.4s, v19.4s   \n"
            "trn2   v5.4s, v16.4s, v17.4s   \n"
            "trn2   v7.4s, v18.4s, v19.4s   \n"
            "trn1   v16.4s, v20.4s, v22.4s  \n"
            "trn1   v17.4s, v24.4s, v26.4s  \n"
            "trn2   v18.4s, v20.4s, v22.4s  \n"
            "trn2   v19.4s, v24.4s, v26.4s  \n"
            "trn1   v20.2d, v1.2d, v3.2d    \n"
            "trn1   v22.2d, v5.2d, v7.2d    \n"
            "trn2   v24.2d, v1.2d, v3.2d    \n"
            "trn2   v26.2d, v5.2d, v7.2d    \n"
            "trn1   v1.2d, v16.2d, v17.2d   \n"
            "trn1   v3.2d, v18.2d, v19.2d   \n"
            "trn2   v5.2d, v16.2d, v17.2d   \n"
            "trn2   v7.2d, v18.2d, v19.2d   \n"
            "trn1   v16.4s, v21.4s, v23.4s  \n"
            "trn1   v17.4s, v25.4s, v27.4s  \n"
            "trn2   v18.4s, v21.4s, v23.4s  \n"
            "trn2   v19.4s, v25.4s, v27.4s  \n"
            "trn1   v21.2d, v16.2d, v17.2d  \n"
            "trn1   v23.2d, v18.2d, v19.2d  \n"
            "trn2   v25.2d, v16.2d, v17.2d  \n"
            "trn2   v27.2d, v18.2d, v19.2d  \n"
            "fmul   v0.4s, v0.4s, %4.4s     \n" // mul by vf32x4Scale
            "fmul   v1.4s, v1.4s, %4.4s     \n"
            "fmul   v2.4s, v2.4s, %4.4s     \n"
            "fmul   v3.4s, v3.4s, %4.4s     \n"
            "fmul   v4.4s, v4.4s, %4.4s     \n"
            "fmul   v5.4s, v5.4s, %4.4s     \n"
            "fmul   v6.4s, v6.4s, %4.4s     \n"
            "fmul   v7.4s, v7.4s, %4.4s     \n"
            "fmul   v20.4s, v20.4s, %4.4s   \n"
            "fmul   v21.4s, v21.4s, %4.4s   \n"
            "fmul   v22.4s, v22.4s, %4.4s   \n"
            "fmul   v23.4s, v23.4s, %4.4s   \n"
            "fmul   v24.4s, v24.4s, %4.4s   \n"
            "fmul   v25.4s, v25.4s, %4.4s   \n"
            "fmul   v26.4s, v26.4s, %4.4s   \n"
            "fmul   v27.4s, v27.4s, %4.4s   \n"
            "fsub   v0.4s, v0.4s, %5.4s     \n" // sub to vf32x4Offset
            "fsub   v1.4s, v1.4s, %5.4s     \n"
            "fsub   v2.4s, v2.4s, %5.4s     \n"
            "fsub   v3.4s, v3.4s, %5.4s     \n"
            "fsub   v4.4s, v4.4s, %5.4s     \n"
            "fsub   v5.4s, v5.4s, %5.4s     \n"
            "fsub   v6.4s, v6.4s, %5.4s     \n"
            "fsub   v7.4s, v7.4s, %5.4s     \n"
            "fsub   v20.4s, v20.4s, %5.4s   \n"
            "fsub   v21.4s, v21.4s, %5.4s   \n"
            "fsub   v22.4s, v22.4s, %5.4s   \n"
            "fsub   v23.4s, v23.4s, %5.4s   \n"
            "fsub   v24.4s, v24.4s, %5.4s   \n"
            "fsub   v25.4s, v25.4s, %5.4s   \n"
            "fsub   v26.4s, v26.4s, %5.4s   \n"
            "fsub   v27.4s, v27.4s, %5.4s   \n"
            "fcvtau v0.4s, v0.4s            \n" // float --> uint32
            "fcvtau v1.4s, v1.4s            \n"
            "fcvtau v2.4s, v2.4s            \n"
            "fcvtau v3.4s, v3.4s            \n"
            "fcvtau v4.4s, v4.4s            \n"
            "fcvtau v5.4s, v5.4s            \n"
            "fcvtau v6.4s, v6.4s            \n"
            "fcvtau v7.4s, v7.4s            \n"
            "fcvtau v20.4s, v20.4s          \n"
            "fcvtau v21.4s, v21.4s          \n"
            "fcvtau v22.4s, v22.4s          \n"
            "fcvtau v23.4s, v23.4s          \n"
            "fcvtau v24.4s, v24.4s          \n"
            "fcvtau v25.4s, v25.4s          \n"
            "fcvtau v26.4s, v26.4s          \n"
            "fcvtau v27.4s, v27.4s          \n"
            "uqxtn  v0.4h, v0.4s            \n" // uint32 --> uint16
            "uqxtn  v2.4h, v2.4s            \n"
            "uqxtn  v4.4h, v4.4s            \n"
            "uqxtn  v6.4h, v6.4s            \n"
            "uqxtn  v20.4h, v20.4s          \n"
            "uqxtn  v22.4h, v22.4s          \n"
            "uqxtn  v24.4h, v24.4s          \n"
            "uqxtn  v26.4h, v26.4s          \n"
            "uqxtn2 v0.8h, v1.4s            \n"
            "uqxtn2 v2.8h, v3.4s            \n"
            "uqxtn2 v4.8h, v5.4s            \n"
            "uqxtn2 v6.8h, v7.4s            \n"
            "uqxtn2 v20.8h, v21.4s          \n"
            "uqxtn2 v22.8h, v23.4s          \n"
            "uqxtn2 v24.8h, v25.4s          \n"
            "uqxtn2 v26.8h, v27.4s          \n"
            "uqxtn  v0.8b, v0.8h            \n" // uint16 --> uint8
            "uqxtn  v2.8b, v2.8h            \n"
            "uqxtn  v4.8b, v4.8h            \n"
            "uqxtn  v6.8b, v6.8h            \n"
            "uqxtn  v20.8b, v20.8h          \n"
            "uqxtn  v22.8b, v22.8h          \n"
            "uqxtn  v24.8b, v24.8h          \n"
            "uqxtn  v26.8b, v26.8h          \n"
            "str    d0, [%1]                \n" // save to tgt
            "add    %1, %1, %2              \n"
            "str    d2, [%1]                \n"
            "add    %1, %1, %2              \n"
            "str    d4, [%1]                \n"
            "add    %1, %1, %2              \n"
            "str    d6, [%1]                \n"
            "add    %1, %1, %2              \n"
            "str    d20, [%1]               \n"
            "add    %1, %1, %2              \n"
            "str    d22, [%1]               \n"
            "add    %1, %1, %2              \n"
            "str    d24, [%1]               \n"
            "add    %1, %1, %2              \n"
            "str    d26, [%1]               \n"
            : "=r"(src), "=r"(tgt), "=r"(_m), "=r"(_n), "=w"(vf32x4Scale), "=w"(vf32x4Offset)
            : "0"(src), "1"(tgt), "2"(_m), "3"(_n), "4"(vf32x4Scale), "5"(vf32x4Offset)
            : "cc",
              "memory",
              "v0",
              "v1",
              "v2",
              "v3",
              "v4",
              "v5",
              "v6",
              "v7",
              "v16",
              "v17",
              "v18",
              "v19",
              "v20",
              "v21",
              "v22",
              "v23",
              "v24",
              "v25",
              "v26",
              "v27");
    }

    void transpose_ufixed_quant_8x8_neon(float* src,
                                         uint16_t* tgt,
                                         int m,
                                         int n,
                                         int offset,
                                         float scale) {
        long _m                  = sizeof(uint16_t) * m;
        long _n                  = sizeof(float) * n;
        float32x4_t vf32x4Scale  = vdupq_n_f32(1.f / scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile(
            "ld1    {v0.4s, v1.4s}, [%0]    \n" // load src
            "add    %0, %0, %3              \n"
            "ld1    {v2.4s, v3.4s}, [%0]    \n"
            "add    %0, %0, %3              \n"
            "ld1    {v4.4s, v5.4s}, [%0]    \n"
            "add    %0, %0, %3              \n"
            "ld1    {v6.4s, v7.4s}, [%0]    \n"
            "add    %0, %0, %3              \n"
            "ld1    {v20.4s, v21.4s}, [%0]  \n"
            "add    %0, %0, %3              \n"
            "ld1    {v22.4s, v23.4s}, [%0]  \n"
            "add    %0, %0, %3              \n"
            "ld1    {v24.4s, v25.4s}, [%0]  \n"
            "add    %0, %0, %3              \n"
            "ld1    {v26.4s, v27.4s}, [%0]  \n"
            "trn1   v16.4s, v0.4s, v2.4s    \n" // transpose 8x8
            "trn1   v17.4s, v4.4s, v6.4s    \n"
            "trn2   v18.4s, v0.4s, v2.4s    \n"
            "trn2   v19.4s, v4.4s, v6.4s    \n"
            "trn1   v0.2d, v16.2d, v17.2d   \n"
            "trn1   v2.2d, v18.2d, v19.2d   \n"
            "trn2   v4.2d, v16.2d, v17.2d   \n"
            "trn2   v6.2d, v18.2d, v19.2d   \n"
            "mov    v16.4s, v1.4s           \n"
            "mov    v17.4s, v3.4s           \n"
            "mov    v18.4s, v5.4s           \n"
            "mov    v19.4s, v7.4s           \n"
            "trn1   v1.4s, v16.4s, v17.4s   \n"
            "trn1   v3.4s, v18.4s, v19.4s   \n"
            "trn2   v5.4s, v16.4s, v17.4s   \n"
            "trn2   v7.4s, v18.4s, v19.4s   \n"
            "trn1   v16.4s, v20.4s, v22.4s  \n"
            "trn1   v17.4s, v24.4s, v26.4s  \n"
            "trn2   v18.4s, v20.4s, v22.4s  \n"
            "trn2   v19.4s, v24.4s, v26.4s  \n"
            "trn1   v20.2d, v1.2d, v3.2d    \n"
            "trn1   v22.2d, v5.2d, v7.2d    \n"
            "trn2   v24.2d, v1.2d, v3.2d    \n"
            "trn2   v26.2d, v5.2d, v7.2d    \n"
            "trn1   v1.2d, v16.2d, v17.2d   \n"
            "trn1   v3.2d, v18.2d, v19.2d   \n"
            "trn2   v5.2d, v16.2d, v17.2d   \n"
            "trn2   v7.2d, v18.2d, v19.2d   \n"
            "trn1   v16.4s, v21.4s, v23.4s  \n"
            "trn1   v17.4s, v25.4s, v27.4s  \n"
            "trn2   v18.4s, v21.4s, v23.4s  \n"
            "trn2   v19.4s, v25.4s, v27.4s  \n"
            "trn1   v21.2d, v16.2d, v17.2d  \n"
            "trn1   v23.2d, v18.2d, v19.2d  \n"
            "trn2   v25.2d, v16.2d, v17.2d  \n"
            "trn2   v27.2d, v18.2d, v19.2d  \n"
            "fmul   v0.4s, v0.4s, %4.4s     \n" // mul by vf32x4Scale
            "fmul   v1.4s, v1.4s, %4.4s     \n"
            "fmul   v2.4s, v2.4s, %4.4s     \n"
            "fmul   v3.4s, v3.4s, %4.4s     \n"
            "fmul   v4.4s, v4.4s, %4.4s     \n"
            "fmul   v5.4s, v5.4s, %4.4s     \n"
            "fmul   v6.4s, v6.4s, %4.4s     \n"
            "fmul   v7.4s, v7.4s, %4.4s     \n"
            "fmul   v20.4s, v20.4s, %4.4s   \n"
            "fmul   v21.4s, v21.4s, %4.4s   \n"
            "fmul   v22.4s, v22.4s, %4.4s   \n"
            "fmul   v23.4s, v23.4s, %4.4s   \n"
            "fmul   v24.4s, v24.4s, %4.4s   \n"
            "fmul   v25.4s, v25.4s, %4.4s   \n"
            "fmul   v26.4s, v26.4s, %4.4s   \n"
            "fmul   v27.4s, v27.4s, %4.4s   \n"
            "fsub   v0.4s, v0.4s, %5.4s     \n" // sub to vf32x4Offset
            "fsub   v1.4s, v1.4s, %5.4s     \n"
            "fsub   v2.4s, v2.4s, %5.4s     \n"
            "fsub   v3.4s, v3.4s, %5.4s     \n"
            "fsub   v4.4s, v4.4s, %5.4s     \n"
            "fsub   v5.4s, v5.4s, %5.4s     \n"
            "fsub   v6.4s, v6.4s, %5.4s     \n"
            "fsub   v7.4s, v7.4s, %5.4s     \n"
            "fsub   v20.4s, v20.4s, %5.4s   \n"
            "fsub   v21.4s, v21.4s, %5.4s   \n"
            "fsub   v22.4s, v22.4s, %5.4s   \n"
            "fsub   v23.4s, v23.4s, %5.4s   \n"
            "fsub   v24.4s, v24.4s, %5.4s   \n"
            "fsub   v25.4s, v25.4s, %5.4s   \n"
            "fsub   v26.4s, v26.4s, %5.4s   \n"
            "fsub   v27.4s, v27.4s, %5.4s   \n"
            "fcvtau v0.4s, v0.4s            \n" // float --> uint32
            "fcvtau v1.4s, v1.4s            \n"
            "fcvtau v2.4s, v2.4s            \n"
            "fcvtau v3.4s, v3.4s            \n"
            "fcvtau v4.4s, v4.4s            \n"
            "fcvtau v5.4s, v5.4s            \n"
            "fcvtau v6.4s, v6.4s            \n"
            "fcvtau v7.4s, v7.4s            \n"
            "fcvtau v20.4s, v20.4s          \n"
            "fcvtau v21.4s, v21.4s          \n"
            "fcvtau v22.4s, v22.4s          \n"
            "fcvtau v23.4s, v23.4s          \n"
            "fcvtau v24.4s, v24.4s          \n"
            "fcvtau v25.4s, v25.4s          \n"
            "fcvtau v26.4s, v26.4s          \n"
            "fcvtau v27.4s, v27.4s          \n"
            "uqxtn  v0.4h, v0.4s            \n" // uint32 --> uint16
            "uqxtn  v2.4h, v2.4s            \n"
            "uqxtn  v4.4h, v4.4s            \n"
            "uqxtn  v6.4h, v6.4s            \n"
            "uqxtn  v20.4h, v20.4s          \n"
            "uqxtn  v22.4h, v22.4s          \n"
            "uqxtn  v24.4h, v24.4s          \n"
            "uqxtn  v26.4h, v26.4s          \n"
            "uqxtn2 v0.8h, v1.4s            \n"
            "uqxtn2 v2.8h, v3.4s            \n"
            "uqxtn2 v4.8h, v5.4s            \n"
            "uqxtn2 v6.8h, v7.4s            \n"
            "uqxtn2 v20.8h, v21.4s          \n"
            "uqxtn2 v22.8h, v23.4s          \n"
            "uqxtn2 v24.8h, v25.4s          \n"
            "uqxtn2 v26.8h, v27.4s          \n"
            "st1    {v0.4s}, [%1]           \n" // save to tgt
            "add    %1, %1, %2              \n"
            "st1    {v2.4s}, [%1]           \n"
            "add    %1, %1, %2              \n"
            "st1    {v4.4s}, [%1]           \n"
            "add    %1, %1, %2              \n"
            "st1    {v6.4s}, [%1]           \n"
            "add    %1, %1, %2              \n"
            "st1    {v20.4s}, [%1]          \n"
            "add    %1, %1, %2              \n"
            "st1    {v22.4s}, [%1]          \n"
            "add    %1, %1, %2              \n"
            "st1    {v24.4s}, [%1]          \n"
            "add    %1, %1, %2              \n"
            "st1    {v26.4s}, [%1]          \n"
            : "=r"(src), "=r"(tgt), "=r"(_m), "=r"(_n), "=w"(vf32x4Scale), "=w"(vf32x4Offset)
            : "0"(src), "1"(tgt), "2"(_m), "3"(_n), "4"(vf32x4Scale), "5"(vf32x4Offset)
            : "cc",
              "memory",
              "v0",
              "v1",
              "v2",
              "v3",
              "v4",
              "v5",
              "v6",
              "v7",
              "v16",
              "v17",
              "v18",
              "v19",
              "v20",
              "v21",
              "v22",
              "v23",
              "v24",
              "v25",
              "v26",
              "v27");
    }

    void transpose_ufixed_dequant_4x4_neon(uint8_t* src,
                                           float* tgt,
                                           int m,
                                           int n,
                                           int offset,
                                           float scale) {
        long _m                  = sizeof(float) * m;
        long _n                  = sizeof(uint8_t) * n;
        float32x4_t vf32x4Scale  = vdupq_n_f32(scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile(
            "ldr    s0, [%0]            \n"
            "add    %0, %0, %3          \n"
            "ldr    s1, [%0]            \n"
            "add    %0, %0, %3          \n"
            "ldr    s2, [%0]            \n"
            "add    %0, %0, %3          \n"
            "ldr    s3, [%0]            \n"
            "uxtl   v0.8h, v0.8b        \n"
            "uxtl   v1.8h, v1.8b        \n"
            "uxtl   v2.8h, v2.8b        \n"
            "uxtl   v3.8h, v3.8b        \n"
            "uxtl   v0.4s, v0.4h        \n"
            "uxtl   v1.4s, v1.4h        \n"
            "uxtl   v2.4s, v2.4h        \n"
            "uxtl   v3.4s, v3.4h        \n"
            "ucvtf  v0.4s, v0.4s        \n"
            "ucvtf  v1.4s, v1.4s        \n"
            "ucvtf  v2.4s, v2.4s        \n"
            "ucvtf  v3.4s, v3.4s        \n"
            "fadd   v0.4s, v0.4s, %5.4s \n"
            "fadd   v1.4s, v1.4s, %5.4s \n"
            "fadd   v2.4s, v2.4s, %5.4s \n"
            "fadd   v3.4s, v3.4s, %5.4s \n"
            "fmul   v0.4s, v0.4s, %4.4s \n"
            "fmul   v1.4s, v1.4s, %4.4s \n"
            "fmul   v2.4s, v2.4s, %4.4s \n"
            "fmul   v3.4s, v3.4s, %4.4s \n"
            "trn1   v4.4s, v0.4s, v1.4s \n"
            "trn1   v5.4s, v2.4s, v3.4s \n"
            "trn2   v6.4s, v0.4s, v1.4s \n"
            "trn2   v7.4s, v2.4s, v3.4s \n"
            "trn1   v0.2d, v4.2d, v5.2d \n"
            "trn1   v1.2d, v6.2d, v7.2d \n"
            "trn2   v2.2d, v4.2d, v5.2d \n"
            "trn2   v3.2d, v6.2d, v7.2d \n"
            "st1    {v0.4s}, [%1]       \n"
            "add    %1, %1, %2          \n"
            "st1    {v1.4s}, [%1]       \n"
            "add    %1, %1, %2          \n"
            "st1    {v2.4s}, [%1]       \n"
            "add    %1, %1, %2          \n"
            "st1    {v3.4s}, [%1]       \n"
            : "=r"(src), "=r"(tgt), "=r"(_m), "=r"(_n), "=w"(vf32x4Scale), "=w"(vf32x4Offset)
            : "0"(src), "1"(tgt), "2"(_m), "3"(_n), "4"(vf32x4Scale), "5"(vf32x4Offset)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }

    void transpose_ufixed_dequant_4x4_neon(uint16_t* src,
                                           float* tgt,
                                           int m,
                                           int n,
                                           int offset,
                                           float scale) {
        long _m                  = sizeof(float) * m;
        long _n                  = sizeof(uint16_t) * n;
        float32x4_t vf32x4Scale  = vdupq_n_f32(scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile(
            "ldr    d0, [%0]            \n"
            "add    %0, %0, %3          \n"
            "ldr    d1, [%0]            \n"
            "add    %0, %0, %3          \n"
            "ldr    d2, [%0]            \n"
            "add    %0, %0, %3          \n"
            "ldr    d3, [%0]            \n"
            "uxtl   v0.4s, v0.4h        \n"
            "uxtl   v1.4s, v1.4h        \n"
            "uxtl   v2.4s, v2.4h        \n"
            "uxtl   v3.4s, v3.4h        \n"
            "ucvtf  v0.4s, v0.4s        \n"
            "ucvtf  v1.4s, v1.4s        \n"
            "ucvtf  v2.4s, v2.4s        \n"
            "ucvtf  v3.4s, v3.4s        \n"
            "fadd   v0.4s, v0.4s, %5.4s \n"
            "fadd   v1.4s, v1.4s, %5.4s \n"
            "fadd   v2.4s, v2.4s, %5.4s \n"
            "fadd   v3.4s, v3.4s, %5.4s \n"
            "fmul   v0.4s, v0.4s, %4.4s \n"
            "fmul   v1.4s, v1.4s, %4.4s \n"
            "fmul   v2.4s, v2.4s, %4.4s \n"
            "fmul   v3.4s, v3.4s, %4.4s \n"
            "trn1   v4.4s, v0.4s, v1.4s \n"
            "trn1   v5.4s, v2.4s, v3.4s \n"
            "trn2   v6.4s, v0.4s, v1.4s \n"
            "trn2   v7.4s, v2.4s, v3.4s \n"
            "trn1   v0.2d, v4.2d, v5.2d \n"
            "trn1   v1.2d, v6.2d, v7.2d \n"
            "trn2   v2.2d, v4.2d, v5.2d \n"
            "trn2   v3.2d, v6.2d, v7.2d \n"
            "st1    {v0.4s}, [%1]       \n"
            "add    %1, %1, %2          \n"
            "st1    {v1.4s}, [%1]       \n"
            "add    %1, %1, %2          \n"
            "st1    {v2.4s}, [%1]       \n"
            "add    %1, %1, %2          \n"
            "st1    {v3.4s}, [%1]       \n"
            : "=r"(src), "=r"(tgt), "=r"(_m), "=r"(_n), "=w"(vf32x4Scale), "=w"(vf32x4Offset)
            : "0"(src), "1"(tgt), "2"(_m), "3"(_n), "4"(vf32x4Scale), "5"(vf32x4Offset)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }

    void transpose_ufixed_dequant_8x8_neon(uint8_t* src,
                                           float* tgt,
                                           int m,
                                           int n,
                                           int offset,
                                           float scale) {
        long _m                  = sizeof(float) * m;
        long _n                  = sizeof(uint8_t) * n;
        float32x4_t vf32x4Scale  = vdupq_n_f32(scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile(
            "ldr    d0, [%0]                \n"
            "add    %0, %0, %3              \n"
            "ldr    d2, [%0]                \n"
            "add    %0, %0, %3              \n"
            "ldr    d4, [%0]                \n"
            "add    %0, %0, %3              \n"
            "ldr    d6, [%0]                \n"
            "add    %0, %0, %3              \n"
            "ldr    d20, [%0]               \n"
            "add    %0, %0, %3              \n"
            "ldr    d22, [%0]               \n"
            "add    %0, %0, %3              \n"
            "ldr    d24, [%0]               \n"
            "add    %0, %0, %3              \n"
            "ldr    d26, [%0]               \n"
            "uxtl   v0.8h, v0.8b            \n"
            "uxtl   v2.8h, v2.8b            \n"
            "uxtl   v4.8h, v4.8b            \n"
            "uxtl   v6.8h, v6.8b            \n"
            "uxtl   v20.8h, v20.8b          \n"
            "uxtl   v22.8h, v22.8b          \n"
            "uxtl   v24.8h, v24.8b          \n"
            "uxtl   v26.8h, v26.8b          \n"
            "uxtl2  v1.4s, v0.8h            \n"
            "uxtl2  v3.4s, v2.8h            \n"
            "uxtl2  v5.4s, v4.8h            \n"
            "uxtl2  v7.4s, v6.8h            \n"
            "uxtl2  v21.4s, v20.8h          \n"
            "uxtl2  v23.4s, v22.8h          \n"
            "uxtl2  v25.4s, v24.8h          \n"
            "uxtl2  v27.4s, v26.8h          \n"
            "uxtl   v0.4s, v0.4h            \n"
            "uxtl   v2.4s, v2.4h            \n"
            "uxtl   v4.4s, v4.4h            \n"
            "uxtl   v6.4s, v6.4h            \n"
            "uxtl   v20.4s, v20.4h          \n"
            "uxtl   v22.4s, v22.4h          \n"
            "uxtl   v24.4s, v24.4h          \n"
            "uxtl   v26.4s, v26.4h          \n"
            "ucvtf  v0.4s, v0.4s            \n"
            "ucvtf  v1.4s, v1.4s            \n"
            "ucvtf  v2.4s, v2.4s            \n"
            "ucvtf  v3.4s, v3.4s            \n"
            "ucvtf  v4.4s, v4.4s            \n"
            "ucvtf  v5.4s, v5.4s            \n"
            "ucvtf  v6.4s, v6.4s            \n"
            "ucvtf  v7.4s, v7.4s            \n"
            "ucvtf  v20.4s, v20.4s          \n"
            "ucvtf  v21.4s, v21.4s          \n"
            "ucvtf  v22.4s, v22.4s          \n"
            "ucvtf  v23.4s, v23.4s          \n"
            "ucvtf  v24.4s, v24.4s          \n"
            "ucvtf  v25.4s, v25.4s          \n"
            "ucvtf  v26.4s, v26.4s          \n"
            "ucvtf  v27.4s, v27.4s          \n"
            "fadd   v0.4s, v0.4s, %5.4s     \n"
            "fadd   v1.4s, v1.4s, %5.4s     \n"
            "fadd   v2.4s, v2.4s, %5.4s     \n"
            "fadd   v3.4s, v3.4s, %5.4s     \n"
            "fadd   v4.4s, v4.4s, %5.4s     \n"
            "fadd   v5.4s, v5.4s, %5.4s     \n"
            "fadd   v6.4s, v6.4s, %5.4s     \n"
            "fadd   v7.4s, v7.4s, %5.4s     \n"
            "fadd   v20.4s, v20.4s, %5.4s   \n"
            "fadd   v21.4s, v21.4s, %5.4s   \n"
            "fadd   v22.4s, v22.4s, %5.4s   \n"
            "fadd   v23.4s, v23.4s, %5.4s   \n"
            "fadd   v24.4s, v24.4s, %5.4s   \n"
            "fadd   v25.4s, v25.4s, %5.4s   \n"
            "fadd   v26.4s, v26.4s, %5.4s   \n"
            "fadd   v27.4s, v27.4s, %5.4s   \n"
            "fmul   v0.4s, v0.4s, %4.4s     \n"
            "fmul   v1.4s, v1.4s, %4.4s     \n"
            "fmul   v2.4s, v2.4s, %4.4s     \n"
            "fmul   v3.4s, v3.4s, %4.4s     \n"
            "fmul   v4.4s, v4.4s, %4.4s     \n"
            "fmul   v5.4s, v5.4s, %4.4s     \n"
            "fmul   v6.4s, v6.4s, %4.4s     \n"
            "fmul   v7.4s, v7.4s, %4.4s     \n"
            "fmul   v20.4s, v20.4s, %4.4s   \n"
            "fmul   v21.4s, v21.4s, %4.4s   \n"
            "fmul   v22.4s, v22.4s, %4.4s   \n"
            "fmul   v23.4s, v23.4s, %4.4s   \n"
            "fmul   v24.4s, v24.4s, %4.4s   \n"
            "fmul   v25.4s, v25.4s, %4.4s   \n"
            "fmul   v26.4s, v26.4s, %4.4s   \n"
            "fmul   v27.4s, v27.4s, %4.4s   \n"
            "trn1   v16.4s, v0.4s, v2.4s    \n"
            "trn1   v17.4s, v4.4s, v6.4s    \n"
            "trn2   v18.4s, v0.4s, v2.4s    \n"
            "trn2   v19.4s, v4.4s, v6.4s    \n"
            "trn1   v0.2d, v16.2d, v17.2d   \n"
            "trn1   v2.2d, v18.2d, v19.2d   \n"
            "trn2   v4.2d, v16.2d, v17.2d   \n"
            "trn2   v6.2d, v18.2d, v19.2d   \n"
            "mov    v16.4s, v1.4s           \n"
            "mov    v17.4s, v3.4s           \n"
            "mov    v18.4s, v5.4s           \n"
            "mov    v19.4s, v7.4s           \n"
            "trn1   v1.4s, v16.4s, v17.4s   \n"
            "trn1   v3.4s, v18.4s, v19.4s   \n"
            "trn2   v5.4s, v16.4s, v17.4s   \n"
            "trn2   v7.4s, v18.4s, v19.4s   \n"
            "trn1   v16.4s, v20.4s, v22.4s  \n"
            "trn1   v17.4s, v24.4s, v26.4s  \n"
            "trn2   v18.4s, v20.4s, v22.4s  \n"
            "trn2   v19.4s, v24.4s, v26.4s  \n"
            "trn1   v20.2d, v1.2d, v3.2d    \n"
            "trn1   v22.2d, v5.2d, v7.2d    \n"
            "trn2   v24.2d, v1.2d, v3.2d    \n"
            "trn2   v26.2d, v5.2d, v7.2d    \n"
            "trn1   v1.2d, v16.2d, v17.2d   \n"
            "trn1   v3.2d, v18.2d, v19.2d   \n"
            "trn2   v5.2d, v16.2d, v17.2d   \n"
            "trn2   v7.2d, v18.2d, v19.2d   \n"
            "trn1   v16.4s, v21.4s, v23.4s  \n"
            "trn1   v17.4s, v25.4s, v27.4s  \n"
            "trn2   v18.4s, v21.4s, v23.4s  \n"
            "trn2   v19.4s, v25.4s, v27.4s  \n"
            "trn1   v21.2d, v16.2d, v17.2d  \n"
            "trn1   v23.2d, v18.2d, v19.2d  \n"
            "trn2   v25.2d, v16.2d, v17.2d  \n"
            "trn2   v27.2d, v18.2d, v19.2d  \n"
            "st1    {v0.4s, v1.4s}, [%1]    \n"
            "add    %1, %1, %2              \n"
            "st1    {v2.4s, v3.4s}, [%1]    \n"
            "add    %1, %1, %2              \n"
            "st1    {v4.4s, v5.4s}, [%1]    \n"
            "add    %1, %1, %2              \n"
            "st1    {v6.4s, v7.4s}, [%1]    \n"
            "add    %1, %1, %2              \n"
            "st1    {v20.4s, v21.4s}, [%1]  \n"
            "add    %1, %1, %2              \n"
            "st1    {v22.4s, v23.4s}, [%1]  \n"
            "add    %1, %1, %2              \n"
            "st1    {v24.4s, v25.4s}, [%1]  \n"
            "add    %1, %1, %2              \n"
            "st1    {v26.4s, v27.4s}, [%1]  \n"
            : "=r"(src), "=r"(tgt), "=r"(_m), "=r"(_n), "=w"(vf32x4Scale), "=w"(vf32x4Offset)
            : "0"(src), "1"(tgt), "2"(_m), "3"(_n), "4"(vf32x4Scale), "5"(vf32x4Offset)
            : "cc",
              "memory",
              "v0",
              "v1",
              "v2",
              "v3",
              "v4",
              "v5",
              "v6",
              "v7",
              "v16",
              "v17",
              "v18",
              "v19",
              "v20",
              "v21",
              "v22",
              "v23",
              "v24",
              "v25",
              "v26",
              "v27");
    }

    void transpose_ufixed_dequant_8x8_neon(uint16_t* src,
                                           float* tgt,
                                           int m,
                                           int n,
                                           int offset,
                                           float scale) {
        long _m                  = sizeof(float) * m;
        long _n                  = sizeof(uint16_t) * n;
        float32x4_t vf32x4Scale  = vdupq_n_f32(scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile(
            "ld1    {v0.8h}, [%0]           \n"
            "add    %0, %0, %3              \n"
            "ld1    {v2.8h}, [%0]           \n"
            "add    %0, %0, %3              \n"
            "ld1    {v4.8h}, [%0]           \n"
            "add    %0, %0, %3              \n"
            "ld1    {v6.8h}, [%0]           \n"
            "add    %0, %0, %3              \n"
            "ld1    {v20.8h}, [%0]          \n"
            "add    %0, %0, %3              \n"
            "ld1    {v22.8h}, [%0]          \n"
            "add    %0, %0, %3              \n"
            "ld1    {v24.8h}, [%0]          \n"
            "add    %0, %0, %3              \n"
            "ld1    {v26.8h}, [%0]          \n"
            "uxtl2  v1.4s, v0.8h            \n"
            "uxtl2  v3.4s, v2.8h            \n"
            "uxtl2  v5.4s, v4.8h            \n"
            "uxtl2  v7.4s, v6.8h            \n"
            "uxtl2  v21.4s, v20.8h          \n"
            "uxtl2  v23.4s, v22.8h          \n"
            "uxtl2  v25.4s, v24.8h          \n"
            "uxtl2  v27.4s, v26.8h          \n"
            "uxtl   v0.4s, v0.4h            \n"
            "uxtl   v2.4s, v2.4h            \n"
            "uxtl   v4.4s, v4.4h            \n"
            "uxtl   v6.4s, v6.4h            \n"
            "uxtl   v20.4s, v20.4h          \n"
            "uxtl   v22.4s, v22.4h          \n"
            "uxtl   v24.4s, v24.4h          \n"
            "uxtl   v26.4s, v26.4h          \n"
            "ucvtf  v0.4s, v0.4s            \n"
            "ucvtf  v1.4s, v1.4s            \n"
            "ucvtf  v2.4s, v2.4s            \n"
            "ucvtf  v3.4s, v3.4s            \n"
            "ucvtf  v4.4s, v4.4s            \n"
            "ucvtf  v5.4s, v5.4s            \n"
            "ucvtf  v6.4s, v6.4s            \n"
            "ucvtf  v7.4s, v7.4s            \n"
            "ucvtf  v20.4s, v20.4s          \n"
            "ucvtf  v21.4s, v21.4s          \n"
            "ucvtf  v22.4s, v22.4s          \n"
            "ucvtf  v23.4s, v23.4s          \n"
            "ucvtf  v24.4s, v24.4s          \n"
            "ucvtf  v25.4s, v25.4s          \n"
            "ucvtf  v26.4s, v26.4s          \n"
            "ucvtf  v27.4s, v27.4s          \n"
            "fadd   v0.4s, v0.4s, %5.4s     \n"
            "fadd   v1.4s, v1.4s, %5.4s     \n"
            "fadd   v2.4s, v2.4s, %5.4s     \n"
            "fadd   v3.4s, v3.4s, %5.4s     \n"
            "fadd   v4.4s, v4.4s, %5.4s     \n"
            "fadd   v5.4s, v5.4s, %5.4s     \n"
            "fadd   v6.4s, v6.4s, %5.4s     \n"
            "fadd   v7.4s, v7.4s, %5.4s     \n"
            "fadd   v20.4s, v20.4s, %5.4s   \n"
            "fadd   v21.4s, v21.4s, %5.4s   \n"
            "fadd   v22.4s, v22.4s, %5.4s   \n"
            "fadd   v23.4s, v23.4s, %5.4s   \n"
            "fadd   v24.4s, v24.4s, %5.4s   \n"
            "fadd   v25.4s, v25.4s, %5.4s   \n"
            "fadd   v26.4s, v26.4s, %5.4s   \n"
            "fadd   v27.4s, v27.4s, %5.4s   \n"
            "fmul   v0.4s, v0.4s, %4.4s     \n"
            "fmul   v1.4s, v1.4s, %4.4s     \n"
            "fmul   v2.4s, v2.4s, %4.4s     \n"
            "fmul   v3.4s, v3.4s, %4.4s     \n"
            "fmul   v4.4s, v4.4s, %4.4s     \n"
            "fmul   v5.4s, v5.4s, %4.4s     \n"
            "fmul   v6.4s, v6.4s, %4.4s     \n"
            "fmul   v7.4s, v7.4s, %4.4s     \n"
            "fmul   v20.4s, v20.4s, %4.4s   \n"
            "fmul   v21.4s, v21.4s, %4.4s   \n"
            "fmul   v22.4s, v22.4s, %4.4s   \n"
            "fmul   v23.4s, v23.4s, %4.4s   \n"
            "fmul   v24.4s, v24.4s, %4.4s   \n"
            "fmul   v25.4s, v25.4s, %4.4s   \n"
            "fmul   v26.4s, v26.4s, %4.4s   \n"
            "fmul   v27.4s, v27.4s, %4.4s   \n"
            "trn1   v16.4s, v0.4s, v2.4s    \n"
            "trn1   v17.4s, v4.4s, v6.4s    \n"
            "trn2   v18.4s, v0.4s, v2.4s    \n"
            "trn2   v19.4s, v4.4s, v6.4s    \n"
            "trn1   v0.2d, v16.2d, v17.2d   \n"
            "trn1   v2.2d, v18.2d, v19.2d   \n"
            "trn2   v4.2d, v16.2d, v17.2d   \n"
            "trn2   v6.2d, v18.2d, v19.2d   \n"
            "mov    v16.4s, v1.4s           \n"
            "mov    v17.4s, v3.4s           \n"
            "mov    v18.4s, v5.4s           \n"
            "mov    v19.4s, v7.4s           \n"
            "trn1   v1.4s, v16.4s, v17.4s   \n"
            "trn1   v3.4s, v18.4s, v19.4s   \n"
            "trn2   v5.4s, v16.4s, v17.4s   \n"
            "trn2   v7.4s, v18.4s, v19.4s   \n"
            "trn1   v16.4s, v20.4s, v22.4s  \n"
            "trn1   v17.4s, v24.4s, v26.4s  \n"
            "trn2   v18.4s, v20.4s, v22.4s  \n"
            "trn2   v19.4s, v24.4s, v26.4s  \n"
            "trn1   v20.2d, v1.2d, v3.2d    \n"
            "trn1   v22.2d, v5.2d, v7.2d    \n"
            "trn2   v24.2d, v1.2d, v3.2d    \n"
            "trn2   v26.2d, v5.2d, v7.2d    \n"
            "trn1   v1.2d, v16.2d, v17.2d   \n"
            "trn1   v3.2d, v18.2d, v19.2d   \n"
            "trn2   v5.2d, v16.2d, v17.2d   \n"
            "trn2   v7.2d, v18.2d, v19.2d   \n"
            "trn1   v16.4s, v21.4s, v23.4s  \n"
            "trn1   v17.4s, v25.4s, v27.4s  \n"
            "trn2   v18.4s, v21.4s, v23.4s  \n"
            "trn2   v19.4s, v25.4s, v27.4s  \n"
            "trn1   v21.2d, v16.2d, v17.2d  \n"
            "trn1   v23.2d, v18.2d, v19.2d  \n"
            "trn2   v25.2d, v16.2d, v17.2d  \n"
            "trn2   v27.2d, v18.2d, v19.2d  \n"
            "st1    {v0.4s, v1.4s}, [%1]    \n"
            "add    %1, %1, %2              \n"
            "st1    {v2.4s, v3.4s}, [%1]    \n"
            "add    %1, %1, %2              \n"
            "st1    {v4.4s, v5.4s}, [%1]    \n"
            "add    %1, %1, %2              \n"
            "st1    {v6.4s, v7.4s}, [%1]    \n"
            "add    %1, %1, %2              \n"
            "st1    {v20.4s, v21.4s}, [%1]  \n"
            "add    %1, %1, %2              \n"
            "st1    {v22.4s, v23.4s}, [%1]  \n"
            "add    %1, %1, %2              \n"
            "st1    {v24.4s, v25.4s}, [%1]  \n"
            "add    %1, %1, %2              \n"
            "st1    {v26.4s, v27.4s}, [%1]  \n"
            : "=r"(src), "=r"(tgt), "=r"(_m), "=r"(_n), "=w"(vf32x4Scale), "=w"(vf32x4Offset)
            : "0"(src), "1"(tgt), "2"(_m), "3"(_n), "4"(vf32x4Scale), "5"(vf32x4Offset)
            : "cc",
              "memory",
              "v0",
              "v1",
              "v2",
              "v3",
              "v4",
              "v5",
              "v6",
              "v7",
              "v16",
              "v17",
              "v18",
              "v19",
              "v20",
              "v21",
              "v22",
              "v23",
              "v24",
              "v25",
              "v26",
              "v27");
    }

    void
    transpose_ufixed_dequant_Mx3_neon(uint8_t* src, float* tgt, int m, int offset, float scale) {
        uint8_t* src0 = src;
        float* tgt0   = tgt;
        float* tgt1   = tgt + m;
        float* tgt2   = tgt + 2 * m;

        int ii = m / 16;
        if (ii == 0) {
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    tgt[m * i + j] = static_cast<float>((src[3 * j + i] + offset) * scale);
                }
            }
            return;
        }

        float32x4_t vf32x4Scale  = vdupq_n_f32(scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile("0:                                 \n"
                     "ld3    {v0.16b-v2.16b}, [%0], #48  \n"
                     "uxtl   v16.8h, v0.8b               \n"
                     "uxtl   v20.8h, v1.8b               \n"
                     "uxtl   v24.8h, v2.8b               \n"
                     "uxtl2  v18.8h, v0.16b              \n"
                     "uxtl2  v22.8h, v1.16b              \n"
                     "uxtl2  v26.8h, v2.16b              \n"
                     "uxtl2  v17.4s, v16.8h              \n"
                     "uxtl2  v19.4s, v18.8h              \n"
                     "uxtl2  v21.4s, v20.8h              \n"
                     "uxtl2  v23.4s, v22.8h              \n"
                     "uxtl2  v25.4s, v24.8h              \n"
                     "uxtl2  v27.4s, v26.8h              \n"
                     "uxtl   v16.4s, v16.4h              \n"
                     "uxtl   v18.4s, v18.4h              \n"
                     "uxtl   v20.4s, v20.4h              \n"
                     "uxtl   v22.4s, v22.4h              \n"
                     "uxtl   v24.4s, v24.4h              \n"
                     "uxtl   v26.4s, v26.4h              \n"
                     "ucvtf  v16.4s, v16.4s              \n"
                     "ucvtf  v17.4s, v17.4s              \n"
                     "ucvtf  v18.4s, v18.4s              \n"
                     "ucvtf  v19.4s, v19.4s              \n"
                     "ucvtf  v20.4s, v20.4s              \n"
                     "ucvtf  v21.4s, v21.4s              \n"
                     "ucvtf  v22.4s, v22.4s              \n"
                     "ucvtf  v23.4s, v23.4s              \n"
                     "ucvtf  v24.4s, v24.4s              \n"
                     "ucvtf  v25.4s, v25.4s              \n"
                     "ucvtf  v26.4s, v26.4s              \n"
                     "ucvtf  v27.4s, v27.4s              \n"
                     "fadd   v16.4s, v16.4s, %6.4s       \n"
                     "fadd   v17.4s, v17.4s, %6.4s       \n"
                     "fadd   v18.4s, v18.4s, %6.4s       \n"
                     "fadd   v19.4s, v19.4s, %6.4s       \n"
                     "fadd   v20.4s, v20.4s, %6.4s       \n"
                     "fadd   v21.4s, v21.4s, %6.4s       \n"
                     "fadd   v22.4s, v22.4s, %6.4s       \n"
                     "fadd   v23.4s, v23.4s, %6.4s       \n"
                     "fadd   v24.4s, v24.4s, %6.4s       \n"
                     "fadd   v25.4s, v25.4s, %6.4s       \n"
                     "fadd   v26.4s, v26.4s, %6.4s       \n"
                     "fadd   v27.4s, v27.4s, %6.4s       \n"
                     "fmul   v16.4s, v16.4s, %5.4s       \n"
                     "fmul   v17.4s, v17.4s, %5.4s       \n"
                     "fmul   v18.4s, v18.4s, %5.4s       \n"
                     "fmul   v19.4s, v19.4s, %5.4s       \n"
                     "fmul   v20.4s, v20.4s, %5.4s       \n"
                     "fmul   v21.4s, v21.4s, %5.4s       \n"
                     "fmul   v22.4s, v22.4s, %5.4s       \n"
                     "fmul   v23.4s, v23.4s, %5.4s       \n"
                     "fmul   v24.4s, v24.4s, %5.4s       \n"
                     "fmul   v25.4s, v25.4s, %5.4s       \n"
                     "fmul   v26.4s, v26.4s, %5.4s       \n"
                     "fmul   v27.4s, v27.4s, %5.4s       \n"
                     "st1    {v16.4s-v19.4s}, [%1], 64   \n"
                     "st1    {v20.4s-v23.4s}, [%2], 64   \n"
                     "st1    {v24.4s-v27.4s}, [%3], 64   \n"
                     "subs   %w4, %w4, #1                \n"
                     "bgt    0b                          \n"
                     : "=r"(src0),
                       "=r"(tgt0),
                       "=r"(tgt1),
                       "=r"(tgt2),
                       "=r"(ii),
                       "=w"(vf32x4Scale),
                       "=w"(vf32x4Offset)
                     : "0"(src0),
                       "1"(tgt0),
                       "2"(tgt1),
                       "3"(tgt2),
                       "4"(ii),
                       "5"(vf32x4Scale),
                       "6"(vf32x4Offset)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22",
                       "v23",
                       "v24",
                       "v25",
                       "v26",
                       "v27");

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = m / 16 * 16; j < m; ++j) {
                tgt[m * i + j] = static_cast<float>((src[3 * j + i] + offset) * scale);
            }
        }
    }

    void
    transpose_ufixed_dequant_Mx3_neon(uint16_t* src, float* tgt, int m, int offset, float scale) {
        uint16_t* src0 = src;
        float* tgt0    = tgt;
        float* tgt1    = tgt + m;
        float* tgt2    = tgt + 2 * m;

        int ii = m / 16;
        if (ii == 0) {
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    tgt[m * i + j] = static_cast<float>((src[3 * j + i] + offset) * scale);
                }
            }
            return;
        }

        float32x4_t vf32x4Scale  = vdupq_n_f32(scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile("0:                                 \n"
                     "ld3    {v0.8h-v2.8h}, [%0], #48    \n"
                     "ld3    {v3.8h-v5.8h}, [%0], #48    \n"
                     "uxtl   v16.4s, v0.4h               \n"
                     "uxtl   v18.4s, v3.4h               \n"
                     "uxtl   v20.4s, v1.4h               \n"
                     "uxtl   v22.4s, v4.4h               \n"
                     "uxtl   v24.4s, v2.4h               \n"
                     "uxtl   v26.4s, v5.4h               \n"
                     "uxtl2  v17.4s, v0.8h               \n"
                     "uxtl2  v19.4s, v3.8h               \n"
                     "uxtl2  v21.4s, v1.8h               \n"
                     "uxtl2  v23.4s, v4.8h               \n"
                     "uxtl2  v25.4s, v2.8h               \n"
                     "uxtl2  v27.4s, v5.8h               \n"
                     "ucvtf  v16.4s, v16.4s              \n"
                     "ucvtf  v17.4s, v17.4s              \n"
                     "ucvtf  v18.4s, v18.4s              \n"
                     "ucvtf  v19.4s, v19.4s              \n"
                     "ucvtf  v20.4s, v20.4s              \n"
                     "ucvtf  v21.4s, v21.4s              \n"
                     "ucvtf  v22.4s, v22.4s              \n"
                     "ucvtf  v23.4s, v23.4s              \n"
                     "ucvtf  v24.4s, v24.4s              \n"
                     "ucvtf  v25.4s, v25.4s              \n"
                     "ucvtf  v26.4s, v26.4s              \n"
                     "ucvtf  v27.4s, v27.4s              \n"
                     "fadd   v16.4s, v16.4s, %6.4s       \n"
                     "fadd   v17.4s, v17.4s, %6.4s       \n"
                     "fadd   v18.4s, v18.4s, %6.4s       \n"
                     "fadd   v19.4s, v19.4s, %6.4s       \n"
                     "fadd   v20.4s, v20.4s, %6.4s       \n"
                     "fadd   v21.4s, v21.4s, %6.4s       \n"
                     "fadd   v22.4s, v22.4s, %6.4s       \n"
                     "fadd   v23.4s, v23.4s, %6.4s       \n"
                     "fadd   v24.4s, v24.4s, %6.4s       \n"
                     "fadd   v25.4s, v25.4s, %6.4s       \n"
                     "fadd   v26.4s, v26.4s, %6.4s       \n"
                     "fadd   v27.4s, v27.4s, %6.4s       \n"
                     "fmul   v16.4s, v16.4s, %5.4s       \n"
                     "fmul   v17.4s, v17.4s, %5.4s       \n"
                     "fmul   v18.4s, v18.4s, %5.4s       \n"
                     "fmul   v19.4s, v19.4s, %5.4s       \n"
                     "fmul   v20.4s, v20.4s, %5.4s       \n"
                     "fmul   v21.4s, v21.4s, %5.4s       \n"
                     "fmul   v22.4s, v22.4s, %5.4s       \n"
                     "fmul   v23.4s, v23.4s, %5.4s       \n"
                     "fmul   v24.4s, v24.4s, %5.4s       \n"
                     "fmul   v25.4s, v25.4s, %5.4s       \n"
                     "fmul   v26.4s, v26.4s, %5.4s       \n"
                     "fmul   v27.4s, v27.4s, %5.4s       \n"
                     "st1    {v16.4s-v19.4s}, [%1], 64   \n"
                     "st1    {v20.4s-v23.4s}, [%2], 64   \n"
                     "st1    {v24.4s-v27.4s}, [%3], 64   \n"
                     "subs   %w4, %w4, #1                \n"
                     "bgt    0b                          \n"
                     : "=r"(src0),
                       "=r"(tgt0),
                       "=r"(tgt1),
                       "=r"(tgt2),
                       "=r"(ii),
                       "=w"(vf32x4Scale),
                       "=w"(vf32x4Offset)
                     : "0"(src0),
                       "1"(tgt0),
                       "2"(tgt1),
                       "3"(tgt2),
                       "4"(ii),
                       "5"(vf32x4Scale),
                       "6"(vf32x4Offset)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22",
                       "v23",
                       "v24",
                       "v25",
                       "v26",
                       "v27");

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = m / 16 * 16; j < m; ++j) {
                tgt[m * i + j] = static_cast<float>((src[3 * j + i] + offset) * scale);
            }
        }
    }

    void
    transpose_ufixed_dequant_Mx4_neon(uint8_t* src, float* tgt, int m, int offset, float scale) {
        uint8_t* src0 = src;
        float* tgt0   = tgt;
        float* tgt1   = tgt + m;
        float* tgt2   = tgt + 2 * m;
        float* tgt3   = tgt + 3 * m;

        int ii = m / 16;
        if (ii == 0) {
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    tgt[m * i + j] = static_cast<float>((src[4 * j + i] + offset) * scale);
                }
            }
            return;
        }

        float32x4_t vf32x4Scale  = vdupq_n_f32(scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile("0:                                 \n"
                     "ld4    {v0.16b-v3.16b}, [%0], #64  \n"
                     "uxtl   v16.8h, v0.8b               \n"
                     "uxtl   v20.8h, v1.8b               \n"
                     "uxtl   v24.8h, v2.8b               \n"
                     "uxtl   v28.8h, v3.8b               \n"
                     "uxtl2  v18.8h, v0.16b              \n"
                     "uxtl2  v22.8h, v1.16b              \n"
                     "uxtl2  v26.8h, v2.16b              \n"
                     "uxtl2  v30.8h, v3.16b              \n"
                     "uxtl2  v17.4s, v16.8h              \n"
                     "uxtl2  v19.4s, v18.8h              \n"
                     "uxtl2  v21.4s, v20.8h              \n"
                     "uxtl2  v23.4s, v22.8h              \n"
                     "uxtl2  v25.4s, v24.8h              \n"
                     "uxtl2  v27.4s, v26.8h              \n"
                     "uxtl2  v29.4s, v28.8h              \n"
                     "uxtl2  v31.4s, v30.8h              \n"
                     "uxtl   v16.4s, v16.4h              \n"
                     "uxtl   v18.4s, v18.4h              \n"
                     "uxtl   v20.4s, v20.4h              \n"
                     "uxtl   v22.4s, v22.4h              \n"
                     "uxtl   v24.4s, v24.4h              \n"
                     "uxtl   v26.4s, v26.4h              \n"
                     "uxtl   v28.4s, v28.4h              \n"
                     "uxtl   v30.4s, v30.4h              \n"
                     "ucvtf  v16.4s, v16.4s              \n"
                     "ucvtf  v17.4s, v17.4s              \n"
                     "ucvtf  v18.4s, v18.4s              \n"
                     "ucvtf  v19.4s, v19.4s              \n"
                     "ucvtf  v20.4s, v20.4s              \n"
                     "ucvtf  v21.4s, v21.4s              \n"
                     "ucvtf  v22.4s, v22.4s              \n"
                     "ucvtf  v23.4s, v23.4s              \n"
                     "ucvtf  v24.4s, v24.4s              \n"
                     "ucvtf  v25.4s, v25.4s              \n"
                     "ucvtf  v26.4s, v26.4s              \n"
                     "ucvtf  v27.4s, v27.4s              \n"
                     "ucvtf  v28.4s, v28.4s              \n"
                     "ucvtf  v29.4s, v29.4s              \n"
                     "ucvtf  v30.4s, v30.4s              \n"
                     "ucvtf  v31.4s, v31.4s              \n"
                     "fadd   v16.4s, v16.4s, %7.4s       \n"
                     "fadd   v17.4s, v17.4s, %7.4s       \n"
                     "fadd   v18.4s, v18.4s, %7.4s       \n"
                     "fadd   v19.4s, v19.4s, %7.4s       \n"
                     "fadd   v20.4s, v20.4s, %7.4s       \n"
                     "fadd   v21.4s, v21.4s, %7.4s       \n"
                     "fadd   v22.4s, v22.4s, %7.4s       \n"
                     "fadd   v23.4s, v23.4s, %7.4s       \n"
                     "fadd   v24.4s, v24.4s, %7.4s       \n"
                     "fadd   v25.4s, v25.4s, %7.4s       \n"
                     "fadd   v26.4s, v26.4s, %7.4s       \n"
                     "fadd   v27.4s, v27.4s, %7.4s       \n"
                     "fadd   v28.4s, v28.4s, %7.4s       \n"
                     "fadd   v29.4s, v29.4s, %7.4s       \n"
                     "fadd   v30.4s, v30.4s, %7.4s       \n"
                     "fadd   v31.4s, v31.4s, %7.4s       \n"
                     "fmul   v16.4s, v16.4s, %6.4s       \n"
                     "fmul   v17.4s, v17.4s, %6.4s       \n"
                     "fmul   v18.4s, v18.4s, %6.4s       \n"
                     "fmul   v19.4s, v19.4s, %6.4s       \n"
                     "fmul   v20.4s, v20.4s, %6.4s       \n"
                     "fmul   v21.4s, v21.4s, %6.4s       \n"
                     "fmul   v22.4s, v22.4s, %6.4s       \n"
                     "fmul   v23.4s, v23.4s, %6.4s       \n"
                     "fmul   v24.4s, v24.4s, %6.4s       \n"
                     "fmul   v25.4s, v25.4s, %6.4s       \n"
                     "fmul   v26.4s, v26.4s, %6.4s       \n"
                     "fmul   v27.4s, v27.4s, %6.4s       \n"
                     "fmul   v28.4s, v28.4s, %6.4s       \n"
                     "fmul   v29.4s, v29.4s, %6.4s       \n"
                     "fmul   v30.4s, v30.4s, %6.4s       \n"
                     "fmul   v31.4s, v31.4s, %6.4s       \n"
                     "st1    {v16.4s-v19.4s}, [%1], 64   \n"
                     "st1    {v20.4s-v23.4s}, [%2], 64   \n"
                     "st1    {v24.4s-v27.4s}, [%3], 64   \n"
                     "st1    {v28.4s-v31.4s}, [%4], 64   \n"
                     "subs   %w5, %w5, #1                \n"
                     "bgt    0b                          \n"
                     : "=r"(src0),
                       "=r"(tgt0),
                       "=r"(tgt1),
                       "=r"(tgt2),
                       "=r"(tgt3),
                       "=r"(ii),
                       "=w"(vf32x4Scale),
                       "=w"(vf32x4Offset)
                     : "0"(src0),
                       "1"(tgt0),
                       "2"(tgt1),
                       "3"(tgt2),
                       "4"(tgt3),
                       "5"(ii),
                       "6"(vf32x4Scale),
                       "7"(vf32x4Offset)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22",
                       "v23",
                       "v24",
                       "v25",
                       "v26",
                       "v27",
                       "v28",
                       "v29",
                       "v30",
                       "v31");

        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = m / 16 * 16; j < m; ++j) {
                tgt[m * i + j] = static_cast<float>((src[4 * j + i] + offset) * scale);
            }
        }
    }

    void
    transpose_ufixed_dequant_Mx4_neon(uint16_t* src, float* tgt, int m, int offset, float scale) {
        uint16_t* src0 = src;
        float* tgt0    = tgt;
        float* tgt1    = tgt + m;
        float* tgt2    = tgt + 2 * m;
        float* tgt3    = tgt + 3 * m;

        int ii = m / 16;
        if (ii == 0) {
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    tgt[m * i + j] = static_cast<float>((src[4 * j + i] + offset) * scale);
                }
            }
            return;
        }

        float32x4_t vf32x4Scale  = vdupq_n_f32(scale);
        float32x4_t vf32x4Offset = vdupq_n_f32(offset + 0.f);
        asm volatile("0:                                 \n"
                     "ld4    {v0.8h-v3.8h}, [%0], #64    \n"
                     "ld4    {v4.8h-v7.8h}, [%0], #64    \n"
                     "uxtl   v16.4s, v0.4h               \n"
                     "uxtl   v18.4s, v4.4h               \n"
                     "uxtl   v20.4s, v1.4h               \n"
                     "uxtl   v22.4s, v5.4h               \n"
                     "uxtl   v24.4s, v2.4h               \n"
                     "uxtl   v26.4s, v6.4h               \n"
                     "uxtl   v28.4s, v3.4h               \n"
                     "uxtl   v30.4s, v7.4h               \n"
                     "uxtl2  v17.4s, v0.8h               \n"
                     "uxtl2  v19.4s, v4.8h               \n"
                     "uxtl2  v21.4s, v1.8h               \n"
                     "uxtl2  v23.4s, v5.8h               \n"
                     "uxtl2  v25.4s, v2.8h               \n"
                     "uxtl2  v27.4s, v6.8h               \n"
                     "uxtl2  v29.4s, v3.8h               \n"
                     "uxtl2  v31.4s, v7.8h               \n"
                     "ucvtf  v16.4s, v16.4s              \n"
                     "ucvtf  v17.4s, v17.4s              \n"
                     "ucvtf  v18.4s, v18.4s              \n"
                     "ucvtf  v19.4s, v19.4s              \n"
                     "ucvtf  v20.4s, v20.4s              \n"
                     "ucvtf  v21.4s, v21.4s              \n"
                     "ucvtf  v22.4s, v22.4s              \n"
                     "ucvtf  v23.4s, v23.4s              \n"
                     "ucvtf  v24.4s, v24.4s              \n"
                     "ucvtf  v25.4s, v25.4s              \n"
                     "ucvtf  v26.4s, v26.4s              \n"
                     "ucvtf  v27.4s, v27.4s              \n"
                     "ucvtf  v28.4s, v28.4s              \n"
                     "ucvtf  v29.4s, v29.4s              \n"
                     "ucvtf  v30.4s, v30.4s              \n"
                     "ucvtf  v31.4s, v31.4s              \n"
                     "fadd   v16.4s, v16.4s, %7.4s       \n"
                     "fadd   v17.4s, v17.4s, %7.4s       \n"
                     "fadd   v18.4s, v18.4s, %7.4s       \n"
                     "fadd   v19.4s, v19.4s, %7.4s       \n"
                     "fadd   v20.4s, v20.4s, %7.4s       \n"
                     "fadd   v21.4s, v21.4s, %7.4s       \n"
                     "fadd   v22.4s, v22.4s, %7.4s       \n"
                     "fadd   v23.4s, v23.4s, %7.4s       \n"
                     "fadd   v24.4s, v24.4s, %7.4s       \n"
                     "fadd   v25.4s, v25.4s, %7.4s       \n"
                     "fadd   v26.4s, v26.4s, %7.4s       \n"
                     "fadd   v27.4s, v27.4s, %7.4s       \n"
                     "fadd   v28.4s, v28.4s, %7.4s       \n"
                     "fadd   v29.4s, v29.4s, %7.4s       \n"
                     "fadd   v30.4s, v30.4s, %7.4s       \n"
                     "fadd   v31.4s, v31.4s, %7.4s       \n"
                     "fmul   v16.4s, v16.4s, %6.4s       \n"
                     "fmul   v17.4s, v17.4s, %6.4s       \n"
                     "fmul   v18.4s, v18.4s, %6.4s       \n"
                     "fmul   v19.4s, v19.4s, %6.4s       \n"
                     "fmul   v20.4s, v20.4s, %6.4s       \n"
                     "fmul   v21.4s, v21.4s, %6.4s       \n"
                     "fmul   v22.4s, v22.4s, %6.4s       \n"
                     "fmul   v23.4s, v23.4s, %6.4s       \n"
                     "fmul   v24.4s, v24.4s, %6.4s       \n"
                     "fmul   v25.4s, v25.4s, %6.4s       \n"
                     "fmul   v26.4s, v26.4s, %6.4s       \n"
                     "fmul   v27.4s, v27.4s, %6.4s       \n"
                     "fmul   v28.4s, v28.4s, %6.4s       \n"
                     "fmul   v29.4s, v29.4s, %6.4s       \n"
                     "fmul   v30.4s, v30.4s, %6.4s       \n"
                     "fmul   v31.4s, v31.4s, %6.4s       \n"
                     "st1    {v16.4s-v19.4s}, [%1], 64   \n"
                     "st1    {v20.4s-v23.4s}, [%2], 64   \n"
                     "st1    {v24.4s-v27.4s}, [%3], 64   \n"
                     "st1    {v28.4s-v31.4s}, [%4], 64   \n"
                     "subs   %w5, %w5, #1                \n"
                     "bgt    0b                          \n"
                     : "=r"(src0),
                       "=r"(tgt0),
                       "=r"(tgt1),
                       "=r"(tgt2),
                       "=r"(tgt3),
                       "=r"(ii),
                       "=w"(vf32x4Scale),
                       "=w"(vf32x4Offset)
                     : "0"(src0),
                       "1"(tgt0),
                       "2"(tgt1),
                       "3"(tgt2),
                       "4"(tgt3),
                       "5"(ii),
                       "6"(vf32x4Scale),
                       "7"(vf32x4Offset)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22",
                       "v23",
                       "v24",
                       "v25",
                       "v26",
                       "v27",
                       "v28",
                       "v29",
                       "v30",
                       "v31");

        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = m / 16 * 16; j < m; ++j) {
                tgt[m * i + j] = static_cast<float>((src[4 * j + i] + offset) * scale);
            }
        }
    }

#endif // (defined __ARM_NEON) && ((defined __arm64__) || (defined __aarch64__))
} // namespace wrap
} // namespace nn

#endif // SIMPLE_NN_QNN_UTILS_H_