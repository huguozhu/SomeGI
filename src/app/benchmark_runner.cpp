#include "app/benchmark_runner.h"
#include <cstdio>
#include <fstream>
#include <algorithm>

namespace somegi {

static const char* kGiNames[] = {
    "None", "IBL", "SSGI", "RSM", "LPV", "VXGI", "PRT",
    "DDGI", "GTGI", "SDFGI", "RT GI", "ReSTIR", "Lumen"
};
static const char* kAaNames[] = {"None", "MSAA", "TAA", "SMAA"};

void BenchmarkRunner::start(ApplyFn apply) {
    m_results.clear();
    m_gi = 0; m_aa = 0; m_ao = 0;
    m_running = true;
    m_collecting = false;
    m_timer = 0.0f;
    std::printf("[bench] starting — GI(0..12) x AA(0..3) x AO(0..2) = 156 tests\n");
    apply(m_gi, m_aa, m_ao);
}

void BenchmarkRunner::tick(float dt, float gpuMs, ApplyFn apply) {
    if (!m_running) return;

    // Cap dt to avoid waitIdle spikes skewing the timer
    m_timer += (dt > 0.1f ? 0.016f : dt);

    // Stabilization phase: wait for pipeline to settle
    if (!m_collecting) {
        if (m_timer >= 0.8f) {
            m_collecting = true;
            m_timer = 0.0f;
            m_frameCnt = 0;
            m_gpuSum = 0.0f;
        }
        return;
    }

    // Collection phase
    m_gpuSum += gpuMs;
    m_frameCnt++;

    if (m_timer >= 1.0f) {
        Result r{m_gi, m_aa, m_ao,
                 (float)m_frameCnt / m_timer,
                 m_gpuSum / (float)m_frameCnt};
        m_results.push_back(r);
        std::printf("[bench] GI=%2d AA=%d AO=%d  fps=%6.1f  gpu=%.2fms\n",
                    r.gi, r.aa, r.ao, r.fps, r.gpuMs);

        // Advance to next combination
        m_ao++;
        if (m_ao >= 3) { m_ao = 0; m_aa++; }
        if (m_aa >= 4) { m_aa = 0; m_gi++; }

        if (m_gi >= 13) {
            m_running = false;
            finish();
        } else {
            m_collecting = false;
            m_timer = 0.0f;
            apply(m_gi, m_aa, m_ao);
        }
    }
}

void BenchmarkRunner::finish() {
    std::printf("\n[bench] === Performance Matrix (fps / gpu ms) ===\n");
    std::printf("[bench] GI technique               | None     | MSAA     | TAA      | SMAA     |\n");
    std::printf("[bench] --------------------------- | -------- | -------- | -------- | -------- |\n");
    for (int gi = 0; gi < 13; ++gi) {
        std::printf("[bench] %-28s |", kGiNames[gi]);
        for (int aa = 0; aa < 4; ++aa) {
            float sumFps = 0, sumGpu = 0;
            int count = 0;
            for (auto& r : m_results) {
                if (r.gi == gi && r.aa == aa) { sumFps += r.fps; sumGpu += r.gpuMs; count++; }
            }
            if (count > 0)
                std::printf(" %4.0f/%4.2f |", sumFps / count, sumGpu / count);
            else
                std::printf(" %8s |", "—");
        }
        std::printf("\n");
    }
    std::printf("[bench] Done.\n");

    // Write detailed CSV
    {
        static const char* kAoNameCsv[] = {"None", "SSAO", "GTAO"};
        std::ofstream f("benchmark_results.csv");
        if (f) {
            f << "GI,AA,AO,FPS,GPU_ms\n";
            for (auto& r : m_results)
                f << kGiNames[r.gi] << "," << kAaNames[r.aa] << ","
                  << kAoNameCsv[r.ao] << "," << r.fps << "," << r.gpuMs << "\n";
            std::printf("[bench] Wrote benchmark_results.csv (%zu rows)\n", m_results.size());
        }
    }
    // Write matrix CSV
    {
        std::ofstream f("benchmark_matrix.csv");
        if (f) {
            f << "GI";
            for (int aa = 0; aa < 4; ++aa) f << "," << kAaNames[aa] << "_fps" << "," << kAaNames[aa] << "_gpu";
            f << "\n";
            for (int gi = 0; gi < 13; ++gi) {
                f << kGiNames[gi];
                for (int aa = 0; aa < 4; ++aa) {
                    float sumFps = 0, sumGpu = 0;
                    int count = 0;
                    for (auto& r : m_results) {
                        if (r.gi == gi && r.aa == aa) { sumFps += r.fps; sumGpu += r.gpuMs; count++; }
                    }
                    if (count > 0) f << "," << sumFps / count << "," << sumGpu / count;
                    else           f << ",,";
                }
                f << "\n";
            }
            std::printf("[bench] Wrote benchmark_matrix.csv\n");
        }
    }
}

} // namespace somegi
