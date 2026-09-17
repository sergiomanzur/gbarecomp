#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define RECOMP_AUDIO_DRC_IMPL
#include "recomp_audio_drc.h"

int main() {
    rab_config cfg;
    rab_config_defaults(&cfg);
    cfg.channels = 1;
    cfg.source_rate = 48000.0;
    cfg.host_rate = 48000.0;
    cfg.target_ms = 10.0;
    cfg.ring_ms = 100.0;
    cfg.preroll_ms = 10.0;
    cfg.stretch_limit_ms = 20.0;

    rab_bridge bridge{};
    if (rab_init(&bridge, &cfg) != 0) {
        std::fprintf(stderr, "rab_init failed\n");
        return 1;
    }

    std::vector<int16_t> input(2400);
    for (std::size_t i = 0; i < input.size(); ++i) {
        constexpr double kPi = 3.14159265358979323846;
        input[i] = static_cast<int16_t>(
            std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) /
                     cfg.source_rate) *
            12000.0);
    }
    rab_push(&bridge, input.data(), static_cast<int>(input.size()));

    std::vector<int16_t> output(6000);
    rab_pull(&bridge, output.data(), static_cast<int>(output.size()));

    rab_stats stats{};
    rab_get_stats(&bridge, &stats);
    const uint64_t limit_frames =
        static_cast<uint64_t>(cfg.stretch_limit_ms * cfg.host_rate / 1000.0);
    if (stats.stretch_frames == 0 || stats.stretch_frames > limit_frames) {
        std::fprintf(stderr,
                     "stall concealment exceeded limit: stretch=%llu limit=%llu\n",
                     static_cast<unsigned long long>(stats.stretch_frames),
                     static_cast<unsigned long long>(limit_frames));
        rab_free(&bridge);
        return 2;
    }

    const auto tail_begin = output.end() - 256;
    const int16_t tail_peak = *std::max_element(
        tail_begin, output.end(),
        [](int16_t a, int16_t b) { return std::abs(a) < std::abs(b); });
    if (std::abs(static_cast<int>(tail_peak)) > 1) {
        std::fprintf(stderr, "stalled output did not fade to silence: peak=%d\n",
                     static_cast<int>(tail_peak));
        rab_free(&bridge);
        return 3;
    }

    rab_free(&bridge);

    // --- Trim scenario: the one-shot preroll-surplus trim (recomp_audio_drc.h).
    // preroll_ms is set comfortably above target_ms + 20 so the trim's fill_ms
    // guard actually fires, and enough controller updates run before the trim
    // for b->corr to saturate at +max_correction first -- the same way the
    // real 250ms/60ms preroll/target gap pins the servo at +0.500% for
    // 30-45s on real hardware. That makes this the scenario the anti-windup
    // reset (b->err_lp = b->corr = 0.0 at the trim) exists for: without it,
    // this saturated correction keeps consuming samples too fast right after
    // the trim and corr stays pinned near +0.5% for ~26 iterations instead of
    // dropping back to ~0, which check (d) below catches.
    {
        rab_config cfg2;
        rab_config_defaults(&cfg2);
        cfg2.channels      = 1;
        cfg2.source_rate   = 48000.0;
        cfg2.host_rate     = 48000.0;
        cfg2.target_ms     = 10.0;
        cfg2.preroll_ms    = 60.0;   // > target_ms + 20 (30) so the fill_ms guard fires
        cfg2.trim_after_ms = 100.0;  // short wait so the test runs in ~150ms of simulated audio

        rab_bridge bridge2{};
        if (rab_init(&bridge2, &cfg2) != 0) {
            std::fprintf(stderr, "rab_init (trim scenario) failed\n");
            return 4;
        }

        // Prime immediately: push well past preroll_ms so the very first pull
        // observes b->primed == 1 and starts accumulating elapsed time.
        std::vector<int16_t> prime_in(6000, 0);
        rab_push(&bridge2, prime_in.data(), static_cast<int>(prime_in.size()));

        // 1 frame/call so any consumption that happens *after* the trim but
        // still inside the same rab_pull() call (the per-frame resample loop
        // keeps running once the one-shot block above it has moved out_pos)
        // is negligible against the 1.0ms tolerance in check (b) below --
        // with a coarser block, that same-call trailing consumption is what
        // fill_at_trim would also include, understating the landing point by
        // up to one whole block.
        constexpr int kBlock = 1;
        std::vector<int16_t> block_in(kBlock, 0);
        std::vector<int16_t> block_out(kBlock);

        bool saw_trim = false;
        int trim_iter = -1;
        double fill_at_trim = 0.0;
        int iters_since_trim = 0;
        // >> the ~56-call window max_correction needs to saturate, and the
        // matching ~26-call window it needs to release, at this kp/slew.
        constexpr int kPostTrimIters = 200;
        // trim_after_ms=100ms at 1 frame/48kHz call needs ~4800 calls; this
        // is a safety net against an infinite loop, not a tuned expectation.
        constexpr int kMaxIters = 20000;

        for (int i = 0; i < kMaxIters; ++i) {
            // Push at the same rate we pull so fill stays roughly flat
            // pre-trim (mirrors a steady producer) instead of draining from
            // the one big initial push as the loop runs.
            rab_push(&bridge2, block_in.data(), kBlock);
            rab_pull(&bridge2, block_out.data(), kBlock);

            rab_stats st{};
            rab_get_stats(&bridge2, &st);

            // (c) the read cursor must never run ahead of what's been pushed.
            const double raw_fill =
                static_cast<double>(bridge2.in_count) - bridge2.out_pos;
            if (raw_fill < -1e-6) {
                std::fprintf(stderr,
                             "trim scenario: out_pos exceeded in_count at iter %d "
                             "(in_count=%llu out_pos=%.3f)\n",
                             i, static_cast<unsigned long long>(bridge2.in_count),
                             bridge2.out_pos);
                rab_free(&bridge2);
                return 5;
            }

            if (!saw_trim && st.trim_events == 1) {
                saw_trim = true;
                trim_iter = i;
                fill_at_trim = rab_fill_ms(&bridge2);
            } else if (saw_trim && st.trim_events != 1) {
                // (a) exactly one trim, ever -- must never re-trigger.
                std::fprintf(stderr,
                             "trim scenario: trim_events changed after the first "
                             "trim (now %llu) at iter %d\n",
                             static_cast<unsigned long long>(st.trim_events), i);
                rab_free(&bridge2);
                return 6;
            }

            if (saw_trim) {
                // (d) with err_lp/corr reset alongside the out_pos jump, corr
                // must not swing outside +/-0.2% while it settles. Skip the
                // trim's own iteration: stats.last_correction for that call
                // was captured by rab__update_controller() at the TOP of
                // rab_pull(), before this same call's trim resets b->corr --
                // it reports the old saturated value for that one already-
                // consumed frame (immaterial -- one host frame of audio), and
                // every call from here on reflects the reset state.
                if (i > trim_iter) {
                    const double corr_pct = st.last_correction * 100.0;
                    if (corr_pct > 0.2 || corr_pct < -0.2) {
                        std::fprintf(stderr,
                                     "trim scenario: corr=%.3f%% outside +/-0.2%% "
                                     "post-trim at iter %d\n",
                                     corr_pct, i);
                        rab_free(&bridge2);
                        return 7;
                    }
                }
                if (++iters_since_trim >= kPostTrimIters) break;
            }
        }

        if (!saw_trim) {
            std::fprintf(stderr, "trim scenario: trim never fired\n");
            rab_free(&bridge2);
            return 8;
        }

        // (b) the trim must land the fill at target_ms, not just below the
        // fill_ms > target_ms + 20 guard. 1.0ms tolerance matches the
        // controller's own deadband_ms (the smallest fill error the servo
        // itself treats as "at target").
        const double target_ms = cfg2.target_ms;
        if (fill_at_trim < target_ms - 1.0 || fill_at_trim > target_ms + 1.0) {
            std::fprintf(stderr,
                         "trim scenario: fill_ms=%.3f did not land at target_ms=%.1f "
                         "(+/-1.0ms) at trim (iter %d)\n",
                         fill_at_trim, target_ms, trim_iter);
            rab_free(&bridge2);
            return 9;
        }

        rab_stats final_stats{};
        rab_get_stats(&bridge2, &final_stats);
        if (final_stats.underrun_events != 0 || final_stats.overflow_drops != 0) {
            std::fprintf(stderr,
                         "trim scenario: unexpected underrun=%llu overflow_drops=%llu\n",
                         static_cast<unsigned long long>(final_stats.underrun_events),
                         static_cast<unsigned long long>(final_stats.overflow_drops));
            rab_free(&bridge2);
            return 10;
        }

        rab_free(&bridge2);
    }

    return 0;
}
