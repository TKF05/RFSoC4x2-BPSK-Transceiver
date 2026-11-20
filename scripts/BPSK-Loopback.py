
import numpy as np
import casperfpga
import sys
import matplotlib.pyplot as plt
import time


# Parameters
fs = 1966.08e6
fc = 131.027e6

BRAM_DEPTH = 8192
SAMPLES_PER_WORD = 8
TOTAL_SAMPLES = BRAM_DEPTH * SAMPLES_PER_WORD

SYMBOL_SAMPLES = 12 # Strange behavior caused by this one 

NUM_BITS = 1024

AMPLITUDE = 64

SRRC_ROLLOFF = 0.25
SRRC_SPAN = 10
# AMPLITUDE = 40

## Preamble and Data generation
PREAMBLE = np.array(
    [
        1, 1, 0, 1, 0, 0, 1, 0,
        1, 0, 1, 1, 1, 0, 0, 0,
        0, 1, 1, 0, 0, 1, 0, 1,
        1, 1, 0, 0, 1, 0, 1, 0
    ],
    dtype=np.uint8
)

PREAMBLE_LENGTH = len(PREAMBLE)

rng = np.random.default_rng()

bits = rng.integers(
    0, 2, NUM_BITS, dtype=np.uint8
)

frame_bits = np.concatenate((PREAMBLE, bits))

print("\n========== PARAMETERS ==========")
print("Carrier frequency:", fc / 1e6, "MHz")
print("Sample rate:", fs / 1e6, "MHz")
print("Samples per symbol:", SYMBOL_SAMPLES)
print("Preamble bits:", len(PREAMBLE))
print("Payload bits:", len(bits))
print("Total frame bits:", len(frame_bits))

def srrc_filter(alpha, sps, span):

    t = np.arange(
        -span * sps // 2,
        span * sps // 2 + 1
    ) / sps

    h = np.zeros(len(t))

    for i, ti in enumerate(t):

        if np.isclose(ti, 0.0):

            h[i] = (
                1 - alpha
                + 4 * alpha / np.pi
            )

        elif alpha != 0 and np.isclose(
            abs(ti), 1 / (4 * alpha)
        ):

            h[i] = (
                alpha / np.sqrt(2)
                * (
                    (1 + 2 / np.pi)
                    * np.sin(np.pi / (4 * alpha))
                    +
                    (1 - 2 / np.pi)
                    * np.cos(np.pi / (4 * alpha))
                )
            )

        else:

            h[i] = (
                np.sin(np.pi * ti * (1 - alpha))
                +
                4 * alpha * ti
                * np.cos(np.pi * ti * (1 + alpha))
            ) / (
                np.pi * ti
                * (1 - (4 * alpha * ti)**2)
            )

    # normalize energy
    h /= np.sqrt(np.sum(h**2))

    return h

def bpsk_modulate(bits, fs, fc, sps, amplitude):

    # Map bits to BPSK symbols:
    # 0 -> +1
    # 1 -> -1

    symbols = 1.0 - 2.0 * bits.astype(np.float64)

    # SRRC parameters
    # rolloff = 0.25
    rolloff = SRRC_ROLLOFF
    # span = 10
    span = SRRC_SPAN
    h = srrc_filter(rolloff, sps, span)

    upsampled = np.zeros(
        len(symbols) * sps,
        dtype=np.float64
    )
    upsampled[::sps] = symbols

    #Pulse shaping filter
    symbol_waveform = np.convolve(upsampled, h, mode='full')

    # Continuous carrier
    n = np.arange(len(symbol_waveform))

    carrier = np.cos(2 * np.pi * fc / fs * n)

    # Direct BPSK modulation
    waveform = amplitude * symbol_waveform * carrier

    return np.round(waveform).astype(np.int16)


tx_frame = bpsk_modulate(frame_bits, fs, fc, SYMBOL_SAMPLES, AMPLITUDE)


## Transmitrepeat

gap_samples = 100

repeats = int(np.ceil(
    TOTAL_SAMPLES / (len(tx_frame) + gap_samples)
))

frames = []

for i in range(repeats):
    frames.append(tx_frame)
    frames.append(np.zeros(gap_samples, dtype=np.int16))

tx_samples = np.concatenate(frames)[:TOTAL_SAMPLES]

# repeats = int(
#     np.ceil(TOTAL_SAMPLES / len(tx_frame))
# )
#
# tx_samples = np.tile(
#     tx_frame,
#     repeats
# )[:TOTAL_SAMPLES]

buf = tx_samples.astype('<i2').tobytes()

print("\n========== TX BRAM ==========")
print("Frame samples:", len(tx_frame))
print("Repetitions:", repeats)
print("Total samples:", len(tx_samples))
print("Buffer size:", len(buf))

assert len(buf) == 131072

fpga.write('tx_bram', buf)

print("TX BRAM written successfully")
time.sleep(2)

## RX CHAIN

print("\n========== RX IQ SNAPSHOT ==========")

# Each snapshot word contains four signed 16-bit samples.
# THIS SECTION FROM RFDC TUTORIAL
def to_signed(v, bits=16):
    mask = 1 << (bits - 1)
    return -(v & mask) | (v & (mask - 1))


def deinterleave_component(words, bits=16):
    samples = []

    for word in words:
        samples.extend(
            to_signed((word >> shift) & 0xffff, bits)
            for shift in (0, 16, 32, 48)
        )

    return np.asarray(samples, dtype=np.float64)


#take snapshot
for snapshot in fpga.snapshots:
    snapshot.arm()

fpga.registers.snapshot_ctrl.write(start='pulse')

adc_data = {}
for snapshot in fpga.snapshots:
    adc_data.update(snapshot.read(arm=False)['data'])

# ADC SAMPLE RATE 3932.16 decimated 4x is half of DAC fs
RX_FS = fs / 2.0
RX_MIX_FC = fc
RX_FC = 0.0  # residual carrier offset after the digital downconverter
RX_SYMBOL_SAMPLES = int(round(SYMBOL_SAMPLES * RX_FS / fs))
if RX_SYMBOL_SAMPLES < 1:
    raise ValueError("Invalid RX samples-per-symbol setting")

#Symbol Timing Synchronization:
# fractional: fractional sample value interpolation
# fixed: fixed integrer values 
# gardner: gardner TED (Non-Data-aided, independent of phase offset)
TIMING_MODE = "gardner"
#Packet BER rejection threshold
MAX_PACKET_BER = 0.10

x_real = deinterleave_component(adc_data['m00'])
raw_dc_offset = np.mean(x_real)
x_real -= raw_dc_offset
sample_n = np.arange(len(x_real), dtype=np.float64)
lo = np.exp(-1j * 2 * np.pi * RX_MIX_FC / RX_FS * sample_n)
x = 2.0 * x_real * lo

# SRRC matched filter to the baseband signal.

rx_filter = srrc_filter(SRRC_ROLLOFF, RX_SYMBOL_SAMPLES, SRRC_SPAN)
rx_matched = np.convolve(x, rx_filter, mode='full')

print("RX input mode: real passband from m00")
print("Ignored m01 peak:", max(abs(v) for v in adc_data['m01']))
print("Captured real samples:", len(x_real))
print("Raw DC offset:", raw_dc_offset)
print("RX sample rate:", RX_FS / 1e6, "MHz")
print("RX carrier downconversion:", RX_MIX_FC / 1e6, "MHz")
print("RX samples per symbol:", RX_SYMBOL_SAMPLES)
print("RX baseband peak:", np.max(np.abs(x)))


# BPSK soft decision
def bpsk_soft_demodulate(samples, fs, fc, sps, offset):

    # The input has already passed through the receive SRRC matched filter.
    # Sample once at each candidate symbol instant instead of integrating a
    # whole symbol window.
    sample_indices = np.arange(offset, len(samples), sps)
    sample_indices = sample_indices[sample_indices < len(samples)]

    # Preserve absolute sample phase when correcting any residual carrier.
    oscillator = np.exp(
        -1j * 2 * np.pi * fc / fs * sample_indices
    )

    return samples[sample_indices] * oscillator



#cubic interpolation courtesy of chatgpt
def interpolate_complex(samples, sample_time):
    # Cubic Lagrange interpolation using four neighboring samples.
    # Return zero outside the captured interval.
    if sample_time < 1 or sample_time >= len(samples) - 2:
        return 0.0j

    index = int(np.floor(sample_time))
    mu = sample_time - index

    y0 = samples[index - 1]
    y1 = samples[index]
    y2 = samples[index + 1]
    y3 = samples[index + 2]

    # Cubic interpolation coefficients
    c0 = -mu * (mu - 1) * (mu - 2) / 6.0
    c1 = (mu + 1) * (mu - 1) * (mu - 2) / 2.0
    c2 = -(mu + 1) * mu * (mu - 2) / 2.0
    c3 = (mu + 1) * mu * (mu - 1) / 6.0

    return c0 * y0 + c1 * y1 + c2 * y2 + c3 * y3

# Linear Interpolation
# def interpolate_complex(samples, sample_time):
#
#     if sample_time < 0 or sample_time >= len(samples) - 1:
#         return 0.0j
#
#     index = int(np.floor(sample_time))
#     fraction = sample_time - index
#
#     return (
#         samples[index] * (1.0 - fraction)
#         + samples[index + 1] * fraction
#     )

#Chatgpt fractional TED DEPRECATED
# def estimate_fractional_timing(
#     samples,
#     coarse_start,
#     nominal_sps,
#     reference_symbols,
#     rate_search_fraction=0.02,
#     rate_steps=81,
#     start_search_radius=0.75,
#     start_steps=31,
#     refinement_rounds=3,
#     refinement_steps=21
# ):
#     """Fit fractional start time and symbol period to a known preamble.
#
#     Run a coarse grid search, then refine locally so the final samples/symbol
#     estimate is much finer than the initial grid spacing.
#     """
#
#     samples = np.asarray(samples, dtype=np.complex128)
#     reference_symbols = np.asarray(reference_symbols, dtype=np.float64)
#     symbol_numbers = np.arange(len(reference_symbols), dtype=np.float64)
#     eps = 1e-12
#
#     min_sps = nominal_sps * (1.0 - rate_search_fraction)
#     max_sps = nominal_sps * (1.0 + rate_search_fraction)
#     min_start = coarse_start - start_search_radius
#     max_start = coarse_start + start_search_radius
#
#     def score_candidate(candidate_start, candidate_sps):
#         positions = candidate_start + candidate_sps * symbol_numbers
#
#         if positions[0] < 0 or positions[-1] >= len(samples) - 1:
#             return None
#
#         indices = np.floor(positions).astype(np.int64)
#         fractions = positions - indices
#
#         values = (
#             samples[indices] * (1.0 - fractions)
#             + samples[indices + 1] * fractions
#         )
#
#         correlation = abs(np.sum(values * reference_symbols))
#         energy = np.sum(np.abs(values) ** 2)
#
#         return float(
#             correlation / np.sqrt(len(reference_symbols) * energy + eps)
#         )
#
#     def search(rate_candidates, start_candidates, best_start, best_sps, best_metric):
#         for candidate_sps in rate_candidates:
#             for candidate_start in start_candidates:
#                 metric = score_candidate(candidate_start, candidate_sps)
#
#                 if metric is not None and metric > best_metric:
#                     best_metric = metric
#                     best_start = float(candidate_start)
#                     best_sps = float(candidate_sps)
#
#         return best_start, best_sps, best_metric
#
#     # Coarse search.
#     rate_candidates = np.linspace(min_sps, max_sps, rate_steps)
#     start_candidates = np.linspace(min_start, max_start, start_steps)
#
#     best_start = float(coarse_start)
#     best_sps = float(nominal_sps)
#     best_metric = -1.0
#
#     best_start, best_sps, best_metric = search(
#         rate_candidates,
#         start_candidates,
#         best_start,
#         best_sps,
#         best_metric
#     )
#
#     # Refine around the best coarse candidate.
#     rate_step = rate_candidates[1] - rate_candidates[0]
#     start_step = start_candidates[1] - start_candidates[0]
#
#     for _ in range(refinement_rounds):
#         rate_candidates = np.linspace(
#             max(min_sps, best_sps - rate_step),
#             min(max_sps, best_sps + rate_step),
#             refinement_steps
#         )
#         start_candidates = np.linspace(
#             max(min_start, best_start - start_step),
#             min(max_start, best_start + start_step),
#             refinement_steps
#         )
#
#         best_start, best_sps, best_metric = search(
#             rate_candidates,
#             start_candidates,
#             best_start,
#             best_sps,
#             best_metric
#         )
#
#         rate_step = rate_candidates[1] - rate_candidates[0]
#         start_step = start_candidates[1] - start_candidates[0]
#
#     return best_start, best_sps, best_metric


def gardner_timing_recovery(
    samples,
    nominal_sps,
    start_sample,
    alpha=0.05,
    beta=0.0005,
    max_rate_error=0.02
):

    """Recover symbol timing with a normalized Gardner TED and linear interpolator.

    `start_sample` is the coarse integer-sample timing estimate from the known
    preamble. The loop then adjusts both fractional timing and samples per
    symbol while walking through the matched-filter output.
    """

    nominal_sps = float(nominal_sps)
    omega = nominal_sps
    omega_min = nominal_sps * (1.0 - max_rate_error)
    omega_max = nominal_sps * (1.0 + max_rate_error)
    sample_time = float(start_sample)

    recovered = []
    timing_errors = []
    timing_positions = []
    omega_history = []

    while (
        sample_time - omega / 2.0 >= 0
        and sample_time + omega / 2.0 < len(samples) - 1
    ):

        on_time = interpolate_complex(samples, sample_time)
        early = interpolate_complex(samples, sample_time - omega / 2.0)
        late = interpolate_complex(samples, sample_time + omega / 2.0)

        #  negative error means the current sample
        # is late, so the correction moves the next sample earlier.
        scale = (
            abs(on_time) ** 2
            + 0.5 * (abs(early) ** 2 + abs(late) ** 2)
            + 1e-12
        )
        error = float(
            np.real((late - early) * np.conj(on_time)) / scale
        )

        recovered.append(on_time)
        timing_errors.append(error)
        timing_positions.append(sample_time)
        omega_history.append(omega)

        omega = float(np.clip(
            omega + beta * error,
            omega_min,
            omega_max
        ))
        sample_time += omega + alpha * error

    return {
        'symbols': np.asarray(recovered, dtype=np.complex128),
        'errors': np.asarray(timing_errors, dtype=np.float64),
        'positions': np.asarray(timing_positions, dtype=np.float64),
        'omega': np.asarray(omega_history, dtype=np.float64)
    }


## PREAMBLE DETECTION

def find_preamble_candidates(
    samples,
    fs,
    fc,
    sps,
    preamble,
    min_confidence=0.65,
    min_separation_samples=None,
    max_candidates=32
):
    preamble_len = len(preamble)
    preamble_symbols = 1.0 - 2.0 * preamble.astype(np.float64)
    candidates = []
    eps = 1e-12

    for offset in range(sps):
        symbols = bpsk_soft_demodulate(samples, fs, fc, sps, offset)
        if len(symbols) < preamble_len:
            continue

        correlation = np.correlate(symbols, preamble_symbols, mode='valid')
        symbol_energy = np.abs(symbols) ** 2
        window_energy = np.convolve(
            symbol_energy, np.ones(preamble_len), mode='valid'
        )
        metric = np.abs(correlation) / np.sqrt(
            preamble_len * window_energy + eps
        )

        if len(metric) < 3:
            continue

        peaks = np.flatnonzero(
            (metric[1:-1] >= metric[:-2])
            & (metric[1:-1] >= metric[2:])
            & (metric[1:-1] >= min_confidence)
        ) + 1

        for peak in peaks:
            candidates.append({
                'offset': offset,
                'symbol_index': int(peak),
                'start_sample': int(offset + peak * sps),
                'symbols': symbols,
                'correlation': correlation,
                'metric': metric,
                'phase': float(np.angle(correlation[peak])),
                'confidence': float(metric[peak])
            })

    if min_separation_samples is None:
        min_separation_samples = sps * preamble_len

    candidates.sort(key=lambda item: item['confidence'], reverse=True)
    selected = []
    for candidate in candidates:
        if all(
            abs(candidate['start_sample'] - chosen['start_sample'])
            >= min_separation_samples
            for chosen in selected
        ):
            selected.append(candidate)
            if len(selected) >= max_candidates:
                break

    return sorted(selected, key=lambda item: item['start_sample'])


print("\n========== PACKET SEARCH ==========")

symbols_needed = PREAMBLE_LENGTH + NUM_BITS
preamble_ref = 1.0 - 2.0 * PREAMBLE.astype(np.float64)
packet_period_samples = len(tx_frame) * RX_FS / fs
candidate_separation = 0.5 * packet_period_samples
packet_candidates = find_preamble_candidates(
    rx_matched,
    RX_FS,
    RX_FC,
    RX_SYMBOL_SAMPLES,
    PREAMBLE,
    min_confidence=0.65,
    min_separation_samples=candidate_separation,
    max_candidates=32
)

if not packet_candidates:
    raise RuntimeError("No packet preambles found in the snapshot")

print("Preamble candidates found:", len(packet_candidates))
print("Packet BER threshold:", MAX_PACKET_BER)


def decode_packet_candidate(candidate):
    candidate_preamble_start = candidate['symbol_index']
    candidate_preamble_end = candidate_preamble_start + PREAMBLE_LENGTH
    preamble_rx = candidate['symbols'][
        candidate_preamble_start:candidate_preamble_end
    ]
    if len(preamble_rx) != PREAMBLE_LENGTH:
        return None

    phase_track = np.unwrap(np.angle(preamble_rx * preamble_ref))
    preamble_n = np.arange(PREAMBLE_LENGTH, dtype=np.float64)
    phase_slope, phase_at_start = np.polyfit(
        preamble_n, phase_track, 1
    )

    sample_indices = np.arange(len(rx_matched), dtype=np.float64)
    sample_phase_correction = phase_at_start + (
        phase_slope / RX_SYMBOL_SAMPLES
    ) * (sample_indices - candidate['start_sample'])
    corrected_samples = rx_matched * np.exp(-1j * sample_phase_correction)

    calibrated_start, calibrated_sps, calibrated_metric = (
        estimate_fractional_timing(
            corrected_samples,
            candidate['start_sample'],
            RX_SYMBOL_SAMPLES,
            preamble_ref
        )
    )

    calibrated_positions = calibrated_start + (
        np.arange(symbols_needed, dtype=np.float64) * calibrated_sps
    )
    if calibrated_positions[-1] >= len(corrected_samples) - 1:
        return None

    calibrated_symbols = np.asarray([
        interpolate_complex(corrected_samples, position)
        for position in calibrated_positions
    ], dtype=np.complex128)

    fixed_positions = candidate['start_sample'] + (
        np.arange(symbols_needed, dtype=np.float64) * RX_SYMBOL_SAMPLES
    )
    if fixed_positions[-1] >= len(corrected_samples) - 1:
        return None

    fixed_symbols = np.asarray([
        interpolate_complex(corrected_samples, position)
        for position in fixed_positions
    ], dtype=np.complex128)

    timing_result = gardner_timing_recovery(
        corrected_samples,
        calibrated_sps,
        calibrated_start
    )
    if len(timing_result['symbols']) < symbols_needed:
        return None
    gardner_symbols = timing_result['symbols'][:symbols_needed]

    symbol_paths = {
        'fixed': fixed_symbols,
        'calibrated': calibrated_symbols,
        'gardner': gardner_symbols
    }
    decoded_paths = {
        name: (np.real(values) < 0).astype(np.uint8)
        for name, values in symbol_paths.items()
    }
    path_errors = {
        name: int(np.count_nonzero(
            bits != decoded[PREAMBLE_LENGTH:symbols_needed]
        ))
        for name, decoded in decoded_paths.items()
    }

    selected_errors = path_errors[TIMING_MODE]
    residual_carrier_hz = (
        phase_slope * (RX_FS / RX_SYMBOL_SAMPLES) / (2 * np.pi)
    )

    return {
        'candidate': candidate,
        'corrected_samples': corrected_samples,
        'calibrated_start': calibrated_start,
        'calibrated_sps': calibrated_sps,
        'calibrated_metric': calibrated_metric,
        'phase_at_start': phase_at_start,
        'residual_carrier_hz': residual_carrier_hz,
        'timing_result': timing_result,
        'fixed_symbols': fixed_symbols,
        'calibrated_symbols': calibrated_symbols,
        'gardner_symbols': gardner_symbols,
        'decoded_paths': decoded_paths,
        'path_errors': path_errors,
        'ber': selected_errors / NUM_BITS
    }


best_packet = None
packets_tested = 0
ordered_candidates = sorted(
    packet_candidates,
    key=lambda item: item['confidence'],
    reverse=True
)

for attempt, candidate in enumerate(ordered_candidates, start=1):
    packet = decode_packet_candidate(candidate)
    if packet is None:
        print(f"Candidate {attempt}: incomplete packet window, skipped")
        continue
    packets_tested += 1

    print(
        f"Candidate {attempt}: sample={candidate['start_sample']}, "
        f"preamble confidence={candidate['confidence']:.4f}, "
        f"{TIMING_MODE} BER={packet['ber']:.6f}"
    )

    if best_packet is None or packet['ber'] < best_packet['ber']:
        best_packet = packet

    if packet['ber'] <= MAX_PACKET_BER:
        best_packet = packet
        break

    print("High BER; trying another packet in the snapshot")

if best_packet is None:
    raise RuntimeError("No complete packet candidate could be decoded")

result = best_packet['candidate']
rx_matched_corrected = best_packet['corrected_samples']
calibrated_start = best_packet['calibrated_start']
calibrated_sps = best_packet['calibrated_sps']
calibrated_metric = best_packet['calibrated_metric']
phase_at_start = best_packet['phase_at_start']
residual_carrier_hz = best_packet['residual_carrier_hz']
timing_result = best_packet['timing_result']
fixed_symbols = best_packet['fixed_symbols']
calibrated_symbols = best_packet['calibrated_symbols']
gardner_symbols = best_packet['gardner_symbols']
decoded_fixed = best_packet['decoded_paths']['fixed']
decoded_calibrated = best_packet['decoded_paths']['calibrated']
decoded_gardner = best_packet['decoded_paths']['gardner']
preamble_start = 0
preamble_end = PREAMBLE_LENGTH

print("\n========== SELECTED PACKET ==========")
print("Complete packets decoded:", packets_tested)
print("Selected sample:", result['start_sample'])
print("Preamble confidence:", round(result['confidence'], 4))
print("Estimated carrier phase:", np.degrees(phase_at_start), "degrees")
print("Estimated residual carrier offset:", residual_carrier_hz / 1e3, "kHz")
print("Preamble timing fit confidence:", calibrated_metric)
print("Nominal samples/symbol:", RX_SYMBOL_SAMPLES)
print("Preamble-fit samples/symbol:", calibrated_sps)
print("Timing loop start sample:", timing_result['positions'][0])
print("Timing loop initial samples/symbol:", timing_result['omega'][0])
print("Gardner TED mean error:", np.mean(timing_result['errors']))
print("Gardner TED RMS error:", np.sqrt(np.mean(timing_result['errors'] ** 2)))
print("Gardner samples/symbol, last:", timing_result['omega'][-1])
print(
    "Preamble-fit rate difference:",
    (calibrated_sps / RX_SYMBOL_SAMPLES - 1.0) * 1e6,
    "ppm"
)

print("\n========== TIMING CURSOR DIAGNOSTIC ==========")
for k in (0, 32, 64, 96, 127, 160, 256, 512, 1024):
    if k < len(timing_result['positions']):
        reference_position = calibrated_start + k * calibrated_sps
        cursor_drift = timing_result['positions'][k] - reference_position
        print(
            f"Recovered symbol {k}: cursor drift={cursor_drift:.4f} samples, "
            f"omega={timing_result['omega'][k]:.6f}, "
            f"TED error={timing_result['errors'][k]:+.6f}"
        )


def decode_and_report_timing_path(name, symbol_values):

    path_decoded = (np.real(symbol_values) < 0).astype(np.uint8)
    path_preamble_errors = np.count_nonzero(
        path_decoded[:PREAMBLE_LENGTH] != PREAMBLE
    )
    path_rx_bits = path_decoded[PREAMBLE_LENGTH:symbols_needed]
    path_errors = np.count_nonzero(bits != path_rx_bits)

    if path_errors:
        first_error = int(np.flatnonzero(bits != path_rx_bits)[0]) + 1
    else:
        first_error = None

    print(
        f"{name}: preamble errors={path_preamble_errors}, "
        f"payload errors={path_errors}/{NUM_BITS}, "
        f"BER={path_errors / NUM_BITS:.6f}, "
        f"first payload error={first_error}"
    )
    return path_decoded


print("\n========== TIMING PATH COMPARISON ==========")
decoded_fixed = decode_and_report_timing_path("Fixed-rate", fixed_symbols)
decoded_calibrated = decode_and_report_timing_path(
    "Preamble-calibrated", calibrated_symbols
)
decoded_gardner = decode_and_report_timing_path("Gardner TED", gardner_symbols)

# Select the configured timing path for the main BER report and plots.
if TIMING_MODE == "fixed":
    corrected_symbols = fixed_symbols
    decoded = decoded_fixed
elif TIMING_MODE == "calibrated":
    corrected_symbols = calibrated_symbols
    decoded = decoded_calibrated
elif TIMING_MODE == "gardner":
    corrected_symbols = gardner_symbols
    decoded = decoded_gardner
else:
    raise ValueError("TIMING_MODE must be 'fixed', 'calibrated', or 'gardner'")

# Verify detected preamble
detected_preamble = decoded[
    preamble_start:
    preamble_end
]

preamble_errors = np.count_nonzero(
    detected_preamble != PREAMBLE
)
preamble_errors_inv = np.count_nonzero(
    (1-detected_preamble) != PREAMBLE
)

print("\n========== SYNCHRONIZATION ==========")

print("Preamble errors:", preamble_errors)
print(f"Inverse preamble errors:: {preamble_errors_inv}\n\n")
print("Preamble BER:",
      preamble_errors / PREAMBLE_LENGTH)

if result['confidence'] < 0.50:
    raise RuntimeError(
        "Weak preamble correlation. "
    )


## EXTRACT DATA

payload_start = (
    preamble_start + PREAMBLE_LENGTH
)

payload_end = payload_start + NUM_BITS

if payload_end > len(decoded):
    raise RuntimeError(
        "Snapshot does not contain the complete payload"
    )

rx_bits = decoded[payload_start:payload_end]


## STATS

errors = np.count_nonzero(bits != rx_bits)

ber = errors / NUM_BITS

print("\n========== BER RESULTS ==========")

print("TX bits:", ''.join(map(str, bits)))
print("RX bits:", ''.join(map(str, rx_bits)))

print("Total bits:", NUM_BITS)
print("Bit errors:", errors)
print("BER:", ber)
print("Complete packets decoded:", packets_tested)


## PLOTS

plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
plt.plot(np.real(x[:1000]), label='I')
plt.plot(np.imag(x[:1000]), label='Q')
plt.title("Received IQ Snapshot")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(result['metric'])
plt.axvline(
    result['symbol_index'],
    color='red',
    linestyle='--'
)
plt.title("Normalized Soft Preamble Correlation")
plt.ylabel("Correlation")
plt.grid()

# Display the synchronized preamble neighborhood, not the first samples in
# the capture (which may be silence/noise before the detected frame).
view_start = max(0, preamble_start - 8)
view_end = min(len(corrected_symbols), preamble_end + 16)
view_symbols = corrected_symbols[view_start:view_end]
view_indices = np.arange(view_start, view_end) - preamble_start

plt.subplot(4, 1, 3)
plt.plot(view_indices, np.real(view_symbols), '.-', label='I')
plt.plot(view_indices, np.imag(view_symbols), '.-', label='Q')
plt.axvline(0, color='black', linestyle='--', alpha=0.6)
plt.axvline(PREAMBLE_LENGTH, color='red', linestyle='--', alpha=0.6)
plt.title("Matched-Filter Samples Around Detected Preamble")
plt.xlabel("Symbol index relative to preamble")
plt.legend()
plt.grid()

# plt.subplot(4, 1, 4)
# plt.plot(timing_result['errors'])
# plt.title("Gardner Timing Error")
# plt.xlabel("Recovered symbol index")
# plt.ylabel("Normalized TED error")
# plt.grid()

# plt.subplot(4, 1, 4)
#
# num_plot = min(100, NUM_BITS)
#
# plt.step(
#     np.arange(num_plot),
#     bits[:num_plot],
#     where='post',
#     label='TX'
# )
#
# plt.step(
#     np.arange(num_plot),
#     rx_bits[:num_plot] + 0.05,
#     where='post',
#     label='RX'
# )
#
# plt.title("TX vs RX")
# plt.xlabel("Bit")
# plt.legend()
# plt.grid()

plt.tight_layout()
# plt.show()


payload_symbols = corrected_symbols[payload_start:payload_end]

plt.figure(figsize=(7, 7))

plt.scatter(
    np.real(payload_symbols),
    np.imag(payload_symbols),
    s=35,
    alpha=0.7
)

plt.axhline(0, color='red', linestyle='--')
plt.axvline(0, color='red', linestyle='--')

plt.title("Decoded BPSK Payload Constellation")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")

plt.axis('equal')
plt.grid()
plt.tight_layout()
plt.show()

