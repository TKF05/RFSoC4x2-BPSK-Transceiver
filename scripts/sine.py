import numpy as np
import sys 

fs = 1966.08e6
# f = 30.72e6
f = 131.027e6
# f = 5.25e9
# f = 10e6

if len(sys.argv) < 2: f = f
else: f = float(sys.argv[1])

BRAM_DEPTH = 8192
SAMPLES_PER_WORD = 8
TOTAL_SAMPLES = BRAM_DEPTH * SAMPLES_PER_WORD

n = np.arange(TOTAL_SAMPLES)

y = np.cos(2 * np.pi * f / fs * n)

# Signed 16-bit samples
y_scaled = np.round(64 * y).astype(np.int16)

# Pack 8 samples per 128-bit word
buf = y_scaled.astype('<i2').tobytes()
# buf = b''.join(
#     struct.pack('<8h', *tx[i:i+8])
#     for i in range(0, TOTAL_SAMPLES, 8)
# )

print("Samples:", len(y_scaled))
print("Buffer size:", len(buf))

assert len(buf) == 131072

r.write('tx_bram', buf)
