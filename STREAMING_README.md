# PySpectrometer2 - ZeroMQ Streaming

This enhanced version of PySpectrometer2 includes real-time spectral data streaming over ZeroMQ using a high-performance binary protocol designed for lossless, low-latency telemetry.

## Features

- **Lossless streaming**: 16-bit intensity data with no interpolation
- **Low latency**: Optimized binary framing with zero-copy operations
- **Optional compression**: LZ4 frame compression for reduced bandwidth
- **Calibration management**: Wavelength calibration sent once, referenced by hash
- **Multi-client support**: ZeroMQ PUB/SUB allows multiple simultaneous viewers
- **Timestamping**: Monotonic and UTC timestamps for each frame
- **CRC32 validation**: Data integrity checking on every frame

## Protocol Specification

### Binary Frame Format

All frames use a 64-byte header followed by variable-length payload:

```
struct FrameHeader {
    char     magic[4];        // "HSPC"
    uint8_t  version;         // 1
    uint8_t  flags;           // bit0=LZ4, bit1=ZSTD, bit2=HAS_DARK, etc.
    uint16_t header_bytes;    // 64
    uint32_t stream_id;       // device stream identifier
    uint64_t frame_idx;       // incrementing frame counter (0=calibration)
    uint64_t t_monotonic_ns;  // CLOCK_MONOTONIC capture time
    uint64_t t_utc_ns;        // CLOCK_REALTIME in nanoseconds
    uint8_t  wavelength_id[16]; // blake3-128 hash of wavelength table
    uint32_t n_pixels;        // number of pixels/wavelengths
    uint8_t  sample_bits;     // 16 or 32
    uint8_t  reserved8[3];    // padding
    uint32_t payload_len;     // bytes after header
    uint32_t crc32;           // CRC32 over header+payload
}
```

### Frame Types

1. **Calibration Block** (frame_idx=0): Sent once at startup
   - Contains wavelength array (float32) + JSON metadata
   - Establishes wavelength_id for all subsequent frames

2. **Intensity Frame** (frame_idx>0): Sent for each captured spectrum
   - Contains intensity array (uint16 little-endian)
   - References wavelength_id from calibration

## Installation

### Dependencies

Install required packages:

```bash
pip install -r requirements-streaming.txt
```

Or install individually:

```bash
pip install pyzmq lz4 blake3 matplotlib numpy opencv-python
```

## Usage

### Running the Spectrometer (Server)

Start the spectrometer with streaming enabled:

```bash
# Default settings (port 5555, stream ID 1)
./PySpectrometer2-USB-v1.0.py

# Custom port
./PySpectrometer2-USB-v1.0.py --port 5556

# Multiple devices on same network (use different stream IDs)
./PySpectrometer2-USB-v1.0.py --stream-id 1 --device 0
./PySpectrometer2-USB-v1.0.py --stream-id 2 --device 1 --port 5556

# Enable LZ4 compression (recommended for network streaming)
./PySpectrometer2-USB-v1.0.py --compress
```

All original PySpectrometer2 options remain available:

```bash
# Fullscreen mode with streaming
./PySpectrometer2-USB-v1.0.py --fullscreen --port 5555

# Waterfall display with streaming
./PySpectrometer2-USB-v1.0.py --waterfall --compress
```

### Running the Client Viewer

View the live stream on the same machine or over the network:

```bash
# Local viewing
./spectrometer_client.py

# Remote viewing
./spectrometer_client.py --host 192.168.1.100 --port 5555

# View specific stream when multiple devices are running
./spectrometer_client.py --stream-id 2
```

The client will:
1. Connect to the ZeroMQ publisher
2. Wait for calibration block
3. Display live spectral data with matplotlib
4. Show real-time FPS in the title

### Multiple Viewers

You can run multiple clients simultaneously to view the same stream:

```bash
# Terminal 1 - Server
./PySpectrometer2-USB-v1.0.py --port 5555

# Terminal 2 - Viewer 1
./spectrometer_client.py

# Terminal 3 - Viewer 2
./spectrometer_client.py

# Terminal 4 - Remote viewer
ssh user@remote-host
./spectrometer_client.py --host 192.168.1.100
```

## Command-Line Options

### Server (PySpectrometer2-USB-v1.0.py)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--device` | int | 0 | Video device number |
| `--fps` | int | 30 | Frame rate |
| `--port` | int | 5555 | ZeroMQ streaming port |
| `--stream-id` | int | 1 | Stream identifier |
| `--compress` | flag | False | Enable LZ4 compression |
| `--fullscreen` | flag | False | Fullscreen mode (800×480) |
| `--waterfall` | flag | False | Enable waterfall display |

### Client (spectrometer_client.py)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host` | str | localhost | Spectrometer host address |
| `--port` | int | 5555 | ZeroMQ port |
| `--stream-id` | int | 1 | Stream ID to subscribe to |

## Performance

### Bandwidth Estimates

For 800 pixels (PySpectrometer2 default):

| Frame Rate | Uncompressed | With LZ4 (est.) |
|------------|--------------|-----------------|
| 21 fps | ~34 KB/s | ~17-25 KB/s |
| 30 fps | ~48 KB/s | ~24-36 KB/s |
| 60 fps | ~96 KB/s | ~48-72 KB/s |

These rates are trivial for any modern network (1 Gbps = ~125 MB/s).

### Latency

- Typical end-to-end latency: 5-15ms on LAN
- ZeroMQ adds <1ms overhead
- LZ4 compression adds <1ms
- Matplotlib rendering is the main latency source (~10-20ms)

## Network Configuration

### Firewall

If viewing from remote machines, ensure the port is accessible:

```bash
# Allow incoming TCP on default port
sudo ufw allow 5555/tcp

# Or for a specific range
sudo ufw allow 5555:5560/tcp
```

### Finding the Server IP

On the server machine:

```bash
# Linux
ip addr show

# Show just the primary interface
hostname -I
```

## Advanced Usage

### Custom Integration

The streaming protocol can be integrated into custom applications:

```python
import zmq
import struct
import numpy as np

# Subscribe to stream
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:5555")
sub.subscribe(b"hspc.stream.1")

while True:
    topic, frame = sub.recv_multipart()
    # Parse header (see parse_header in spectrometer_client.py)
    # Process intensity data
    # Your custom processing here
```

### Data Logging

The binary frames can be saved directly to disk:

```python
# In your subscriber
with open("spectral_log.hspc", "ab") as f:
    topic, frame = sub.recv_multipart()
    f.write(frame)  # Concatenate frames directly
```

Replay later by reading frames sequentially.

### Multi-Device Synchronization

Run multiple spectrometers with different stream IDs:

```bash
# Device 1
./PySpectrometer2-USB-v1.0.py --device 0 --stream-id 1 --port 5555

# Device 2 (different port or same port with different stream-id)
./PySpectrometer2-USB-v1.0.py --device 1 --stream-id 2 --port 5555
```

Clients can subscribe to specific streams or all streams:

```python
# Subscribe to all streams
sub.subscribe(b"hspc.stream.")  # Wildcard prefix

# Or specific stream
sub.subscribe(b"hspc.stream.1")
```

## Troubleshooting

### No calibration received

- Ensure server started successfully
- Check firewall settings
- Verify correct port and host
- Server sends calibration at startup; restart if needed

### Frame drops

- Enable compression with `--compress`
- Check network bandwidth
- Reduce frame rate with `--fps`
- Verify client can keep up with processing

### CRC errors

- Check for network issues
- Verify same protocol version on client/server
- Check for data corruption

### Import errors

```bash
# If blake3 not available
pip install blake3

# If lz4 not available (compression won't work)
pip install lz4

# If matplotlib not available (client won't work)
pip install matplotlib
```

## Protocol Details

### Wavelength ID

The wavelength_id is a blake3 hash (or md5 fallback) of the wavelength calibration array. This allows:
- Verification that client/server wavelength tables match
- Efficient calibration updates (only send if hash changes)
- Detection of recalibration events

### Compression

LZ4 compression typically provides:
- 1.3-2x compression on smooth spectra
- <1ms compression/decompression latency
- Minimal CPU overhead

Enable with `--compress` flag. Client automatically detects and decompresses.

### Timestamps

Each frame includes two timestamps:
- `t_monotonic_ns`: Monotonic clock, never goes backwards, for frame ordering
- `t_utc_ns`: Real-time clock in nanoseconds since Unix epoch, for absolute time

Use monotonic time for timing analysis, UTC for logging.

## Future Enhancements

Possible additions:
- QUIC transport for WAN/multi-tenant networks
- Dark frame and reference subtraction streaming
- Saturation mask bitfield
- Integration time metadata in frames
- PTP time synchronization for multi-device setups
- Circular buffer recording mode
- Automated discovery via UDP beacon

## License

Same as PySpectrometer2 - see main LICENSE file.

## Author

Streaming implementation based on specifications for high-performance spectral telemetry.
Original PySpectrometer2 by Les Wright.


