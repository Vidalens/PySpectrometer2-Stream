#!/usr/bin/env python3

'''
PySpectrometer2 Streaming Test/Demo
Simulates a spectrometer stream for testing the client without hardware
'''

import zmq
import struct
import zlib
import numpy as np
import time
import argparse

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False
    import hashlib

# Protocol constants
MAGIC = b"HSPC"
VERSION = 1
HEADER_SIZE = 64

def compute_wavelength_id(wavelengths_f32):
    """Compute unique ID for wavelength calibration"""
    raw = np.asarray(wavelengths_f32, dtype=np.float32).tobytes()
    if HAS_BLAKE3:
        wl_id = blake3.blake3(raw).digest(length=16)
    else:
        wl_id = hashlib.md5(raw).digest()
    return wl_id, raw

def build_header(flags, stream_id, frame_idx, t_mono_ns, t_utc_ns, wl_id, n_pixels, sample_bits, payload_len, payload_bytes):
    """Build 64-byte binary frame header with CRC32"""
    header_wo_crc = struct.pack(
        "<4sB B H I Q Q Q 16s I B 3s I",
        MAGIC, VERSION, flags, HEADER_SIZE,
        stream_id, frame_idx, t_mono_ns, t_utc_ns,
        wl_id, n_pixels, sample_bits, b"\x00\x00\x00", payload_len
    )
    crc = zlib.crc32(header_wo_crc + payload_bytes) & 0xFFFFFFFF
    return header_wo_crc + struct.pack("<I", crc)

def main():
    parser = argparse.ArgumentParser(description='PySpectrometer2 Stream Simulator')
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ port (default: 5555)")
    parser.add_argument("--stream-id", type=int, default=1, help="Stream ID (default: 1)")
    parser.add_argument("--fps", type=int, default=21, help="Simulated frame rate (default: 21)")
    parser.add_argument("--n-pixels", type=int, default=800, help="Number of pixels (default: 800)")
    parser.add_argument("--compress", action="store_true", help="Enable LZ4 compression")
    args = parser.parse_args()
    
    # Initialize ZeroMQ
    print(f"[info] Starting simulated spectrometer on port {args.port}")
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    
    topic = f"hspc.stream.{args.stream_id}".encode()
    
    # Generate wavelength calibration (simulate 380-780nm range)
    wavelengths = np.linspace(380.0, 780.0, args.n_pixels, dtype=np.float32)
    wl_id, wl_raw = compute_wavelength_id(wavelengths)
    
    # Send calibration block
    print(f"[info] Sending calibration: {args.n_pixels} pixels, {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm")
    print(f"[info] Wavelength ID: {wl_id.hex()}")
    meta = b'{"model":"Simulator","version":"1.0","mode":"demo"}'
    calib_payload = wl_raw + meta
    calib_hdr = build_header(
        flags=0, stream_id=args.stream_id, frame_idx=0,
        t_mono_ns=time.monotonic_ns(), t_utc_ns=time.time_ns(),
        wl_id=wl_id, n_pixels=args.n_pixels, sample_bits=16,
        payload_len=len(calib_payload), payload_bytes=calib_payload
    )
    sock.send_multipart([topic, calib_hdr + calib_payload], copy=False)
    print(f"[info] Calibration sent")
    
    # Small delay for subscribers to connect
    time.sleep(0.5)
    
    # Generate and stream frames
    frame_idx = 1
    frame_interval = 1.0 / args.fps
    print(f"[info] Streaming at {args.fps} fps (Ctrl+C to stop)")
    print(f"[info] Compression: {'enabled' if args.compress and HAS_LZ4 else 'disabled'}")
    
    try:
        while True:
            start_time = time.time()
            
            # Generate simulated spectrum (moving Gaussian peaks)
            x = np.arange(args.n_pixels, dtype=np.float32)
            
            # Create multiple moving peaks
            spectrum = np.zeros(args.n_pixels, dtype=np.float32)
            
            # Peak 1: Moving across spectrum
            peak1_pos = (args.n_pixels // 2) + (args.n_pixels // 4) * np.sin(frame_idx * 0.02)
            spectrum += 15000 * np.exp(-((x - peak1_pos) ** 2) / 2000)
            
            # Peak 2: Pulsing intensity
            peak2_pos = args.n_pixels // 3
            peak2_amp = 10000 + 5000 * np.sin(frame_idx * 0.05)
            spectrum += peak2_amp * np.exp(-((x - peak2_pos) ** 2) / 1500)
            
            # Peak 3: Fixed reference
            peak3_pos = 2 * args.n_pixels // 3
            spectrum += 8000 * np.exp(-((x - peak3_pos) ** 2) / 1200)
            
            # Add baseline and noise
            spectrum += 2000 + np.random.normal(0, 200, args.n_pixels)
            
            # Clip to uint16 range
            spectrum = np.clip(spectrum, 0, 65535).astype(np.uint16)
            
            # Build frame
            payload = spectrum.tobytes()
            flags = 0
            
            if args.compress and HAS_LZ4:
                payload = lz4.frame.compress(payload, block_linked=False, store_size=False)
                flags |= 0x01
            
            hdr = build_header(
                flags=flags, stream_id=args.stream_id, frame_idx=frame_idx,
                t_mono_ns=time.monotonic_ns(), t_utc_ns=time.time_ns(),
                wl_id=wl_id, n_pixels=args.n_pixels, sample_bits=16,
                payload_len=len(payload), payload_bytes=payload
            )
            
            sock.send_multipart([topic, hdr + payload], copy=False)
            
            if frame_idx % 100 == 0:
                print(f"[info] Frame {frame_idx} sent")
            
            frame_idx += 1
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n[info] Stopping simulator")
    
    # Cleanup
    sock.close()
    ctx.term()
    print("[info] Simulator stopped")

if __name__ == "__main__":
    main()


