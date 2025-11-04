#!/usr/bin/env python3

'''
PySpectrometer2 ZeroMQ Client
Receives and displays spectral data streamed from PySpectrometer2-USB
'''

import zmq
import struct
import zlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys
import time

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("[warning] lz4 not available, compressed streams will fail")

# Protocol constants
MAGIC = b"HSPC"
VERSION = 1
HEADER_SIZE = 64  # Header without CRC
FULL_HEADER_SIZE = 68  # Header with CRC (64 + 4)

# Global state
wavelengths = None
wavelength_id = None
frame_count = 0
last_frame_time = time.time()
fps_estimate = 0.0

def parse_header(buf):
    """Parse 68-byte binary frame header (64 bytes + 4 byte CRC)"""
    if len(buf) < FULL_HEADER_SIZE:
        raise ValueError("Buffer too small for header")
    
    fields = struct.unpack("<4sB B H I Q Q Q 16s I B 3s I I", buf[:FULL_HEADER_SIZE])
    
    magic = fields[0]
    version = fields[1]
    flags = fields[2]
    header_bytes = fields[3]
    stream_id = fields[4]
    frame_idx = fields[5]
    t_monotonic_ns = fields[6]
    t_utc_ns = fields[7]
    wl_id = fields[8]
    n_pixels = fields[9]
    sample_bits = fields[10]
    payload_len = fields[12]
    crc = fields[13]
    
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic}")
    
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")
    
    # Verify CRC
    if len(buf) < FULL_HEADER_SIZE + payload_len:
        raise ValueError("Buffer too small for payload")
    
    # CRC is calculated over header (64 bytes) + payload
    calc_crc = zlib.crc32(buf[:HEADER_SIZE] + buf[FULL_HEADER_SIZE:FULL_HEADER_SIZE+payload_len]) & 0xFFFFFFFF
    if calc_crc != crc:
        raise ValueError(f"CRC mismatch: expected {crc:08x}, got {calc_crc:08x}")
    
    return {
        'flags': flags,
        'stream_id': stream_id,
        'frame_idx': frame_idx,
        't_monotonic_ns': t_monotonic_ns,
        't_utc_ns': t_utc_ns,
        'wavelength_id': wl_id,
        'n_pixels': n_pixels,
        'sample_bits': sample_bits,
        'payload_len': payload_len
    }

def process_calibration(payload, hdr):
    """Process calibration block to extract wavelengths"""
    global wavelengths, wavelength_id
    
    n_pixels = hdr['n_pixels']
    wl_bytes = n_pixels * 4  # float32
    
    if len(payload) < wl_bytes:
        raise ValueError("Calibration payload too small")
    
    wavelengths = np.frombuffer(payload[:wl_bytes], dtype=np.float32)
    wavelength_id = hdr['wavelength_id']
    
    # Extract metadata if present
    if len(payload) > wl_bytes:
        meta_bytes = payload[wl_bytes:]
        try:
            meta_str = meta_bytes.decode('utf-8')
            print(f"[info] Metadata: {meta_str}")
        except:
            pass
    
    print(f"[info] Received calibration: {n_pixels} pixels, {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm")
    print(f"[info] Wavelength ID: {wavelength_id.hex()}")

def process_intensity_frame(payload, hdr):
    """Process intensity frame and return intensity array"""
    global frame_count, last_frame_time, fps_estimate
    
    # Decompress if needed
    if hdr['flags'] & 0x01:  # LZ4 flag
        if not HAS_LZ4:
            raise ValueError("Frame is LZ4 compressed but lz4 not available")
        payload = lz4.frame.decompress(payload)
    
    # Parse intensity data
    dtype = np.uint16 if hdr['sample_bits'] == 16 else np.uint32
    intensity = np.frombuffer(payload, dtype=dtype, count=hdr['n_pixels'])
    
    # Update FPS estimate
    current_time = time.time()
    if frame_count == 0:
        # Initialize timing on first frame
        last_frame_time = current_time
    elif frame_count % 10 == 0:
        elapsed = current_time - last_frame_time
        fps_estimate = 10.0 / elapsed if elapsed > 0 else 0
        last_frame_time = current_time
    
    frame_count += 1
    
    return intensity

def init_plot():
    """Initialize matplotlib plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], 'b-', linewidth=0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('PySpectrometer2 - Live Stream')
    ax.grid(True, alpha=0.3)
    # Set Y-axis for 8-bit intensity data (0-255 range)
    ax.set_ylim(0, 255)
    
    return fig, ax, line

def animate(frame, sub, ax, line):
    """Animation update function"""
    global wavelengths, wavelength_id, frame_count, fps_estimate
    
    try:
        # Non-blocking receive
        if sub.poll(timeout=100):
            topic, data = sub.recv_multipart()
            
            hdr = parse_header(data)
            payload = data[FULL_HEADER_SIZE:FULL_HEADER_SIZE + hdr['payload_len']]
            
            # Check if it's a calibration block (frame_idx == 0)
            if hdr['frame_idx'] == 0:
                process_calibration(payload, hdr)
                # Reinitialize plot with wavelength range
                if wavelengths is not None:
                    ax.set_xlim(wavelengths[0], wavelengths[-1])
                    ax.set_ylim(0, 255)
            else:
                # Regular intensity frame
                if wavelengths is None:
                    print("[warning] Received frame before calibration, skipping")
                    return line,
                
                # Check wavelength ID matches
                if hdr['wavelength_id'] != wavelength_id:
                    print("[warning] Wavelength ID mismatch, requesting recalibration")
                    return line,
                
                intensity = process_intensity_frame(payload, hdr)
                
                # Update plot
                line.set_data(wavelengths, intensity)
                ax.set_title(f"PySpectrometer2 - Live Stream (Frame {hdr['frame_idx']}, {fps_estimate:.1f} fps)")
                
    except Exception as e:
        print(f"[error] {e}")
    
    return line,

def main():
    parser = argparse.ArgumentParser(description='PySpectrometer2 ZeroMQ Client')
    parser.add_argument("--host", type=str, default="localhost", help="Spectrometer host (default: localhost)")
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ port (default: 5555)")
    parser.add_argument("--stream-id", type=int, default=1, help="Stream ID to subscribe to (default: 1)")
    parser.add_argument("--save-interval", type=int, default=0, help="Save frames every N seconds (0=disabled)")
    args = parser.parse_args()
    
    # Initialize ZeroMQ subscriber
    print(f"[info] Connecting to {args.host}:{args.port}")
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.port}")
    
    topic = f"hspc.stream.{args.stream_id}".encode()
    sub.setsockopt(zmq.SUBSCRIBE, topic)
    print(f"[info] Subscribed to topic: {topic.decode()}")
    
    # Wait for initial calibration
    print("[info] Waiting for calibration block...")
    while wavelengths is None:
        if sub.poll(timeout=1000):
            topic_recv, data = sub.recv_multipart()
            try:
                hdr = parse_header(data)
                payload = data[FULL_HEADER_SIZE:FULL_HEADER_SIZE + hdr['payload_len']]
                if hdr['frame_idx'] == 0:
                    process_calibration(payload, hdr)
                    break
            except Exception as e:
                print(f"[error] {e}")
        else:
            print("[info] Still waiting...")
    
    # Initialize plot
    fig, ax, line = init_plot()
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_ylim(0, 255)
    
    # Start animation
    ani = FuncAnimation(fig, animate, fargs=(sub, ax, line), 
                       interval=20, blit=True, cache_frame_data=False)
    
    print("[info] Starting live display (close window to exit)")
    plt.show()
    
    # Cleanup
    sub.close()
    ctx.term()
    print("[info] Client stopped")

if __name__ == "__main__":
    main()


