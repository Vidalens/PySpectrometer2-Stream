```bash
v4l2-ctl --list-devices
```

```bash
v4l2-ctl --list-formats-ext --device /dev/video0
```

```bash
source ~/Documents/PySpectrometer2-Stream/pyspec/bin/activate
```

## Running the Spectrometer

**Server (with camera) - full spectrum:**
```bash
cd ~/Documents/PySpectrometer2-Stream/src
python PySpectrometer2-USB-v1.0.py --fullscreen --device 0 --fps 30
```

**Server with custom wavelength bounds (e.g., 400-700nm):**
```bash
cd ~/Documents/PySpectrometer2-Stream/src
python PySpectrometer2-USB-v1.0.py --fullscreen --wl-min 400 --wl-max 700
```

**Server with custom resolution and format:**
```bash
# Use YUYV format at 1280x720
python PySpectrometer2-USB-v1.0.py --fullscreen --format YUYV --width 1280 --height 720 --fps 9

# Use MJPG format at 1920x1080
python PySpectrometer2-USB-v1.0.py --fullscreen --format MJPG --width 1920 --height 1080 --fps 30
```

**Server options:**
- `--width N`: Camera resolution width (default: 800)
- `--height N`: Camera resolution height (default: 600)
- `--format`: Video format - MJPG or YUYV (default: MJPG)
- `--wl-min N`: Minimum wavelength in nm (e.g., 400)
- `--wl-max N`: Maximum wavelength in nm (e.g., 800)
- `--device N`: Video device number (default: 0)
- `--fps N`: Frame rate (default: 30)
- `--compress`: Enable LZ4 compression for streaming

**Client (viewing remotely):**
```bash
cd ~/Documents/PySpectrometer2-Stream/src
python spectrometer_client.py --host localhost --port 5555 --avg-frames 5
```

**Client with custom wavelength display bounds:**
```bash
cd ~/Documents/PySpectrometer2-Stream/src
python spectrometer_client.py --wl-min 400 --wl-max 700 --avg-frames 5
```

**Client options:**
- `--avg-frames N`: Number of frames to average (default: 5, higher = smoother but slower response)
- `--wl-min N`: Minimum wavelength to display (nm)
- `--wl-max N`: Maximum wavelength to display (nm)
- `--host`: Server hostname/IP (default: localhost)
- `--port`: ZeroMQ port (default: 5555)

## Examples

**Focus on visible spectrum only (400-700nm):**
```bash
# Server
python PySpectrometer2-USB-v1.0.py --fullscreen --wl-min 400 --wl-max 700

# Client
python spectrometer_client.py --wl-min 400 --wl-max 700
```

**View near-infrared (700-900nm):**
```bash
# Server
python PySpectrometer2-USB-v1.0.py --fullscreen --wl-min 700 --wl-max 900
```