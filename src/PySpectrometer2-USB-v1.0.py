#!/usr/bin/env python3

'''
PySpectrometer2 Les Wright 2022
https://www.youtube.com/leslaboratory
https://github.com/leswright1977

This project is a follow on from: https://github.com/leswright1977/PySpectrometer 

This is a more advanced, but more flexible version of the original program. Tk Has been dropped as the GUI to allow fullscreen mode on Raspberry Pi systems and the iterface is designed to fit 800*480 screens, which seem to be a common resolutin for RPi LCD's, paving the way for the creation of a stand alone benchtop instrument.

Whats new:
Higher resolution (800px wide graph)
3 row pixel averaging of sensor data
Fullscreen option for the Spectrometer graph
3rd order polymonial fit of calibration data for accurate measurement.
Improved graph labelling
Labelled measurement cursors
Optional waterfall display for recording spectra changes over time.
Key Bindings for all operations

All old features have been kept, including peak hold, peak detect, Savitsky Golay filter, and the ability to save graphs as png and data as CSV.

For instructions please consult the readme!

'''


import cv2
import time
import numpy as np
from specFunctions import wavelength_to_rgb,savitzky_golay,peakIndexes,readcal,writecal,background,generateGraticule
import base64
import argparse
import zmq
import struct
import zlib
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("[warning] lz4 not available, compression disabled")
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False
    print("[warning] blake3 not available, using hashlib")
    import hashlib

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
parser.add_argument("--fps", type=int, default=30, help="Frame Rate e.g. 30")
parser.add_argument("--width", type=int, default=800, help="Camera resolution width (default: 800)")
parser.add_argument("--height", type=int, default=600, help="Camera resolution height (default: 600)")
parser.add_argument("--format", type=str, default="MJPG", choices=["MJPG", "YUYV"], help="Video format (default: MJPG)")
parser.add_argument("--port", type=int, default=5555, help="ZeroMQ streaming port (default: 5555)")
parser.add_argument("--stream-id", type=int, default=1, help="Stream ID for this spectrometer (default: 1)")
parser.add_argument("--compress", help="Enable LZ4 compression for streaming",action="store_true")
parser.add_argument("--wl-min", type=float, default=None, help="Minimum wavelength to display (nm)")
parser.add_argument("--wl-max", type=float, default=None, help="Maximum wavelength to display (nm)")
group = parser.add_mutually_exclusive_group()
group.add_argument("--fullscreen", help="Fullscreen (Native 800*480)",action="store_true")
group.add_argument("--waterfall", help="Enable Waterfall (Windowed only)",action="store_true")
args = parser.parse_args()
dispFullscreen = False
dispWaterfall = False
if args.fullscreen:
	print("Fullscreen Spectrometer enabled")
	dispFullscreen = True
if args.waterfall:
	print("Waterfall display enabled")
	dispWaterfall = True
	
if args.device:
	dev = args.device
else:
	dev = 0
	
if args.fps:
	fps = args.fps
else:
	fps = 30

# ZeroMQ Streaming Protocol Constants
MAGIC = b"HSPC"
VERSION = 1
HEADER_SIZE = 64

# ZeroMQ streaming functions
def compute_wavelength_id(wavelengths_f32):
	"""Compute unique ID for wavelength calibration"""
	raw = np.asarray(wavelengths_f32, dtype=np.float32).tobytes()
	if HAS_BLAKE3:
		wl_id = blake3.blake3(raw).digest(length=16)
	else:
		wl_id = hashlib.md5(raw).digest()  # fallback to md5
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

def send_calibration_block(sock, wavelengths, stream_id, topic):
	"""Send wavelength calibration block once at startup"""
	wl_id, wl_raw = compute_wavelength_id(wavelengths)
	meta = b'{"model":"PySpectrometer2","device":"USB","version":"1.0"}'
	calib_payload = wl_raw + meta
	calib_hdr = build_header(
		flags=0, stream_id=stream_id, frame_idx=0,
		t_mono_ns=time.monotonic_ns(), t_utc_ns=time.time_ns(),
		wl_id=wl_id, n_pixels=len(wavelengths), sample_bits=16,
		payload_len=len(calib_payload), payload_bytes=calib_payload
	)
	sock.send_multipart([topic, calib_hdr + calib_payload], copy=False)
	return wl_id

def send_intensity_frame(sock, intensity, wl_id, stream_id, frame_idx, topic, use_compression=False):
	"""Send intensity frame over ZeroMQ"""
	# Convert intensity to uint16
	intens_u16 = np.asarray(intensity, dtype=np.uint16)
	payload = intens_u16.tobytes()
	flags = 0
	
	if use_compression and HAS_LZ4:
		payload = lz4.frame.compress(payload, block_linked=False, store_size=False)
		flags |= 0x01  # LZ4 flag
	
	hdr = build_header(
		flags=flags, stream_id=stream_id, frame_idx=frame_idx,
		t_mono_ns=time.monotonic_ns(), t_utc_ns=time.time_ns(),
		wl_id=wl_id, n_pixels=len(intens_u16), sample_bits=16,
		payload_len=len(payload), payload_bytes=payload
	)
	
	sock.send_multipart([topic, hdr + payload], copy=False)

frameWidth = args.width
frameHeight = args.height

#init video
cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)
#cap = cv2.VideoCapture(0)

# Set video format
if args.format == "MJPG":
	fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
	cap.set(cv2.CAP_PROP_FOURCC, fourcc)
	print(f"[info] Requesting MJPG format")
elif args.format == "YUYV":
	fourcc = cv2.VideoWriter_fourcc('Y','U','Y','V')
	cap.set(cv2.CAP_PROP_FOURCC, fourcc)
	print(f"[info] Requesting YUYV format")

print(f"[info] Requesting resolution: {frameWidth}x{frameHeight} @ {fps} fps")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(cv2.CAP_PROP_FPS, fps)

# Check what we actually got
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])

print(f"\n{'='*60}")
print(f"[info] Camera initialized:")
print(f"[info]   Format: {fourcc_str}")
print(f"[info]   Resolution: {actual_width}x{actual_height}")
print(f"[info]   Frame rate: {actual_fps:.1f} fps")

# Warn if we didn't get what was requested
if actual_width != frameWidth or actual_height != frameHeight:
	print(f"[warning] Requested {frameWidth}x{frameHeight} but got {actual_width}x{actual_height}")
	print(f"[warning] Using actual camera resolution")
	frameWidth = actual_width
	frameHeight = actual_height

if abs(actual_fps - fps) > 1:
	print(f"[warning] Requested {fps} fps but got {actual_fps:.1f} fps")

print(f"{'='*60}\n")

cfps = actual_fps

# Store original camera width for later use
cameraWidth = frameWidth


title1 = 'PySpectrometer 2 - Spectrograph'
title2 = 'PySpectrometer 2 - Waterfall'
stackHeight = 320+80+80 #height of the displayed CV window (graph+preview+messages)

if dispWaterfall == True:
	#watefall first so spectrum is on top
	cv2.namedWindow(title2,cv2.WINDOW_GUI_NORMAL)
	cv2.resizeWindow(title2,cameraWidth,stackHeight)
	cv2.moveWindow(title2,200,200);

if dispFullscreen == True:
	cv2.namedWindow(title1,cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty(title1,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
else:
	cv2.namedWindow(title1,cv2.WINDOW_GUI_NORMAL)
	cv2.resizeWindow(title1,cameraWidth,stackHeight)
	cv2.moveWindow(title1,0,0);

#settings for peak detect
savpoly = 7 #savgol filter polynomial max val 15
mindist = 50 #minumum distance between peaks max val 100
thresh = 20 #Threshold max val 100

calibrate = False

clickArray = [] 
cursorX = 0
cursorY = 0
def handle_mouse(event,x,y,flags,param):
	global clickArray
	global cursorX
	global cursorY
	mouseYOffset = 160
	if event == cv2.EVENT_MOUSEMOVE:
		cursorX = x
		cursorY = y	
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX = x
		mouseY = y-mouseYOffset
		clickArray.append([mouseX,mouseY])
#listen for click on plot window
cv2.setMouseCallback(title1,handle_mouse)


font=cv2.FONT_HERSHEY_SIMPLEX

# intensity array will be initialized after calibration is loaded and bounds are applied

holdpeaks = False #are we holding peaks?
measure = False #are we measuring?
recPixels = False #are we measuring pixels and recording clicks?


#messages
msg1 = ""
saveMsg = "No data saved"

#Go grab the computed calibration data
caldata = readcal(cameraWidth, frameHeight)
wavelengthData_full = caldata[0]
calmsg1 = caldata[1]
calmsg2 = caldata[2]
calmsg3 = caldata[3]

# Print full calibration range
full_wl_min = min(wavelengthData_full)
full_wl_max = max(wavelengthData_full)
print(f"\n{'='*60}")
print(f"[info] Full calibration range: {full_wl_min:.1f} - {full_wl_max:.1f} nm")
print(f"[info] Full pixel range: 0 - {len(wavelengthData_full)-1} ({len(wavelengthData_full)} pixels)")

# Apply wavelength bounds if specified, with clamping to available range
wl_min_requested = args.wl_min
wl_max_requested = args.wl_max

# Automatically crop negative wavelengths (non-physical)
if full_wl_min < 0:
	print(f"[info] Calibration includes non-physical negative wavelengths ({full_wl_min:.1f} nm)")
	print(f"[info] Automatically cropping to start at 0 nm")
	auto_crop_min = 0.0
else:
	auto_crop_min = full_wl_min

# Use user-specified bounds or auto-cropped minimum
wl_min_bound = args.wl_min if args.wl_min is not None else auto_crop_min
wl_max_bound = args.wl_max if args.wl_max is not None else full_wl_max

# Clamp to available range and warn if out of bounds
if wl_min_bound < full_wl_min:
	print(f"[warning] Requested min wavelength {wl_min_bound:.1f} nm is below available range")
	print(f"[warning] Clamping to minimum available: {full_wl_min:.1f} nm")
	wl_min_bound = full_wl_min

if wl_max_bound > full_wl_max:
	print(f"[warning] Requested max wavelength {wl_max_bound:.1f} nm is above available range")
	print(f"[warning] Clamping to maximum available: {full_wl_max:.1f} nm")
	wl_max_bound = full_wl_max

# Find pixel indices that fall within the wavelength bounds
valid_indices = [i for i, wl in enumerate(wavelengthData_full) if wl_min_bound <= wl <= wl_max_bound]

if len(valid_indices) == 0:
	print(f"[error] No wavelengths found in range {wl_min_bound:.1f}-{wl_max_bound:.1f} nm")
	print(f"[error] Available range: {full_wl_min:.1f}-{full_wl_max:.1f} nm")
	exit(1)

# Crop wavelength data to the specified range
wavelengthData = [wavelengthData_full[i] for i in valid_indices]
pixel_start = valid_indices[0]
pixel_end = valid_indices[-1]
displayWidth = len(wavelengthData)

if wl_min_requested is not None or wl_max_requested is not None:
	print(f"[info] *** WAVELENGTH BOUNDS APPLIED ***")
print(f"[info] Active display range: {wavelengthData[0]:.1f} - {wavelengthData[-1]:.1f} nm")
print(f"[info] Active pixel range: {pixel_start} - {pixel_end} ({displayWidth} pixels)")
print(f"{'='*60}\n")

# Update intensity array size to match display width
intensity = [0] * displayWidth

#blank image for Waterfall (use displayWidth for graph data)
waterfall = np.zeros([320,displayWidth,3],dtype=np.uint8)
waterfall.fill(0) #fill black

# Initialize ZeroMQ streaming
print(f"[info] Initializing ZeroMQ streaming on port {args.port}")
zmq_context = zmq.Context.instance()
zmq_sock = zmq_context.socket(zmq.PUB)
zmq_sock.bind(f"tcp://*:{args.port}")
zmq_topic = f"hspc.stream.{args.stream_id}".encode()
stream_frame_idx = 1

# Send calibration block
print(f"[info] Sending calibration block for stream {args.stream_id}")
wavelength_id = send_calibration_block(zmq_sock, wavelengthData, args.stream_id, zmq_topic)
print(f"[info] Wavelength ID: {wavelength_id.hex()}")
print(f"[info] Compression: {'enabled' if args.compress and HAS_LZ4 else 'disabled'}")

# Calibration resend interval (resend every 5 seconds for new clients)
calib_resend_interval = 5.0  # seconds
last_calib_time = time.time()

#generate the craticule data
graticuleData = generateGraticule(wavelengthData)
tens = (graticuleData[0])
fifties = (graticuleData[1])

def snapshot(savedata):
	now = time.strftime("%Y%m%d--%H%M%S")
	timenow = time.strftime("%H:%M:%S")
	imdata1 = savedata[0]
	graphdata = savedata[1]
	if dispWaterfall == True:
		imdata2 = savedata[2]
		cv2.imwrite("waterfall-" + now + ".png",imdata2)
	cv2.imwrite("spectrum-" + now + ".png",imdata1)
	#print(graphdata[0]) #wavelengths
	#print(graphdata[1]) #intensities
	f = open("Spectrum-"+now+'.csv','w')
	f.write('Wavelength,Intensity\r\n')
	for x in zip(graphdata[0],graphdata[1]):
		f.write(str(x[0])+','+str(x[1])+'\r\n')
	f.close()
	message = "Last Save: "+timenow
	return(message)

while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()

	if ret == True:
		y=int((frameHeight/2)-40) #origin of the vertical crop
		#y=200 	#origin of the vert crop
		x=0   	#origin of the horiz crop
		h=80 	#height of the crop
		w=cameraWidth 	#width of the crop - always use full camera width
		cropped = frame[y:y+h, x:x+w]
		bwimage = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
		rows,cols = bwimage.shape
		halfway =int(rows/2)
		#show our line on the original image
		#now a 3px wide region
		cv2.line(cropped,(0,halfway-2),(cameraWidth,halfway-2),(255,255,255),1)
		cv2.line(cropped,(0,halfway+2),(cameraWidth,halfway+2),(255,255,255),1)

		#banner image
		decoded_data = base64.b64decode(background)
		np_data = np.frombuffer(decoded_data,np.uint8)
		img = cv2.imdecode(np_data,3)
		
		# Keep banner at original size (800px), create black background if camera is wider
		if cameraWidth > 800:
			messages = np.zeros([80, cameraWidth, 3], dtype=np.uint8)
			# Center the logo
			offset = (cameraWidth - 800) // 2
			messages[:, offset:offset+800] = img
		elif cameraWidth < 800:
			# If camera is narrower, crop the logo from center
			offset = (800 - cameraWidth) // 2
			messages = img[:, offset:offset+cameraWidth]
		else:
			messages = img

		#blank image for Graph (use displayWidth for graph data)
		graph = np.zeros([320,displayWidth,3],dtype=np.uint8)
		graph.fill(255) #fill white
		
		# If displayWidth != cameraWidth, we'll need to resize the graph later to match camera width for stacking

		#Display a graticule calibrated with cal data
		textoffset = 12
		#vertial lines every whole 10nm
		for position in tens:
			cv2.line(graph,(position,15),(position,320),(200,200,200),1)

		#vertical lines every whole 50nm
		for positiondata in fifties:
			cv2.line(graph,(positiondata[0],15),(positiondata[0],320),(0,0,0),1)
			cv2.putText(graph,str(positiondata[1])+'nm',(positiondata[0]-textoffset,12),font,0.4,(0,0,0),1, cv2.LINE_AA)

		#horizontal lines
		for i in range (320):
			if i>=64:
				if i%64==0: #suppress the first line then draw the rest...
					cv2.line(graph,(0,i),(displayWidth,i),(100,100,100),1)
		
		#Now process the intensity data and display it
		#intensity = []
		for i in range(displayWidth):
			# Map display pixel to actual camera pixel
			camera_pixel = pixel_start + i
			
			#data = bwimage[halfway,i] #pull the pixel data from the halfway mark	
			#print(type(data)) #numpy.uint8
			#average the data of 3 rows of pixels:
			dataminus1 = bwimage[halfway-1,camera_pixel]
			datazero = bwimage[halfway,camera_pixel] #pull the pixel data from the halfway mark
			dataplus1 = bwimage[halfway+1,camera_pixel]
			data = (int(dataminus1)+int(datazero)+int(dataplus1))/3
			data = np.uint8(data)
					
			
			if holdpeaks == True:
				if data > intensity[i]:
					intensity[i] = data
			else:
				intensity[i] = data

		if dispWaterfall == True:
			#waterfall....
			#data is smoothed at this point!!!!!!
			#create an empty array for the data
			wdata = np.zeros([1,displayWidth,3],dtype=np.uint8)
			index=0
			for i in intensity:
				rgb = wavelength_to_rgb(round(wavelengthData[index]))#derive the color from the wavelenthData array
				luminosity = intensity[index]/255
				b = int(round(rgb[0]*luminosity))
				g = int(round(rgb[1]*luminosity))
				r = int(round(rgb[2]*luminosity))
				#print(b,g,r)
				#wdata[0,index]=(r,g,b) #fix me!!! how do we deal with this data??
				wdata[0,index]=(r,g,b)
				index+=1
			waterfall = np.insert(waterfall, 0, wdata, axis=0) #insert line to beginning of array
			waterfall = waterfall[:-1].copy() #remove last element from array

			hsv = cv2.cvtColor(waterfall, cv2.COLOR_BGR2HSV)



		#Draw the intensity data :-)
		#first filter if not holding peaks!
		
		if holdpeaks == False:
			intensity = savitzky_golay(intensity,17,savpoly)
			intensity = np.array(intensity)
			intensity = intensity.astype(int)
			holdmsg = "Holdpeaks OFF" 
		else:
			holdmsg = "Holdpeaks ON"
			
		
		#now draw the intensity data....
		index=0
		for i in intensity:
			rgb = wavelength_to_rgb(round(wavelengthData[index]))#derive the color from the wvalenthData array
			r = rgb[0]
			g = rgb[1]
			b = rgb[2]
			#or some reason origin is top left.
			cv2.line(graph, (index,320), (index,320-i), (b,g,r), 1)
			cv2.line(graph, (index,319-i), (index,320-i), (0,0,0), 1,cv2.LINE_AA)
			index+=1

		# Stream intensity data over ZeroMQ
		try:
			send_intensity_frame(zmq_sock, intensity, wavelength_id, args.stream_id, 
			                     stream_frame_idx, zmq_topic, args.compress)
			stream_frame_idx += 1
			
			# Periodically resend calibration block for new clients
			current_time = time.time()
			if current_time - last_calib_time >= calib_resend_interval:
				send_calibration_block(zmq_sock, wavelengthData, args.stream_id, zmq_topic)
				last_calib_time = current_time
		except Exception as e:
			print(f"[warning] ZeroMQ send failed: {e}")


		#find peaks and label them
		textoffset = 12
		thresh = int(thresh) #make sure the data is int.
		indexes = peakIndexes(intensity, thres=thresh/max(intensity), min_dist=mindist)
		#print(indexes)
		for i in indexes:
			height = intensity[i]
			height = 310-height
			wavelength = round(wavelengthData[i],1)
			cv2.rectangle(graph,((i-textoffset)-2,height),((i-textoffset)+60,height-15),(0,255,255),-1)
			cv2.rectangle(graph,((i-textoffset)-2,height),((i-textoffset)+60,height-15),(0,0,0),1)
			cv2.putText(graph,str(wavelength)+'nm',(i-textoffset,height-3),font,0.4,(0,0,0),1, cv2.LINE_AA)
			#flagpoles
			cv2.line(graph,(i,height),(i,height+10),(0,0,0),1)


		if measure == True:
			#show the cursor!
			cv2.line(graph,(cursorX,cursorY-140),(cursorX,cursorY-180),(0,0,0),1)
			cv2.line(graph,(cursorX-20,cursorY-160),(cursorX+20,cursorY-160),(0,0,0),1)
			cv2.putText(graph,str(round(wavelengthData[cursorX],2))+'nm',(cursorX+5,cursorY-165),font,0.4,(0,0,0),1, cv2.LINE_AA)

		if recPixels == True:
			#display the points
			cv2.line(graph,(cursorX,cursorY-140),(cursorX,cursorY-180),(0,0,0),1)
			cv2.line(graph,(cursorX-20,cursorY-160),(cursorX+20,cursorY-160),(0,0,0),1)
			cv2.putText(graph,str(cursorX)+'px',(cursorX+5,cursorY-165),font,0.4,(0,0,0),1, cv2.LINE_AA)
		else:
			#also make sure the click array stays empty
			clickArray = []

		if clickArray:
			for data in clickArray:
				mouseX=data[0]
				mouseY=data[1]
				cv2.circle(graph,(mouseX,mouseY),5,(0,0,0),-1)
				#we can display text :-) so we can work out wavelength from x-pos and display it ultimately
				cv2.putText(graph,str(mouseX),(mouseX+5,mouseY),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
		

	

		#stack the images and display the spectrum
		# Resize graph to camera width if needed for proper stacking
		if displayWidth != cameraWidth:
			graph_resized = cv2.resize(graph, (cameraWidth, 320), interpolation=cv2.INTER_LINEAR)
		else:
			graph_resized = graph
			
		spectrum_vertical = np.vstack((messages,cropped, graph_resized))
		#dividing lines...
		cv2.line(spectrum_vertical,(0,80),(cameraWidth,80),(255,255,255),1)
		cv2.line(spectrum_vertical,(0,160),(cameraWidth,160),(255,255,255),1)
		#print the messages
		cv2.putText(spectrum_vertical,calmsg1,(490,15),font,0.4,(0,255,255),1, cv2.LINE_AA)
		cv2.putText(spectrum_vertical,calmsg3,(490,33),font,0.4,(0,255,255),1, cv2.LINE_AA)
		cv2.putText(spectrum_vertical,"Framerate: "+str(cfps),(490,51),font,0.4,(0,255,255),1, cv2.LINE_AA)
		cv2.putText(spectrum_vertical,saveMsg,(490,69),font,0.4,(0,255,255),1, cv2.LINE_AA)
		#Second column
		cv2.putText(spectrum_vertical,holdmsg,(640,15),font,0.4,(0,255,255),1, cv2.LINE_AA)
		cv2.putText(spectrum_vertical,"Savgol Filter: "+str(savpoly),(640,33),font,0.4,(0,255,255),1, cv2.LINE_AA)
		cv2.putText(spectrum_vertical,"Label Peak Width: "+str(mindist),(640,51),font,0.4,(0,255,255),1, cv2.LINE_AA)
		cv2.putText(spectrum_vertical,"Label Threshold: "+str(thresh),(640,69),font,0.4,(0,255,255),1, cv2.LINE_AA)
		cv2.imshow(title1,spectrum_vertical)

		if dispWaterfall == True:
			#stack the images and display the waterfall
			# Resize waterfall to camera width if needed for proper stacking
			if displayWidth != cameraWidth:
				waterfall_resized = cv2.resize(waterfall, (cameraWidth, 320), interpolation=cv2.INTER_LINEAR)
			else:
				waterfall_resized = waterfall
				
			waterfall_vertical = np.vstack((messages,cropped, waterfall_resized))
			#dividing lines...
			cv2.line(waterfall_vertical,(0,80),(cameraWidth,80),(255,255,255),1)
			cv2.line(waterfall_vertical,(0,160),(cameraWidth,160),(255,255,255),1)
			#Draw this stuff over the top of the image!
			#Display a graticule calibrated with cal data
			textoffset = 12

			#vertical lines every whole 50nm
			for positiondata in fifties:
				for i in range(162,480):
					if i%20 == 0:
						cv2.line(waterfall_vertical,(positiondata[0],i),(positiondata[0],i+1),(0,0,0),2)
						cv2.line(waterfall_vertical,(positiondata[0],i),(positiondata[0],i+1),(255,255,255),1)
				cv2.putText(waterfall_vertical,str(positiondata[1])+'nm',(positiondata[0]-textoffset,475),font,0.4,(0,0,0),2, cv2.LINE_AA)
				cv2.putText(waterfall_vertical,str(positiondata[1])+'nm',(positiondata[0]-textoffset,475),font,0.4,(255,255,255),1, cv2.LINE_AA)

			cv2.putText(waterfall_vertical,calmsg1,(490,15),font,0.4,(0,255,255),1, cv2.LINE_AA)
			cv2.putText(waterfall_vertical,calmsg2,(490,33),font,0.4,(0,255,255),1, cv2.LINE_AA)
			cv2.putText(waterfall_vertical,calmsg3,(490,51),font,0.4,(0,255,255),1, cv2.LINE_AA)
			cv2.putText(waterfall_vertical,saveMsg,(490,69),font,0.4,(0,255,255),1, cv2.LINE_AA)

			cv2.putText(waterfall_vertical,holdmsg,(640,15),font,0.4,(0,255,255),1, cv2.LINE_AA)

			cv2.imshow(title2,waterfall_vertical)


		keyPress = cv2.waitKey(1)
		if keyPress == ord('q'):
			break
		elif keyPress == ord('h'):
			if holdpeaks == False:
				holdpeaks = True
			elif holdpeaks == True:
				holdpeaks = False
		elif keyPress == ord("s"):
			#package up the data!
			graphdata = []
			graphdata.append(wavelengthData)
			graphdata.append(intensity)
			if dispWaterfall == True:
				savedata = []
				savedata.append(spectrum_vertical)
				savedata.append(graphdata)
				savedata.append(waterfall_vertical)
			else:
				savedata = []
				savedata.append(spectrum_vertical)
				savedata.append(graphdata)
			saveMsg = snapshot(savedata)
		elif keyPress == ord("c"):
			calcomplete = writecal(clickArray, cameraWidth, frameHeight)
			if calcomplete:
				#overwrite wavelength data
				#Go grab the computed calibration data
				caldata = readcal(cameraWidth, frameHeight)
				wavelengthData = caldata[0]
				calmsg1 = caldata[1]
				calmsg2 = caldata[2]
				calmsg3 = caldata[3]
				#overwrite graticule data
				graticuleData = generateGraticule(wavelengthData)
				tens = (graticuleData[0])
				fifties = (graticuleData[1])
				# Resend calibration block immediately with new wavelength data
				wavelength_id = send_calibration_block(zmq_sock, wavelengthData, args.stream_id, zmq_topic)
				last_calib_time = time.time()
				print(f"[info] Calibration updated and sent to clients")
		elif keyPress == ord("x"):
			clickArray = []
		elif keyPress == ord("m"):
			recPixels = False #turn off recpixels!
			if measure == False:
				measure = True
			elif measure == True:
				measure = False
		elif keyPress == ord("p"):
			measure = False #turn off measure!
			if recPixels == False:
				recPixels = True
			elif recPixels == True:
				recPixels = False
		elif keyPress == ord("o"):#sav up
				savpoly+=1
				if savpoly >=15:
					savpoly=15
		elif keyPress == ord("l"):#sav down
				savpoly-=1
				if savpoly <=0:
					savpoly=0
		elif keyPress == ord("i"):#Peak width up
				mindist+=1
				if mindist >=100:
					mindist=100
		elif keyPress == ord("k"):#Peak Width down
				mindist-=1
				if mindist <=0:
					mindist=0
		elif keyPress == ord("u"):#label thresh up
				thresh+=1
				if thresh >=100:
					thresh=100
		elif keyPress == ord("j"):#label thresh down
				thresh-=1
				if thresh <=0:
					thresh=0
	else: 
		break

 
#Everything done, release the vid
cap.release()

cv2.destroyAllWindows()


