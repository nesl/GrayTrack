import pyshark
import matplotlib.pyplot as plt

PCAP_FILE = "/media/ubuntu/research/carla3/2026_03_04_01_22_34_85/videos/camera_1_decrypted.pcap"
SKIP_PACKETS = 210  # <-- change this

cap = pyshark.FileCapture(
    PCAP_FILE,
    display_filter="rtp",
    decode_as={"udp.port==5000": "rtp"}
)

frame_sizes = []
frame_times = []

current_size = 0
start_time = None

count = 0
rtp_marker_count = 0

for pkt in cap:
    count += 1
    if count <= SKIP_PACKETS:
        continue

    try:
        size = int(pkt.length)
        t = float(pkt.sniff_timestamp)
        marker = int(pkt.rtp.marker)
        seq = int(pkt.rtp.seq)
    except:
        continue

    if seq:
        print(f"t: {t}, seq: {seq}")

    if start_time is None:
        start_time = t

    current_size += size

    if marker == 1:
        rtp_marker_count += 1
        frame_sizes.append(current_size)
        frame_times.append(t - start_time)
        current_size = 0

cap.close()

print(f"RTP marker packets: {rtp_marker_count}")

plt.plot(frame_times, frame_sizes)
plt.xlabel("Time (s)")
plt.ylabel("Frame Size (bytes)")
plt.title("Frame Size vs Time")
plt.show()
