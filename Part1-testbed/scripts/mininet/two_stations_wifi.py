import argparse
import glob
import os
import subprocess
import sys
import time
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
from mininet.log import setLogLevel, info

MONITOR_INTERFACE = "hwsim0"

UDP_PORT = 5000
SDP_PATH = "/tmp/stream.sdp"

def write_sdp_file():
    with open(SDP_PATH, "w") as f:
        f.write(
            "v=0\n"
            "o=- 0 0 IN IP4 10.0.0.200\n"
            "s=RTP H264 Stream\n"
            "c=IN IP4 10.0.0.201\n"
            "t=0 0\n"
            f"m=video {UDP_PORT} RTP/AVP 96\n"
            "a=rtpmap:96 H264/90000\n"
        )

def start_ffmpeg_receiver(sta2, log_path):
    sta2.cmd(f"pkill -f '{SDP_PATH}' || true")
    info("*** Starting ffmpeg receiver on sta2\n")
    sta2.cmd(
        "nohup ffmpeg "
        "-loglevel error "
        "-protocol_whitelist file,udp,rtp "
        f"-i {SDP_PATH} "
        "-f null - "
        f"> {log_path} 2>&1 &"
    )


def stop_ffmpeg_receiver(sta2):
    sta2.cmd(f"pkill -f '{SDP_PATH}' || true")


def start_tcpdump_capture(pcap_path, tcpdump_log_path):
    os.makedirs(os.path.dirname(pcap_path), exist_ok=True)
    os.system(f"ifconfig {MONITOR_INTERFACE} up")
    info(f"*** Starting tcpdump capture to {pcap_path}\n")
    log_handle = open(tcpdump_log_path, "a")
    proc = subprocess.Popen(
        [
            "tcpdump",
            "-s", "0",
            "-i", MONITOR_INTERFACE,
            "-w", pcap_path,
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    # give tcpdump a moment to initialize
    time.sleep(0.5)
    return proc, log_handle


def stop_tcpdump_capture(proc, log_handle):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    log_handle.close()


def stream_single_video(sta1, sta2, video_path, tcpdump_log_path):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    pcap_path = os.path.join(os.path.dirname(video_path), f"{basename}.pcap")
    receiver_log = f"/tmp/ffmpeg_sta2_{basename}.log"
    sender_log = f"/tmp/ffmpeg_sta1_{basename}.log"

    tcpdump_proc, tcpdump_log = start_tcpdump_capture(pcap_path, tcpdump_log_path)
    start_ffmpeg_receiver(sta2, receiver_log)
    time.sleep(1)

    info(f"*** Streaming {video_path} -> {pcap_path}\n")
    sta1.cmd(
        "ffmpeg "
        "-re "
        f"-i {video_path} "
        "-c copy "
        "-bsf:v h264_mp4toannexb "
        "-payload_type 96 "
        "-f rtp "
        f"-sdp_file {SDP_PATH} "
        f"rtp://10.0.0.201:{UDP_PORT} "
        f"> {sender_log} 2>&1"
    )

    stop_ffmpeg_receiver(sta2)
    stop_tcpdump_capture(tcpdump_proc, tcpdump_log)
    info(f"*** Completed {video_path}\n")


def stream_all_videos(sta1, sta2, video_paths_array, tcpdump_log_path):
    if not video_paths_array:
        return

    info(f"*** Streaming {len(video_paths_array)} video(s) sequentially\n")
    for video_path in video_paths_array:
        stream_single_video(sta1, sta2, video_path, tcpdump_log_path)


def cleanup_previous_outputs(tcpdump_log_path):
    try:
        os.remove(tcpdump_log_path)
    except FileNotFoundError:
        pass

def wait_associated(sta, ifname):
    timeout_s = 10.0
    interval = 0.1
    waited = 0.0

    info(f"*** Waiting for {sta.name} to be associated...\n")
    while waited < timeout_s:
        out = sta.cmd(f"iw dev {ifname} link")
        if "Connected" in out:
            return
        time.sleep(interval)
        waited += interval
    info(f"*** Warning: {sta.name} not associated after {timeout_s}s\n")

def build_and_run_topology(video_paths_array, tcpdump_log_path):
    setLogLevel('info')
    # Use wmediumd with interference so frames traverse the "air"
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)

    info("*** Creating nodes\n")
    # Positions help Mininet-WiFi decide associations; keep sniffer near the path
    wifi_passwd = '123456780'
    sta1 = net.addStation(
        'sta1',
        ip='10.0.0.200/24',
        position='10,30,0',
        ssid='ssid-wifi',
        passwd=wifi_passwd,
        encrypt='wpa2',
    )
    sta2 = net.addStation(
        'sta2',
        ip='10.0.0.201/24',
        position='12,30,0',
        ssid='ssid-wifi',
        passwd=wifi_passwd,
        encrypt='wpa2',
    )
    ap1 = net.addAccessPoint(
        'ap1',
        ip='10.0.0.1/24',
        ssid='ssid-wifi',
        mode='g',
        channel='1',
        position='20,30,0',
        datapath='user',
        passwd=wifi_passwd,
        encrypt='wpa2',
    )

    info("*** Configuring wifi nodes\n")
    # Optional propagation model (helps emulate distance/attenuation)
    net.setPropagationModel(model="logDistance", exp=4)
    net.configureWifiNodes()

    # Start handshake capture slightly before bringing up the
    # network so we are guaranteed to see the full WPA2 4‑way
    # handshake when stations associate.
    handshake_pcap_path = os.path.join(root_dir, "handshake.pcap")
    handshake_tcpdump_proc, handshake_tcpdump_log = start_tcpdump_capture(
        handshake_pcap_path, tcpdump_log_path
    )

    info("*** Starting network\n")
    net.build()
    net.start()

    # Keep AP IP; not strictly needed for sta1<->sta2, but harmless
    ap1.setIP('10.0.0.1/24', intf='ap1-wlan1')

    # Force sta1 to associate with ap1's SSID
    # sta1.cmd('iwconfig sta1-wlan0 essid ssid-wifi')
    wait_associated(sta1, 'sta1-wlan0')

    # sta2.cmd('iwconfig sta2-wlan0 essid ssid-wifi')
    wait_associated(sta2, 'sta2-wlan0')

    # --- Disable IPv6 ---
    for node in [sta1, sta2, ap1]:
        node.cmd("sysctl -w net.ipv6.conf.all.disable_ipv6=1")
        node.cmd("sysctl -w net.ipv6.conf.default.disable_ipv6=1")

    # OPTIONAL: visualize positions (requires X/GUI). Comment out if headless.
    # net.plotGraph(max_x=100, max_y=100)

    CLI(net)

    # By the time both stations are associated, the 4‑way
    # handshake should be complete; we can stop the handshake
    # capture now. The resulting `handshake.pcap` will live
    # directly in `root_dir`.
    stop_tcpdump_capture(handshake_tcpdump_proc, handshake_tcpdump_log)

    info("*** Starting sequential streaming workload\n")
    stream_all_videos(sta1, sta2, video_paths_array, tcpdump_log_path)

    info("*** Streaming complete.\n")

    info("exit command received")

    info("*** Stopping network\n")
    net.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mininet-WiFi two-station video streaming")
    parser.add_argument(
        "root_dir",
        nargs="?",
        help="Root directory containing <datetime>/videos/*.mp4 structure",
    )
    args = parser.parse_args()
    if not args.root_dir:
        print("error: root_dir is required", file=sys.stderr)
        sys.exit(0)

    root_dir = os.path.abspath(args.root_dir)
    print(f"*** Streaming videos from {root_dir}\n")

    # Folder structure: <datetime string>/videos/*.mp4 — concatenate all MP4 paths into an array, skipping files with "_overhead" in the filename
    video_paths_array = []
    for dt_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if not os.path.isdir(dt_dir):
            continue
        try:
            seed = int(os.path.basename(dt_dir).rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            continue
        # if seed < 78:
        #     continue
        videos_subdir = os.path.join(dt_dir, "videos")
        if os.path.isdir(videos_subdir):
            mp4_files = sorted(glob.glob(os.path.join(videos_subdir, "*.mp4")))
            filtered_files = [f for f in mp4_files if "_overhead" not in os.path.basename(f)]
            video_paths_array.extend(filtered_files)

    print(f"*** Found {len(video_paths_array)} video files\n")

    print(f"writing sdp file {SDP_PATH}...")
    write_sdp_file()

    tcpdump_log_path = os.path.join(root_dir, "tcpdump.log")
    cleanup_previous_outputs(tcpdump_log_path)
    build_and_run_topology(video_paths_array, tcpdump_log_path)

    if os.path.exists(tcpdump_log_path):
        with open(tcpdump_log_path) as f:
            print(f.read())
    else:
        info(f"*** No tcpdump log found at {tcpdump_log_path}\n")

    print("\nDone!")
