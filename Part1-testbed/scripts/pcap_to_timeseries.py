from pathlib import Path
from scapy.all import rdpcap

# Hardcoded root directory as requested.
ROOT_DIR = Path("/media/ubuntu/Samsung/carla")

def parse_pcap(pcap_path: Path) -> None:
    # Load and decode all packets (smoke test: no downstream processing).
    # Note: for large pcaps this may use significant memory.
    _pkts = rdpcap(str(pcap_path))

    # extract the rtp marker and build tensors for training.

if __name__ == "__main__":
    for pcap_path in sorted(ROOT_DIR.rglob("*.pcap")):
        print(pcap_path)
        parse_pcap(pcap_path)
