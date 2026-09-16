import argparse
import subprocess
import json
import os

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Analyze MP4 frames for type and size.")
parser.add_argument("input_file", help="Path to the input MP4 file")
args = parser.parse_args()

input_file = args.input_file

# --- Infer output file in the same directory as input ---
dir_name = os.path.dirname(os.path.abspath(input_file))  # directory of input file
base_name, _ = os.path.splitext(os.path.basename(input_file))
output_file = os.path.join(dir_name, f"{base_name}_frames.txt")

# --- Run ffprobe to get frame info ---
cmd = [
    "ffprobe",
    "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "frame=pkt_size,pict_type",
    "-of", "json",
    input_file
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("Error running ffprobe:", result.stderr)
    exit(1)

data = json.loads(result.stdout)
frames = data.get("frames", [])

# --- Write output and print I-frames ---
with open(output_file, "w") as f:
    f.write("Frame\tType\tSize(bytes)\n")
    for i, frame in enumerate(frames, start=1):
        if frame['pict_type'] == 'I':
            print(f"I-frame detected: Frame {i}, Time {i * 0.05:.2f}s")
        f.write(f"{i}\t{frame['pict_type']}\t{frame['pkt_size']}\n")

print(f"Frame analysis saved to {output_file}")
