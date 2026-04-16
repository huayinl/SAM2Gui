import os
import re
import sys

def identify_missing_frames(mask_folder, output_txt):
    # 1. Extract frame numbers from filenames like '123_mask.png'
    frame_numbers = []
    pattern = re.compile(r'^(\d+)_mask\.png$')

    for filename in os.listdir(mask_folder):
        match = pattern.match(filename)
        if match:
            frame_numbers.append(int(match.group(1)))

    if not frame_numbers:
        print("No valid mask files found.")
        return

    # 2. Sort frames to find gaps chronologically
    frame_numbers.sort()
    
    start_frame = frame_numbers[0]
    end_frame = frame_numbers[-1]
    
    missing_sections = []
    
    # 3. Identify gaps
    # We iterate through the sorted list and check if the next frame is consecutive
    for i in range(len(frame_numbers) - 1):
        current_f = frame_numbers[i]
        next_f = frame_numbers[i + 1]
        
        if next_f > current_f + 1:
            gap_start = current_f + 1
            gap_end = next_f - 1
            missing_sections.append((gap_start, gap_end))

    # 4. Output results
    report_lines = [
        f"FRAME GAP REPORT",
        f"Target Folder: {mask_folder}",
        f"Total Range: {start_frame} to {end_frame}",
        f"Total Frames Present: {len(frame_numbers)}",
        f"Total Gaps Identified: {len(missing_sections)}",
        "-" * 30
    ]

    if missing_sections:
        report_lines.append("MISSING SECTIONS:")
        for start, end in missing_sections:
            gap_str = f"Frames {start} to {end} (Length: {end - start + 1})"
            report_lines.append(f"  - {gap_str}")
            print(f"Gap found: {gap_str}")
    else:
        report_lines.append("No gaps detected. Sequence is continuous.")
        print("No gaps detected.")

    # Write to file
    with open(output_txt, 'w') as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {output_txt}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python identify_gaps.py <mask_directory>")
        print("Example: python identify_gaps.py /Users/huayinluo/Documents/masks/AVG")
        sys.exit(1)
    
    MASK_DIR = sys.argv[1]
    
    if not os.path.exists(MASK_DIR):
        print(f"Error: Directory '{MASK_DIR}' does not exist.")
        sys.exit(1)
    
    REPORT_FILE = os.path.join(MASK_DIR, "missing_frames_report.txt")
    identify_missing_frames(MASK_DIR, REPORT_FILE)

if __name__ == "__main__":
    main()