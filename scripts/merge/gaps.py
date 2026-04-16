def parse_ranges(filename):
    ranges = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if '-' in line:
                start, end = map(int, line.split('-'))
                ranges.append((start, end))
    return sorted(ranges)

def find_missing_frames(ranges):
    missing_ranges = []
    for i in range(len(ranges) - 1):
        current_end = ranges[i][1]
        next_start = ranges[i+1][0]
        if next_start > current_end + 1:
            missing_ranges.append((current_end + 1, next_start - 1))
    return missing_ranges

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gaps.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    ranges = parse_ranges(filename)
    missing_ranges = find_missing_frames(ranges)
    batch_size = 300
    if missing_ranges:
        print("Missing frame ranges:")
        for start, end in missing_ranges:
            length = end - start + 1
            print(f"{start}-{end} (Length: {length})")
            print(f"needs {length // batch_size} batches of {batch_size} frames each")
    else:
        print("No missing frames.")
    # Print the last end frame
    if ranges:
        print(f"end: {ranges[-1][1]}")