import sys
import re

def get_lines(filename, start, end):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return ''.join(lines[start:end])

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python get_file_content.py <filename> <start_line> <end_line>")
        sys.exit(1)
    print(get_lines(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
