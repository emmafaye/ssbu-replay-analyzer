import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("log", type=str, help="Path to input image")
args = parser.parse_args()


file = open(args.log)


found_lines = []
absent_lines = []
with open(args.log, 'r') as reader:
    for line in reader.readlines():
        # print(line, end='')
        found_line = re.match("Found", line)
        absent_line = re.match("Absent", line)
        if re.search("Found", line):
            found_lines.append(found_line)
        if re.search("Absent", line):
            absent_lines.append(absent_line)

found = len(found_lines)
absent = len(absent_lines)
total = found + absent
score = round(found / total * 100, 2)

if score > 75:
    score_icon = "🟢"
elif score > 25:
    score_icon = "🟡"
else:
    score_icon = "🔴"

print("💬 Toatal Words:", total, "| ✔️  Total Found:", found, "| ❌ Total Absent:", absent)
print(score_icon, "Score:", str(score) + "%")