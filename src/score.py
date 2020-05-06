import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("log", type=str, help="Path to log")
parser.add_argument("-c", "--min-conf")
parser.add_argument("-a", "--max-angle")
parser.add_argument("-n", "--min-nms")
parser.add_argument("-p", "--padding")
parser.add_argument("-l", "--layout")
args = parser.parse_args()

file = open(args.log)

found_lines = []
absent_lines = []
parsed_lines = []
with open(args.log, 'r') as reader:
    for line in reader.readlines():
        found_line = re.match("Found", line)
        absent_line = re.match("Absent", line)
        if re.search("Found", line):
            found_lines.append(found_line)
        if re.search("Absent", line):
            absent_lines.append(absent_line)
            
        parsed_lines.append(line)

found = len(found_lines)
absent = len(absent_lines)
parsed = len(parsed_lines)

if found != 0 and absent != 0:
    total = found + absent
    score = round(found / total * 100, 2)

    if score > 75:
        score_icon = "🟢"
    elif score > 25:
        score_icon = "🟡"
    else:
        score_icon = "🔴"

    print("🧪 Total Tests:", total, "| ✔️  Total Found:", found, "| ❌ Total Absent:", absent, "| 💬 Toatal Words:", parsed)
    print(score_icon, "Score:", str(score) + "%", "| Layout:", args.layout, "| Padding:", args.padding, "| Conf:", args.min_conf, "| NMS:", args.min_nms, "| Angle:", args.max_angle)
else:
    print("⚠️ No Results found, check your test inputs and results.")