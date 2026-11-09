import argparse
import sys
from pathlib import Path

from .reader import Reader


def main() -> None:
    parser = argparse.ArgumentParser(description="Read extracted PDF JSON aloud with navigation.")
    parser.add_argument("json_path", type=str, help="Path to the JSON file produced by the extractor.")
    parser.add_argument("--command", type=str, default="speak", choices=[
        "speak",
        "next",
        "prev",
        "next-page",
        "prev-page",
        "list-headings",
        "jump",
    ], help="Action to perform (default: speak current block).")
    parser.add_argument("--text", type=str, help="Text to search for when using the 'jump' command.")
    parser.add_argument("--speed", type=int, help="Set speech rate (words per minute).")
    parser.add_argument("--voice", type=str, help="Select voice by name substring.")
    args = parser.parse_args()

    try:
        reader = Reader(args.json_path)
    except Exception as e:
        print(f"Error loading JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if args.speed:
        reader.set_speed(args.speed)
    if args.voice:
        try:
            reader.set_voice(args.voice)
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    if args.command == "speak":
        reader.speak_current()
    elif args.command == "next":
        reader.next_block()
        reader.speak_current()
    elif args.command == "prev":
        reader.prev_block()
        reader.speak_current()
    elif args.command == "next-page":
        reader.next_page()
        reader.speak_current()
    elif args.command == "prev-page":
        reader.prev_page()
        reader.speak_current()
    elif args.command == "list-headings":
        headings = reader.list_headings()
        for pi, bi, text in headings:
            print(f"Page {pi+1}, block {bi+1}: {text}")
    elif args.command == "jump":
        if not args.text:
            print("--text is required for jump command", file=sys.stderr)
            sys.exit(1)
        try:
            reader.jump_to_heading(args.text)
            reader.speak_current()
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

