import json
import freesound
import os
import argparse

def load_api_key(credentials_path):
    with open(credentials_path, "r") as f:
        data = json.load(f)
    return data["FREESOUND_API_KEY"]

def main():
    parser = argparse.ArgumentParser(description="Download Freesound audio previews.")
    parser.add_argument("-t", "--term", required=True, help="Search term, e.g. 'chatter'")
    parser.add_argument("-o", "--output", required=True, help="Destination folder for downloads")
    parser.add_argument("-c", "--credentials", default="credentials.json", help="Path to credentials JSON file")
    parser.add_argument("-n", "--num_results", type=int, default=50, help="Number of results to fetch (default=50)")

    args = parser.parse_args()

    api_key = load_api_key(args.credentials)

    client = freesound.FreesoundClient()
    client.set_token(api_key)

    # Create destination folder
    os.makedirs(args.output, exist_ok=True)

    print(f"Searching for sounds tagged '{args.term}'...")
    results = client.text_search(query=args.term, page_size=args.num_results)

    for sound in results:
        print(f"Downloading: {sound.name} (ID: {sound.id})")
        filename = f"{sound.id}.wav"
        sound.retrieve_preview(args.output, filename)
        print(f"Saved to {os.path.join(args.output, filename)}")

    print("Done!")

if __name__ == "__main__":
    main()
