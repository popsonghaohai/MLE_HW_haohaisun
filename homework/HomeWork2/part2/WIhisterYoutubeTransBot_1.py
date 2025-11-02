
"""
Whisper Transcription Bot for YouTube Conference Talks
- Downloads audio using yt-dlp
- Transcribes with Whisper
- Extracts text from video frames using Tesseract OCR
- Downloads and parses YouTube subtitles
- Outputs timestamped JSONL
"""

import yt_dlp
import whisper #pip install openai-whisper
import json
from pathlib import Path
from datetime import timedelta
import pytesseract
from PIL import Image
import cv2
from tqdm import tqdm
import time


class WhisperTranscriptionBot:
    def __init__(self, output_dir="whisper_transcripts", whisper_model="base"):
        """
        Initialize Whisper Transcription Bot

        Args:
            output_dir: Output directory for all files
            whisper_model: Whisper model (tiny, base, small, medium, large)
                          - tiny: fastest, least accurate
                          - base: good balance (recommended for 3min talks)
                          - small: better accuracy, slower
                          - medium/large: best accuracy, much slower
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)

        self.video_dir = self.output_dir / "videos"
        self.video_dir.mkdir(exist_ok=True)

        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        self.subtitles_dir = self.output_dir / "subtitles"
        self.subtitles_dir.mkdir(exist_ok=True)

        # Load Whisper model
        print(f"Loading Whisper model '{whisper_model}'...")
        self.model = whisper.load_model(whisper_model)
        print("✓ Whisper model loaded\n")

    def download_audio(self, youtube_url, custom_id=None, download_subtitles=True):
        """
        Download YouTube audio using yt-dlp

        Args:
            youtube_url: YouTube video URL
            custom_id: Custom identifier (optional)
            download_subtitles: Whether to download subtitles

        Returns:
            dict with audio_path, video_id, title, duration, subtitle_path
        """
        print(f"Downloading: {youtube_url}")

        try:
            # yt-dlp options for audio
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.audio_dir / '%(id)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
            }

            # Add subtitle options if requested
            if download_subtitles:
                ydl_opts['writesubtitles'] = True
                ydl_opts['writeautomaticsub'] = True
                ydl_opts['subtitleslangs'] = ['en']
                ydl_opts['subtitlesformat'] = 'srt'
                ydl_opts['skip_download'] = False

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                video_id = custom_id or info['id']
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)

                audio_path = self.audio_dir / f"{info['id']}.mp3"

                print(f"  ✓ Audio: {title}")
                print(f"  ✓ Duration: {duration}s ({duration / 60:.1f} min)")

                # Check for downloaded subtitles
                subtitle_path = None
                if download_subtitles:
                    subtitle_path = self.find_subtitle_file(info['id'])
                    if subtitle_path:
                        print(f"  ✓ Subtitles: {subtitle_path.name}")
                    else:
                        print(f"  - No subtitles available")

            # Download video for frame extraction
            video_opts = {
                'format': 'best[height<=720]',
                'outtmpl': str(self.video_dir / f"{info['id']}.%(ext)s"),
                'quiet': True,
            }

            video_path = None
            try:
                with yt_dlp.YoutubeDL(video_opts) as ydl:
                    ydl.download([youtube_url])
                    video_path = list(self.video_dir.glob(f"{info['id']}.*"))[0]
                    print(f"  ✓ Video downloaded for OCR")
            except:
                print(f"  - Video download skipped (audio only)")

            return {
                'audio_path': audio_path,
                'video_path': video_path,
                'subtitle_path': subtitle_path,
                'video_id': video_id,
                'title': title,
                'duration': duration,
                'url': youtube_url
            }

        except Exception as e:
            print(f"  ✗ Download error: {e}")
            return None

    def find_subtitle_file(self, video_id):
        """
        Find downloaded subtitle file for video

        Args:
            video_id: YouTube video ID

        Returns:
            Path to subtitle file or None
        """
        # Check in audio directory (where yt-dlp downloads them by default)
        for ext in ['.en.srt', '.srt', '.en.vtt', '.vtt']:
            subtitle_path = self.audio_dir / f"{video_id}{ext}"
            if subtitle_path.exists():
                # Move to subtitles directory
                new_path = self.subtitles_dir / subtitle_path.name
                subtitle_path.rename(new_path)
                return new_path

        # Check if already in subtitles directory
        for ext in ['.en.srt', '.srt', '.en.vtt', '.vtt']:
            subtitle_path = self.subtitles_dir / f"{video_id}{ext}"
            if subtitle_path.exists():
                return subtitle_path

        return None

    def parse_srt_subtitles(self, subtitle_path):
        """
        Parse SRT subtitle file

        Args:
            subtitle_path: Path to SRT file

        Returns:
            List of subtitle entries with timestamps and text
        """
        if not subtitle_path or not subtitle_path.exists():
            return []

        print(f"\nParsing subtitles: {subtitle_path.name}")

        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()

            entries = []
            blocks = content.strip().split('\n\n')

            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # Parse timestamp line (e.g., "00:00:01,000 --> 00:00:03,000")
                    timestamp_line = lines[1]
                    if '-->' in timestamp_line:
                        start_str, end_str = timestamp_line.split('-->')
                        start = self.parse_srt_timestamp(start_str.strip())
                        end = self.parse_srt_timestamp(end_str.strip())

                        # Text is everything after timestamp
                        text = ' '.join(lines[2:]).strip()

                        entries.append({
                            'start': start,
                            'end': end,
                            'text': text,
                            'source': 'youtube_subtitle'
                        })

            print(f"  ✓ Parsed {len(entries)} subtitle entries")
            return entries

        except Exception as e:
            print(f"  ✗ Subtitle parsing error: {e}")
            return []

    def parse_srt_timestamp(self, timestamp_str):
        """
        Convert SRT timestamp to seconds

        Args:
            timestamp_str: Timestamp string like "00:01:23,456"

        Returns:
            Float seconds
        """
        # Format: HH:MM:SS,mmm
        timestamp_str = timestamp_str.replace(',', '.')
        parts = timestamp_str.split(':')

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])

        return hours * 3600 + minutes * 60 + seconds

    def transcribe_with_whisper(self, audio_path):
        """
        Transcribe audio using Whisper

        Returns:
            Whisper result with segments and word timestamps
        """
        print(f"\nTranscribing with Whisper...")

        try:
            result = self.model.transcribe(
                str(audio_path),
                language='en',  # Set to None for auto-detection
                word_timestamps=True,  # Get word-level timing
                verbose=False
            )

            print(f"  ✓ Transcribed {len(result['segments'])} segments")
            print(f"  ✓ Language: {result['language']}")

            return result

        except Exception as e:
            print(f"  ✗ Transcription error: {e}")
            return None

    def extract_frames_for_ocr(self, video_path, interval_seconds=30, max_frames=10):
        """
        Extract frames from video at regular intervals for OCR

        Args:
            video_path: Path to video file
            interval_seconds: Seconds between frame captures
            max_frames: Maximum frames to extract

        Returns:
            List of frame info dicts
        """
        if not video_path or not video_path.exists():
            return []

        print(f"\nExtracting frames (every {interval_seconds}s)...")

        try:
            video_id = video_path.stem
            frame_dir = self.frames_dir / video_id
            frame_dir.mkdir(exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval_seconds)

            frames = []
            frame_count = 0
            saved_count = 0

            while saved_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frame_path = frame_dir / f"frame_{int(timestamp):04d}s.png"
                    cv2.imwrite(str(frame_path), frame)

                    frames.append({
                        'path': frame_path,
                        'timestamp': timestamp
                    })
                    saved_count += 1

                frame_count += 1

            cap.release()
            print(f"  ✓ Extracted {len(frames)} frames")

            return frames

        except Exception as e:
            print(f"  ✗ Frame extraction error: {e}")
            return []

    def ocr_frames(self, frames):
        """
        Perform OCR on extracted frames using Tesseract

        Returns:
            List of OCR results with timestamps
        """
        if not frames:
            return []

        print(f"\nPerforming OCR on {len(frames)} frames...")

        ocr_results = []

        for frame_info in frames:
            try:
                image = Image.open(frame_info['path'])

                # Get OCR data with confidence scores
                ocr_data = pytesseract.image_to_data(
                    image,
                    output_type=pytesseract.Output.DICT
                )

                # Filter text by confidence (>60%)
                filtered_words = []
                for i, conf in enumerate(ocr_data['conf']):
                    if int(conf) > 60 and ocr_data['text'][i].strip():
                        filtered_words.append(ocr_data['text'][i])

                text = ' '.join(filtered_words).strip()

                if text:
                    ocr_results.append({
                        'timestamp': frame_info['timestamp'],
                        'text': text,
                        'source': 'ocr'
                    })
                    print(f"  ✓ {frame_info['timestamp']:.1f}s: {text[:50]}...")

            except Exception as e:
                print(f"  ✗ OCR error at {frame_info['timestamp']:.1f}s: {e}")

        print(f"  ✓ OCR complete: {len(ocr_results)} frames with text")
        return ocr_results

    def format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        millis = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def create_jsonl_entries(self, whisper_result, ocr_results, subtitle_entries):
        """
        Create JSONL entries from Whisper + OCR + YouTube subtitles

        Returns:
            List of transcript entries sorted by timestamp
        """
        entries = []

        # Add Whisper segments
        if whisper_result:
            for segment in whisper_result['segments']:
                entry = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'timestamp': self.format_timestamp(segment['start']),
                    'text': segment['text'].strip(),
                    'source': 'whisper'
                }

                # Add word-level timestamps if available
                if 'words' in segment:
                    entry['words'] = [
                        {
                            'word': w['word'],
                            'start': w['start'],
                            'end': w['end'],
                            'probability': w.get('probability', 1.0)
                        }
                        for w in segment['words']
                    ]

                entries.append(entry)

        # Add YouTube subtitle entries
        for subtitle in subtitle_entries:
            entries.append({
                'start': subtitle['start'],
                'end': subtitle['end'],
                'timestamp': self.format_timestamp(subtitle['start']),
                'text': subtitle['text'],
                'source': 'youtube_subtitle'
            })

        # Add OCR results
        for ocr in ocr_results:
            entries.append({
                'start': ocr['timestamp'],
                'end': ocr['timestamp'],
                'timestamp': self.format_timestamp(ocr['timestamp']),
                'text': ocr['text'],
                'source': 'ocr'
            })

        # Sort by start time
        entries.sort(key=lambda x: x['start'])

        return entries

    def save_jsonl(self, entries, output_path):
        """
        Save transcript as JSONL (one JSON object per line)
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        print(f"\n✓ Saved JSONL: {output_path.name}")

    def process_video(self, youtube_url, custom_id=None, extract_frames=True,
                     download_subtitles=True, use_whisper=True):
        """
        Complete pipeline: download → transcribe → OCR → save JSONL

        Args:
            youtube_url: YouTube video URL
            custom_id: Custom identifier for output files
            extract_frames: Whether to extract frames for OCR
            download_subtitles: Whether to download YouTube subtitles
            use_whisper: Whether to use Whisper transcription
        """
        print(f"\n{'=' * 80}")
        print(f"Processing: {youtube_url}")
        print(f"{'=' * 80}\n")

        # Step 1: Download audio (and video if needed)
        download = self.download_audio(youtube_url, custom_id, download_subtitles)
        if not download:
            return None

        # Step 2: Parse YouTube subtitles
        subtitle_entries = []
        if download_subtitles and download.get('subtitle_path'):
            subtitle_entries = self.parse_srt_subtitles(download['subtitle_path'])

        # Step 3: Whisper transcription (optional if subtitles exist)
        whisper_result = None
        if use_whisper:
            whisper_result = self.transcribe_with_whisper(download['audio_path'])
            if not whisper_result:
                return None

        # Step 4: Extract frames and OCR
        ocr_results = []
        if extract_frames and download['video_path']:
            frames = self.extract_frames_for_ocr(download['video_path'])
            ocr_results = self.ocr_frames(frames)

        # Step 5: Create JSONL entries
        entries = self.create_jsonl_entries(whisper_result, ocr_results, subtitle_entries)

        # Step 6: Save JSONL
        jsonl_path = self.output_dir / f"{download['video_id']}.jsonl"
        self.save_jsonl(entries, jsonl_path)

        # Also save metadata
        metadata_path = self.output_dir / f"{download['video_id']}_metadata.json"
        metadata = {
            'video_id': download['video_id'],
            'title': download['title'],
            'duration': download['duration'],
            'url': download['url'],
            'whisper_segments': len(whisper_result['segments']) if whisper_result else 0,
            'subtitle_segments': len(subtitle_entries),
            'ocr_segments': len(ocr_results),
            'total_entries': len(entries)
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"✓ COMPLETE: {download['title']}")
        if whisper_result:
            print(f"  Whisper segments: {len(whisper_result['segments'])}")
        print(f"  YouTube subtitles: {len(subtitle_entries)}")
        print(f"  OCR segments: {len(ocr_results)}")
        print(f"  Total entries: {len(entries)}")
        print(f"{'=' * 80}\n")

        return metadata

    def batch_process(self, video_list, extract_frames=True, download_subtitles=True,
                     use_whisper=True):
        """
        Process multiple videos

        Args:
            video_list: List of URLs or dicts with {'url': ..., 'id': ...}
            extract_frames: Whether to extract frames for OCR
            download_subtitles: Whether to download YouTube subtitles
            use_whisper: Whether to use Whisper transcription
        """
        results = []
        failed = []

        print(f"\n{'=' * 80}")
        print(f"BATCH PROCESSING: {len(video_list)} videos")
        print(f"{'=' * 80}\n")

        for i, video_input in enumerate(video_list, 1):
            # Handle string URL or dict input
            if isinstance(video_input, str):
                url = video_input
                custom_id = None
            else:
                url = video_input['url']
                custom_id = video_input.get('id')

            print(f"\n[{i}/{len(video_list)}]")

            try:
                result = self.process_video(url, custom_id, extract_frames,
                                           download_subtitles, use_whisper)
                if result:
                    results.append(result)
                else:
                    failed.append(url)
            except Exception as e:
                print(f"✗ Error: {e}")
                failed.append(url)

            # Brief pause between videos
            if i < len(video_list):
                time.sleep(2)

        # Generate summary report
        self.save_batch_report(results, failed)

        return results, failed

    def save_batch_report(self, results, failed):
        """Save batch processing summary"""
        report_path = self.output_dir / "batch_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("WHISPER TRANSCRIPTION BATCH REPORT\n")
            f.write("=" * 80 + "\n\n")

            total = len(results) + len(failed)
            f.write(f"Total videos: {total}\n")
            f.write(f"Success: {len(results)}\n")
            f.write(f"Failed: {len(failed)}\n")
            f.write(f"Success rate: {len(results) / total * 100:.1f}%\n\n")

            if results:
                f.write("\nSuccessful Transcriptions:\n")
                f.write("-" * 80 + "\n")
                for r in results:
                    f.write(f"\n{r['video_id']}: {r['title']}\n")
                    f.write(f"  Duration: {r['duration']}s\n")
                    f.write(f"  Whisper: {r['whisper_segments']} segments\n")
                    f.write(f"  YouTube Subtitles: {r['subtitle_segments']} segments\n")
                    f.write(f"  OCR: {r['ocr_segments']} segments\n")
                    f.write(f"  Total: {r['total_entries']} entries\n")

            if failed:
                f.write("\n\nFailed Videos:\n")
                f.write("-" * 80 + "\n")
                for url in failed:
                    f.write(f"  - {url}\n")

        print(f"\n✓ Batch report: {report_path}")


def main():
    """Example usage for 10 NLP conference talks"""

    # Initialize bot
    bot = WhisperTranscriptionBot(
        output_dir="nlp_conference_transcripts",
        whisper_model="base"  # Good for 3-minute talks
    )

    # Example: 10 short NLP conference talks (~3 minutes each)
    # Replace with actual YouTube URLs
    conference_talks = [
        {'url': 'https://www.youtube.com/watch?v=068nfPdtssI', 'id': 'nlp_talk_01'},
        {'url': 'https://www.youtube.com/watch?v=86odbZ0o37M', 'id': 'nlp_talk_02'},
        {'url': 'https://www.youtube.com/watch?v=PJZNtS4gzq4', 'id': 'nlp_talk_03'},
        {'url': 'https://www.youtube.com/watch?v=RBqf25pbVZ0', 'id': 'nlp_talk_04'},
        {'url': 'https://www.youtube.com/watch?v=t_r2VFbQS5Q', 'id': 'nlp_talk_05'},
        {'url': 'https://www.youtube.com/watch?v=hbQyN1Kyk_w', 'id': 'nlp_talk_06'},
        {'url': 'https://www.youtube.com/watch?v=LOkgqMZut7Y', 'id': 'nlp_talk_07'},
        {'url': 'https://www.youtube.com/watch?v=LOkgqMZut7Y', 'id': 'nlp_talk_08'},
        {'url': 'https://www.youtube.com/watch?v=LOkgqMZut7Y', 'id': 'nlp_talk_09'},
        {'url': 'https://www.youtube.com/watch?v=OgB2poD1iNo', 'id': 'nlp_talk_10'},
    ]

    # Process all talks with YouTube subtitles + Whisper + OCR
    results, failed = bot.batch_process(
        conference_talks,
        extract_frames=True,       # Extract frames for OCR
        download_subtitles=True,   # Download YouTube subtitles
        use_whisper=True           # Use Whisper transcription
    )

    # Summary
    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"✓ Success: {len(results)}")
    print(f"✗ Failed: {len(failed)}")
    print(f"\nOutput: {bot.output_dir}")
    print(f"JSONL files: {list(bot.output_dir.glob('*.jsonl'))}")


if __name__ == "__main__":
    main()