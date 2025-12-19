#!/usr/bin/env python3

import argparse
import os
import re

import torch
import torchaudio as ta

from chatterbox import ChatterboxTTS


def parse_srt(srt_file):
    """
    Parse SRT file and extract subtitle entries
    Returns list of dicts with 'id', 'start', 'end', 'text'
    """
    print(f"Parsing SRT file: {srt_file}")

    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to separate subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())

    subtitles = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                # First line is the ID
                subtitle_id = int(lines[0].strip())

                # Second line is the timestamp (we'll keep it but not use it for now)
                timestamp = lines[1].strip()

                # Remaining lines are the text
                text = ' '.join(lines[2:]).strip()

                subtitles.append({
                    'id': subtitle_id,
                    'timestamp': timestamp,
                    'text': text
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping malformed block: {block[:50]}... Error: {e}")
                continue

    print(f"Parsed {len(subtitles)} subtitle entries")
    return subtitles


def find_reference_audio(reference_dir, subtitle_id, audio_prefix='segment'):
    """
    Find the reference audio file for a given subtitle ID
    Supports formats: segment_001.wav, segment_1.wav, etc.
    """
    # Try different naming patterns
    patterns = [
        f"{audio_prefix}_{subtitle_id:03d}.wav",  # segment_001.wav
        f"{audio_prefix}_{subtitle_id:03d}.mp3",
        f"{audio_prefix}_{subtitle_id}.wav",  # segment_1.wav
        f"{audio_prefix}_{subtitle_id}.mp3",
        f"{audio_prefix}{subtitle_id:03d}.wav",  # segment001.wav
        f"{audio_prefix}{subtitle_id:03d}.mp3",
        f"{audio_prefix}{subtitle_id}.wav",  # segment1.wav
        f"{audio_prefix}{subtitle_id}.mp3",
        f"{audio_prefix}_{subtitle_id:03d}.mp4",  # segment_001.mp4
    ]

    for pattern in patterns:
        audio_path = os.path.join(reference_dir, pattern)
        if os.path.exists(audio_path):
            return audio_path

    return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Voice cloning with - Single or Batch mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SINGLE MODE Examples:
  # Clone voice for English text with single reference
  python voice_clone.py --text "Hello world" --reference voice.mp3
  
  # Clone voice for Chinese text
  python voice_clone.py --text "你好世界" --reference voice.mp3 --language ZH

BATCH MODE Examples:
  # Process SRT file with reference audio directory
  python voice_clone.py --srt subtitles.srt --reference-dir ./audio_segments --language EN
  
  # Custom audio file prefix
  python voice_clone.py --srt subtitles.srt --reference-dir ./audio --audio-prefix audio --language ZH
  
  # Process multiple languages
  python voice_clone.py --srt subtitles.srt --reference-dir ./audio --language EN ZH

"""
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--text',
        type=str,
        help='[SINGLE MODE] Text to synthesize'
    )
    mode_group.add_argument(
        '--srt',
        type=str,
        help='[BATCH MODE] Path to SRT subtitle file'
    )

    # Reference audio
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        '--reference',
        type=str,
        help='[SINGLE MODE] Path to reference audio file (the voice to clone)'
    )
    ref_group.add_argument(
        '--reference-dir',
        type=str,
        help='[BATCH MODE] Directory containing reference audio files'
    )

    ref_group.add_argument(
        '--audio-path',
        type=str,
        help='[BATCH MODE] Directory containing reference audio files'
    )

    # Common arguments
    parser.add_argument(
        '--audio-prefix',
        type=str,
        default='segment',
        help='[BATCH MODE] Prefix for audio files (default: segment). Files should be named like segment_001.wav'
    )

    parser.add_argument(
        '--output-prefix',
        type=str,
        default='clone',
        help='[BATCH MODE] Prefix for output files (default: clone). Output will be named like clone_001.wav'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs_v2',
        help='Output directory for generated audio files'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='model',
        help='Path to converter checkpoint'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda:0, cpu, etc.). Auto-detect if not specified'
    )

    return parser.parse_args()


def process_single_mode(args, model):
    """Process single text with single reference audio"""
    print("\n" + "=" * 60)
    print("SINGLE MODE")
    print("=" * 60)

    # Validate reference audio
    if not os.path.exists(args.reference):
        print(f"Error: Reference audio file not found: {args.reference}")
        return

    try:
        # Process each language
        print(f"Text: {args.text}")
        output_filename = "output.wav"
        output_path = os.path.join(args.output, output_filename)

        wav = model.generate(args.text, audio_prompt_path=args.reference)
        ta.save(output_path, wav, model.sr)

        print(f"  ✓ Saved: {output_path}")
    except Exception as e:
        print(f"Error processing text: {str(e)}")


def process_batch_mode(args, model):
    print("\n" + "=" * 60)
    print("BATCH MODE")
    print("=" * 60)

    if not os.path.exists(args.srt):
        print(f"Error: SRT file not found: {args.srt}")
        return

    if args.reference_dir and not os.path.isdir(args.reference_dir) and args.audio_pat and not os.path.exists(
            args.audio_path):
        print(f"Error: Reference directory or path  not found: {args.audio_path} {args.reference_dir}")
        return

    subtitles = parse_srt(args.srt)
    if not subtitles:
        print("Error: No valid subtitles found in SRT file")
        return

    total = len(subtitles)
    success_count = 0
    skip_count = 0
    error_count = 0

    print(f"\nProcessing {total} subtitle entries...")
    print("-" * 60)

    for idx, subtitle in enumerate(subtitles, 1):
        subtitle_id = subtitle['id']
        text = subtitle['text']

        print(f"\n[{idx}/{total}] ID: {subtitle_id}")
        print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        # Find reference audio
        ref_audio = args.audio_path if args.audio_path else find_reference_audio(args.reference_dir,
                                                                                 subtitle_id,
                                                                                 args.audio_prefix)

        if not ref_audio:
            print(f"  ✗ Warning: Reference audio not found for ID {subtitle_id}")
            error_count += 1
            continue
        try:

            output_filename = f"{args.output_prefix}_{subtitle_id:03d}.wav"
            output_path = os.path.join(args.output, output_filename)

            wav = model.generate(subtitle, audio_prompt_path=ref_audio)
            ta.save(output_path, wav, model.sr)

            print(f" ✓ Saved : {output_filename}")

            success_count += 1

        except Exception as e:
            print(f"  ✗ Error processing ID {subtitle_id}: {str(e)}")
            error_count += 1
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total entries:    {total}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (existing):     {skip_count}")
    print(f"Errors:           {error_count}")
    print(f"Success rate:     {success_count / total * 100:.1f}%")


def main():
    """Main execution function"""
    args = parse_args()

    # Determine device
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("Voice Cloning Script - MeloTTS + OpenVoice")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output directory: {args.output}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize model
    model = ChatterboxTTS.from_local(device=device, ckpt_dir=args.checkpoint)
    # Process based on mode
    if args.text:
        process_single_mode(args, model)
    else:
        process_batch_mode(args, model)

    print("\n" + "=" * 60)
    print("✓ PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Output files saved to: {args.output}")


if __name__ == "__main__":
    main()
