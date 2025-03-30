import cv2
import argparse
import os
import sys

def extract_frame(video_path, frame_number, output_path=None):
    """
    Extract a specific frame from a video and save it as JPG

    Args:
        video_path (str): Path to input video file
        frame_number (int): Frame number to extract (0-based index)
        output_path (str, optional): Path to save output image. Defaults to 'frame_{frame_number}.jpg'

    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False

    if output_path is None:
        output_path = f"frame_{frame_number}.jpg"

    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_number >= total_frames:
            print(f"Error: Frame number {frame_number} exceeds total frames ({total_frames})")
            return False

        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return False

        # Save the frame
        cv2.imwrite(output_path, frame)
        print(f"Successfully saved frame {frame_number} to {output_path}")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    finally:
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract a specific frame from a video')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('frame_number', type=int, help='Frame number to extract (0-based)')
    parser.add_argument('--output', '-o', help='Output image path (default: frame_<number>.jpg)')

    args = parser.parse_args()

    success = extract_frame(args.video_path, args.frame_number, args.output)
    sys.exit(0 if success else 1)
