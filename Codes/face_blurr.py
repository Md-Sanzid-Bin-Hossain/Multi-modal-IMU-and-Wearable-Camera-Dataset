import os
import cv2
import ffmpeg
import shutil
import pandas as pd
import torch
from google.cloud import storage
from insightface.app import FaceAnalysis

# ============ Configuration ============
bucket_name = "digital-pathology-392716-us-notebooks"
gcs_video_folder = "Subject_9/"  # Ensure this ends with "/"
csv_path = "/home/jupyter/Subject_9.csv"  # CSV with no header: filename, flag

# Extract subject name from folder name
subject_name = os.path.basename(os.path.normpath(gcs_video_folder))
local_temp_folder = f"/home/jupyter/temp_processed_videos"
output_zip_path = f"/home/jupyter/{subject_name}.zip"

# ============ Init ============
os.makedirs(local_temp_folder, exist_ok=True)
client = storage.Client()
bucket = client.bucket(bucket_name)
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else None)
face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

# ============ Load CSV ============
df = pd.read_csv(csv_path, header=None)  # No headers in CSV
df[0] = df[0].astype(str)  # Column 0 = filename
df[1] = df[1].astype(int)  # Column 1 = flag

# ============ Process Videos ============
for idx, row in df.iterrows():
    filename = row[0]
    flag = row[1]
    blob = bucket.blob(os.path.join(gcs_video_folder, filename))
    local_input = os.path.join("/home/jupyter", filename)
    local_output = os.path.join(local_temp_folder, filename)

    print(f"\n🔄 Processing {filename} | Face Flag: {flag}")

    # Download video from GCS
    blob.download_to_filename(local_input)

    if flag == 1:
        print("🟡 Face flag set — blurring faces...")
        cap = cv2.VideoCapture(local_input)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_blur_path = local_output.replace(".mp4", "_temp.mp4")
        out = cv2.VideoWriter(temp_blur_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Optional brightness boost (uncomment if needed)
            # frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

            try:
                faces = face_app.get(frame)
            except Exception as e:
                print(f"⚠️ Face detection failed at frame {frame_idx}: {e}")
                faces = []

            for face in faces:
                x1, y1, x2, y2 = list(map(int, face.bbox))
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, width), min(y2, height)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, (31, 31), 30)
                    frame[y1:y2, x1:x2] = blurred

            out.write(frame)

        cap.release()
        out.release()

        # Re-encode to final output
        ffmpeg_cmd = f"ffmpeg -y -i \"{temp_blur_path}\" -b:v 32M -bufsize 32M -c:v libx264 -preset slow -pix_fmt yuv420p \"{local_output}\""
        os.system(ffmpeg_cmd)
        os.remove(temp_blur_path)

    else:
        print("🟢 No face — removing audio only...")
        try:
            (
                ffmpeg
                .input(local_input)
                .output(local_output, **{'c:v': 'copy', 'an': None, 'b:v': '32M', 'bufsize': '32M'})
                .run(overwrite_output=True)
            )
        except Exception as e:
            print(f"❌ ffmpeg failed to process {filename}: {e}")

    os.remove(local_input)  # Cleanup original file

# ============ Zip and Upload ============
print(f"\n📦 Zipping processed videos into {subject_name}.zip...")
shutil.make_archive(output_zip_path.replace(".zip", ""), 'zip', local_temp_folder)

zip_blob = bucket.blob(os.path.join(gcs_video_folder, f"{subject_name}.zip"))
zip_blob.upload_from_filename(output_zip_path)
print(f"✅ Uploaded ZIP to: gs://{bucket_name}/{gcs_video_folder}{subject_name}.zip")

# ============ Cleanup ============
shutil.rmtree(local_temp_folder)
os.remove(output_zip_path)
print("🧹 Cleanup complete.")
